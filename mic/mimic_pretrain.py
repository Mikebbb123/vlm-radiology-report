"""
MIMIC-CXR 纯文本报告预训练
===========================
用 MIMIC-CXR 的 22 万份放射学报告对 Qwen2-VL 的语言侧做 LoRA 预训练,
让模型先学会 "什么是好的放射学报告", 再切回 IU-Xray 做图文联合微调.

使用流程:
  1. 从 PhysioNet 下载 mimic-cxr-reports.zip (135MB) 到 Google Drive
  2. 运行本脚本做语言预训练
  3. 运行原来的 train.py 做 IU-Xray 图文微调 (会自动加载预训练权重)

Usage:
  python mimic_pretrain.py                          # 完整训练
  python mimic_pretrain.py --max_steps 100          # 快速测试
  python mimic_pretrain.py --parse_only             # 只解析报告, 不训练
"""
import os
import re
import json
import glob
import random
import zipfile
import argparse
import gc
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ============================================================
# 配置
# ============================================================

@dataclass
class MIMICPretrainConfig:
    # ----- 路径 -----
    mimic_zip_path: str = "/content/drive/MyDrive/mimic-cxr-reports.zip"
    mimic_parsed_path: str = "/content/drive/MyDrive/mimic_parsed_reports.json"
    pretrain_save_dir: str = "/content/drive/MyDrive/medk_lora_r2gen/lora_mimic_pretrain"

    # ----- 模型 -----
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"

    # ----- LoRA (与 IU-Xray 训练保持一致!) -----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # ----- 训练 -----
    num_epochs: int = 3          # 全量数据, 3 epoch
    batch_size: int = 4          # 纯文本可以用更大 batch
    grad_accum: int = 8          # effective batch = 32
    learning_rate: float = 2e-5  # 预训练阶段 LR 稍高
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 768
    bf16: bool = True
    grad_clip: float = 1.0
    logging_steps: int = 50

    # ----- 数据 -----
    max_reports: int = 0         # 0 = 使用全部报告 (~22万)
    min_findings_words: int = 10 # 过短的 findings 跳过
    min_impression_words: int = 3
    val_ratio: float = 0.02      # 2% 做验证

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


# ============================================================
# 1. 解析 MIMIC-CXR 报告
# ============================================================

def parse_mimic_report(text: str) -> Dict[str, str]:
    """
    解析单份 MIMIC-CXR 报告, 提取 FINDINGS 和 IMPRESSION 部分.

    MIMIC 报告格式:
        FINDINGS:
        The heart is normal in size...

        IMPRESSION:
        No acute cardiopulmonary abnormality.
    """
    sections = {}

    # 常见 section headers
    section_headers = [
        "FINDINGS", "IMPRESSION", "CONCLUSION",
        "INDICATION", "HISTORY", "TECHNIQUE",
        "COMPARISON", "EXAMINATION", "PROCEDURE",
        "RECOMMENDATION", "FINAL REPORT",
    ]

    # 构建正则: 匹配 "FINDINGS:" 或 "FINDINGS :" 等
    pattern = r'(?:^|\n)\s*(' + '|'.join(section_headers) + r')\s*[:\.]?\s*\n?'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    if not matches:
        # 没有明确 section header, 整段作为 findings
        cleaned = text.strip()
        if len(cleaned.split()) >= 5:
            sections["findings"] = cleaned
        return sections

    for i, match in enumerate(matches):
        section_name = match.group(1).upper().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if section_name in ("FINDINGS",):
            sections["findings"] = content
        elif section_name in ("IMPRESSION", "CONCLUSION"):
            sections["impression"] = content

    return sections


def clean_mimic_text(text: str) -> str:
    """清洗 MIMIC 报告文本"""
    text = text.strip()
    text = re.sub(r'___+', '', text)          # 去掉下划线分隔符
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', '', text)  # 去掉 [**de-id**] 标记
    text = re.sub(r'\s+', ' ', text)          # 合并空白
    text = text.strip()
    return text


def parse_all_reports(config: MIMICPretrainConfig) -> List[Dict]:
    """
    从 mimic-cxr-reports.zip 解析所有报告.
    直接从 zip 读取, 不需要解压.
    """
    parsed_path = config.mimic_parsed_path

    # 如果已经解析过, 直接加载
    if os.path.exists(parsed_path):
        print(f"[MIMIC] 加载已解析的报告: {parsed_path}")
        with open(parsed_path) as f:
            reports = json.load(f)
        print(f"[MIMIC] 加载了 {len(reports)} 份报告")
        return reports

    print(f"[MIMIC] 从 zip 解析报告: {config.mimic_zip_path}")

    if not os.path.exists(config.mimic_zip_path):
        raise FileNotFoundError(
            f"找不到 {config.mimic_zip_path}\n"
            f"请从 PhysioNet 下载 mimic-cxr-reports.zip 到 Google Drive"
        )

    reports = []
    skipped = 0

    with zipfile.ZipFile(config.mimic_zip_path, 'r') as zf:
        txt_files = [f for f in zf.namelist() if f.endswith('.txt')]
        print(f"[MIMIC] 找到 {len(txt_files)} 个报告文件")

        for fname in tqdm(txt_files, desc="解析报告"):
            try:
                with zf.open(fname) as f:
                    raw_text = f.read().decode('utf-8', errors='ignore')
            except Exception:
                skipped += 1
                continue

            sections = parse_mimic_report(raw_text)

            findings = clean_mimic_text(sections.get("findings", ""))
            impression = clean_mimic_text(sections.get("impression", ""))

            # 至少需要 findings
            if len(findings.split()) < config.min_findings_words:
                skipped += 1
                continue

            report = {"findings": findings}

            if len(impression.split()) >= config.min_impression_words:
                report["impression"] = impression

            # 完整报告 = findings + impression
            full = findings
            if impression:
                full += " " + impression
            report["full_report"] = full

            reports.append(report)

    print(f"[MIMIC] 解析完成: {len(reports)} 份有效报告, 跳过 {skipped}")

    # 统计
    has_impression = sum(1 for r in reports if "impression" in r)
    avg_findings_len = sum(len(r["findings"].split()) for r in reports) / max(len(reports), 1)
    print(f"[MIMIC] 有 impression: {has_impression}/{len(reports)} "
          f"({has_impression / max(len(reports), 1) * 100:.1f}%)")
    print(f"[MIMIC] 平均 findings 长度: {avg_findings_len:.1f} words")

    # 保存解析结果 (下次直接加载)
    os.makedirs(os.path.dirname(parsed_path), exist_ok=True)
    with open(parsed_path, 'w') as f:
        json.dump(reports, f, ensure_ascii=False)
    print(f"[MIMIC] 已保存到 {parsed_path}")

    return reports


# ============================================================
# 2. 训练任务设计
# ============================================================

# 任务 1: Findings → Impression (最核心的任务)
TASK_F2I_TEMPLATES = [
    {
        "system": "You are a professional radiologist. Generate a concise impression based on the given findings.",
        "user": "Based on the following chest X-ray findings, write the impression:\n\nFindings: {findings}",
        "assistant": "{impression}",
    },
    {
        "system": "You are a professional radiologist. Summarize the key findings into an impression.",
        "user": "Summarize the following radiology findings into a brief impression:\n\n{findings}",
        "assistant": "{impression}",
    },
    {
        "system": "You are a professional radiologist writing radiology reports.",
        "user": "Findings:\n{findings}\n\nWrite the impression section.",
        "assistant": "{impression}",
    },
]

# 任务 2: 报告续写 (给前半段, 生成后半段)
TASK_CONTINUE_TEMPLATES = [
    {
        "system": "You are a professional radiologist. Complete the radiology report based on the beginning.",
        "user": "Complete the following radiology report:\n\n{prefix}",
        "assistant": "{suffix}",
    },
    {
        "system": "You are a professional radiologist writing chest X-ray reports.",
        "user": "Continue writing this chest X-ray report:\n\n{prefix}",
        "assistant": "{suffix}",
    },
]

# 任务 3: 疾病关键词 → 报告 (与你的 disease hint 策略对齐!)
TASK_DISEASE2REPORT_TEMPLATES = [
    {
        "system": (
            "You are a professional radiologist. "
            "Generate an accurate radiology report in English based on the given chest X-ray images. "
            "Always respond in English only."
        ),
        "user": (
            "Generate a radiology report based on these frontal and lateral chest X-ray images. "
            "{hint}"
        ),
        "assistant": "{report}",
    },
]

# 从报告中提取关键词 (复用你的 disease pattern)
DISEASE_PATTERNS = {
    "cardiomegaly": {
        "positive": ["cardiomegaly", "heart is enlarged", "heart is mildly enlarged",
                     "enlarged heart", "cardiac enlargement", "increased cardiac"],
        "negative": ["no cardiomegaly", "heart is normal", "normal heart",
                     "heart size is normal", "normal in size"],
    },
    "pleural effusion": {
        "positive": ["effusion", "effusions", "costophrenic blunting"],
        "negative": ["no effusion", "no pleural effusion", "without effusion",
                     "no large pleural effusion"],
    },
    "consolidation": {
        "positive": ["consolidation", "consolidative"],
        "negative": ["no consolidation", "no focal consolidation", "without consolidation"],
    },
    "lung opacity": {
        "positive": ["opacity", "opacit", "infiltrate", "airspace disease"],
        "negative": ["no opacity", "no infiltrate", "no airspace disease",
                     "no focal airspace", "without opacity"],
    },
    "pneumothorax": {
        "positive": ["pneumothorax"],
        "negative": ["no pneumothorax", "without pneumothorax"],
    },
    "pulmonary edema": {
        "positive": ["edema", "congestion", "cephalization"],
        "negative": ["no edema", "no congestion", "no pulmonary edema"],
    },
    "atelectasis": {
        "positive": ["atelectasis", "atelectatic"],
        "negative": ["no atelectasis"],
    },
    "pneumonia": {
        "positive": ["pneumonia"],
        "negative": ["no pneumonia", "without pneumonia"],
    },
    "fracture": {
        "positive": ["fracture"],
        "negative": ["no fracture", "no acute fracture"],
    },
    "pleural thickening": {
        "positive": ["pleural thickening"],
        "negative": ["no pleural thickening"],
    },
}


def extract_diseases(text: str) -> List[str]:
    """从报告文本提取疾病标签"""
    text_lower = text.lower()
    findings = []
    for disease, patterns in DISEASE_PATTERNS.items():
        has_pos = any(p in text_lower for p in patterns["positive"])
        has_neg = any(n in text_lower for n in patterns["negative"])
        if has_pos and not has_neg:
            findings.append(disease)
    return findings


def make_disease_hint(findings: List[str]) -> str:
    """疾病列表 → hint 文本 (与你的 disease_findings_to_prompt 一致)"""
    if not findings:
        return ("No significant abnormalities detected. "
                "Report the normal findings for each anatomical region.")
    elif len(findings) == 1:
        return (f"Findings suggest: {findings[0]}. "
                f"Report this finding along with other observations.")
    else:
        findings_str = ", ".join(findings[:-1]) + f", and {findings[-1]}"
        return (f"Findings suggest: {findings_str}. "
                f"Report these findings along with other observations.")


def create_training_samples(reports: List[Dict], config: MIMICPretrainConfig) -> List[Dict]:
    """
    从 MIMIC 报告创建训练样本.

    任务分配:
      - 有 impression 的报告: 60% findings→impression, 20% 续写, 20% disease→report
      - 没有 impression 的报告: 50% 续写, 50% disease→report
    """
    samples = []

    # 采样 (max_reports=0 表示全部使用)
    if config.max_reports > 0 and len(reports) > config.max_reports:
        # 优先选有 impression 的
        with_imp = [r for r in reports if "impression" in r]
        without_imp = [r for r in reports if "impression" not in r]

        n_with = min(len(with_imp), int(config.max_reports * 0.7))
        n_without = min(len(without_imp), config.max_reports - n_with)

        selected = random.sample(with_imp, n_with) + random.sample(without_imp, n_without)
        random.shuffle(selected)
    else:
        selected = reports

    print(f"[Tasks] 从 {len(reports)} 份报告中选择 {len(selected)} 份创建训练样本")

    for report in selected:
        findings = report["findings"]
        impression = report.get("impression", "")
        full_report = report["full_report"]

        # ---- 任务 1: Findings → Impression ----
        if impression and len(impression.split()) >= 3:
            template = random.choice(TASK_F2I_TEMPLATES)
            samples.append({
                "system": template["system"],
                "user": template["user"].format(findings=findings),
                "assistant": template["assistant"].format(impression=impression),
                "task": "f2i",
            })

        # ---- 任务 2: 报告续写 ----
        words = full_report.split()
        if len(words) >= 15:
            # 在 40%-70% 位置切分
            split_point = random.randint(int(len(words) * 0.4), int(len(words) * 0.7))
            prefix = " ".join(words[:split_point])
            suffix = " ".join(words[split_point:])

            if len(suffix.split()) >= 5:
                template = random.choice(TASK_CONTINUE_TEMPLATES)
                samples.append({
                    "system": template["system"],
                    "user": template["user"].format(prefix=prefix),
                    "assistant": template["assistant"].format(suffix=suffix),
                    "task": "continue",
                })

        # ---- 任务 3: Disease hint → Report ----
        diseases = extract_diseases(full_report)
        hint = make_disease_hint(diseases)
        template = random.choice(TASK_DISEASE2REPORT_TEMPLATES)
        samples.append({
            "system": template["system"],
            "user": template["user"].format(hint=hint),
            "assistant": template["assistant"].format(report=full_report),
            "task": "disease2report",
        })

    random.shuffle(samples)

    # 统计
    task_counts = {}
    for s in samples:
        task_counts[s["task"]] = task_counts.get(s["task"], 0) + 1
    print(f"[Tasks] 创建了 {len(samples)} 个训练样本:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} ({count / len(samples) * 100:.1f}%)")

    return samples


# ============================================================
# 3. 纯文本 Dataset
# ============================================================

class TextOnlyDataset(Dataset):
    """纯文本训练数据集 (无图像)"""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TextOnlyCollator:
    """纯文本 Collator - 构造 Qwen chat 格式 (无图像)"""

    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict:
        texts = []

        for f in features:
            messages = [
                {"role": "system", "content": f["system"]},
                {"role": "user", "content": f["user"]},
                {"role": "assistant", "content": f["assistant"]},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False
            )
            texts.append(text)

        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 只对 assistant 部分计算 loss
        labels = inputs["input_ids"].clone()
        assistant_token_ids = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        trigger_len = len(assistant_token_ids)

        for i in range(labels.shape[0]):
            ids = inputs["input_ids"][i].tolist()
            found = False
            for j in range(len(ids) - trigger_len - 1, -1, -1):
                if ids[j:j + trigger_len] == assistant_token_ids:
                    labels[i, :j + trigger_len] = -100
                    found = True
                    break
            if not found:
                labels[i, :len(ids) // 2] = -100
            labels[i][inputs["attention_mask"][i] == 0] = -100

        inputs["labels"] = labels
        return inputs


# ============================================================
# 4. 训练
# ============================================================

def run_pretrain(config: MIMICPretrainConfig, args):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import get_cosine_schedule_with_warmup

    # ---- 解析报告 ----
    reports = parse_all_reports(config)

    if args.parse_only:
        print("[Done] 报告解析完成, 退出")
        return

    # ---- 创建训练样本 ----
    samples = create_training_samples(reports, config)

    # ---- 训练/验证 切分 ----
    n_val = max(int(len(samples) * config.val_ratio), 50)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"[Data] Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_dataset = TextOnlyDataset(train_samples)
    val_dataset = TextOnlyDataset(val_samples)

    # ---- 加载模型 ----
    print(f"[Model] Loading {config.model_name}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] Trainable: {trainable / 1e6:.1f}M / {total / 1e9:.2f}B "
          f"({trainable / total * 100:.2f}%)")

    device = next(model.parameters()).device

    # ---- Collator ----
    collator = TextOnlyCollator(processor, max_length=config.max_length)

    # ---- Optimizer & Scheduler ----
    steps_per_epoch = math.ceil(len(train_dataset) / (config.batch_size * config.grad_accum))
    total_steps = steps_per_epoch * config.num_epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    # ---- 训练信息 ----
    print(f"\n{'=' * 70}")
    print(f" MIMIC-CXR 语言预训练")
    print(f" Model: {config.model_name}")
    print(f" Reports: {len(reports)} → Samples: {len(train_samples)}")
    print(f" Epochs: {config.num_epochs} | Total steps: {total_steps}")
    print(f" Batch: {config.batch_size} x {config.grad_accum} = {config.batch_size * config.grad_accum}")
    print(f" LR: {config.learning_rate} | LoRA rank: {config.lora_r}")
    print(f" Tasks: findings→impression / report continuation / disease→report")
    print(f"{'=' * 70}\n")

    # ---- 训练循环 ----
    model.train()
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        indices = torch.randperm(len(train_dataset)).tolist()
        epoch_loss, epoch_count = 0, 0
        optimizer.zero_grad()

        pbar = tqdm(
            range(0, len(indices), config.batch_size),
            desc=f"Epoch {epoch + 1}/{config.num_epochs}"
        )

        for step_idx, start_idx in enumerate(pbar):
            if args.max_steps and global_step >= args.max_steps:
                break

            batch_indices = indices[start_idx:start_idx + config.batch_size]
            batch_data = [train_dataset[i] for i in batch_indices]

            try:
                batch = collator(batch_data)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / config.grad_accum
                loss.backward()
                epoch_loss += outputs.loss.item()
                epoch_count += 1
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                if epoch_count < 5 or epoch_count % 500 == 0:
                    print(f"\n  [Error] step_idx={step_idx}, error: {type(e).__name__}: {e}")
                continue

            if (step_idx + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = epoch_loss / max(epoch_count, 1)
                lr = scheduler.get_last_lr()[0]
                success_rate = epoch_count / max(step_idx + 1, 1) * 100
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}",
                    step=global_step, ok=f"{success_rate:.0f}%"
                )

                if global_step % config.logging_steps == 0:
                    print(f"  Step {global_step}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                          f"Success: {epoch_count}/{step_idx+1} ({success_rate:.0f}%)")

        if args.max_steps and global_step >= args.max_steps:
            break

        # ---- 验证 ----
        val_loss = _eval_val(model, val_dataset, collator, device, config.batch_size)
        train_avg = epoch_loss / max(epoch_count, 1)
        print(f"\n  Epoch {epoch + 1} | Train: {train_avg:.4f} | Val: {val_loss:.4f}")

        # ---- 保存 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config.pretrain_save_dir, exist_ok=True)
            model.save_pretrained(config.pretrain_save_dir)
            processor.save_pretrained(config.pretrain_save_dir)
            print(f"  ✅ Best model saved to {config.pretrain_save_dir}")
        else:
            print(f"  ⚠️ Val loss did not improve ({val_loss:.4f} >= {best_val_loss:.4f})")

    # ---- 确保最终保存 ----
    os.makedirs(config.pretrain_save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(config.pretrain_save_dir, "adapter_config.json")):
        model.save_pretrained(config.pretrain_save_dir)
        processor.save_pretrained(config.pretrain_save_dir)

    print(f"\n{'=' * 70}")
    print(f" 预训练完成!")
    print(f" LoRA 权重: {config.pretrain_save_dir}")
    print(f" Best val loss: {best_val_loss:.4f}")
    print(f"")
    print(f" 下一步: 运行 train.py 做 IU-Xray 图文微调")
    print(f" (使用 --pretrain_path {config.pretrain_save_dir})")
    print(f"{'=' * 70}")


@torch.no_grad()
def _eval_val(model, val_dataset, collator, device, batch_size):
    model.eval()
    total_loss, count = 0, 0
    for i in range(0, len(val_dataset), batch_size):
        batch_data = [val_dataset[j] for j in range(i, min(i + batch_size, len(val_dataset)))]
        try:
            batch = collator(batch_data)
            batch = {k: v.to(device) for k, v in batch.items()}
            total_loss += model(**batch).loss.item()
            count += 1
        except Exception:
            continue
    model.train()
    return total_loss / max(count, 1)


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MIMIC-CXR 语言预训练")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最大训练步数 (快速测试用)")
    parser.add_argument("--max_reports", type=int, default=None,
                        help="最多使用多少份报告 (0=全部)")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--parse_only", action="store_true",
                        help="只解析报告, 不训练")
    parser.add_argument("--mimic_zip", type=str, default=None,
                        help="mimic-cxr-reports.zip 路径")
    args = parser.parse_args()

    config = MIMICPretrainConfig()

    if args.mimic_zip:
        config.mimic_zip_path = args.mimic_zip
    if args.max_reports is not None:
        config.max_reports = args.max_reports
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    random.seed(42)
    run_pretrain(config, args)


if __name__ == "__main__":
    main()