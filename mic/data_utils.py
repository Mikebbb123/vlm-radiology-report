"""
R2Gen 数据加载 + 双视图 + 多标签疾病引导 Prompt
"""
import os
import re
import json
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from config import DataConfig


# ============================================================
# 1. 加载数据
# ============================================================

def load_r2gen_data(config: DataConfig) -> Tuple[list, list, list]:
    print(f"[Data] Loading annotation: {config.annotation_file}")
    with open(config.annotation_file) as f:
        data = json.load(f)
    train, val, test = data["train"], data["val"], data["test"]
    print(f"[Data] Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# ============================================================
# 2. 文本清洗
# ============================================================

def clean_report_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\bxxxx\b', '', text)
    for old, new in {"w/": "with", "w/o": "without", "b/l": "bilateral"}.items():
        text = text.replace(old, new)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = text.strip()
    if text and not text.endswith('.'):
        text += '.'
    return text


# ============================================================
# 3. 多标签疾病标签提取 (从 GT 报告, 训练时使用)
# ============================================================

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
                     "no large pleural effusion", "or effusion"],
    },
    "consolidation": {
        "positive": ["consolidation", "consolidative"],
        "negative": ["no consolidation", "no focal consolidation",
                     "without consolidation", "or consolidation"],
    },
    "lung opacity": {
        "positive": ["opacity", "opacit", "infiltrate", "airspace disease"],
        "negative": ["no opacity", "no infiltrate", "no airspace disease",
                     "no focal airspace", "without opacity", "or opacity"],
    },
    "pneumothorax": {
        "positive": ["pneumothorax"],
        "negative": ["no pneumothorax", "without pneumothorax", "or pneumothorax"],
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
        "negative": ["no fracture", "no acute fracture", "without fracture",
                     "negative for fracture"],
    },
    "pleural thickening": {
        "positive": ["pleural thickening"],
        "negative": ["no pleural thickening"],
    },
}


def extract_disease_labels(report: str) -> List[str]:
    """
    从 GT 报告提取疾病标签 (训练时使用)
    区分肯定/否定: "no effusion" ≠ "effusion"
    """
    report_lower = report.lower()
    findings = []

    for disease, patterns in DISEASE_PATTERNS.items():
        has_positive = any(p in report_lower for p in patterns["positive"])
        has_negative = any(n in report_lower for n in patterns["negative"])

        if has_positive and not has_negative:
            findings.append(disease)
        elif has_positive and has_negative:
            # 看哪个先出现
            pos_idx = min(report_lower.find(p) for p in patterns["positive"] if p in report_lower)
            neg_idx = min(report_lower.find(n) for n in patterns["negative"] if n in report_lower)
            if pos_idx < neg_idx:
                findings.append(disease)

    return findings


def disease_findings_to_prompt(findings: List[str]) -> str:
    """疾病发现列表 → prompt hint"""
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


def is_abnormal_report(report: str) -> bool:
    """判断报告是否描述异常 (用于过采样)"""
    return len(extract_disease_labels(report)) > 0


# ============================================================
# 4. Prompt 模板
# ============================================================

SYSTEM_PROMPT = (
    "You are a professional radiologist. "
    "Generate an accurate radiology report in English based on the given chest X-ray images. "
    "You are provided with both frontal and lateral views. "
    "Always respond in English only."
)

USER_PROMPT_BASIC = "Generate a radiology report based on these frontal and lateral chest X-ray images."

USER_PROMPT_VARIANTS = [
    "Generate a radiology report based on these frontal and lateral chest X-ray images.",
    "Describe the findings in these chest X-ray images (frontal and lateral views).",
    "Based on the provided frontal and lateral chest radiographs, write a findings report.",
    "Analyze both the frontal and lateral chest X-ray views and report the findings.",
    "Write a concise radiology report for these two chest X-ray views.",
    "Provide a diagnostic report based on the frontal and lateral chest X-ray images.",
    "Review the frontal and lateral chest radiographs and summarize the key findings.",
]


# ============================================================
# 5. 双视图数据集 (支持过采样)
# ============================================================

class DualViewDataset(Dataset):
    def __init__(self, raw_data: list, images_dir: str, oversample_factor: float = 0.0):
        self.images_dir = images_dir
        self.samples = []
        abnormal_samples = []

        for item in raw_data:
            report = clean_report_text(item.get("report", ""))
            image_paths = item.get("image_path", [])
            if len(report) < 15 or len(image_paths) < 2:
                continue
            frontal = os.path.join(images_dir, image_paths[0])
            lateral = os.path.join(images_dir, image_paths[1])
            if not os.path.exists(frontal) or not os.path.exists(lateral):
                continue
            sample = {
                "id": item.get("id", ""),
                "frontal_path": frontal,
                "lateral_path": lateral,
                "report": report,
            }
            self.samples.append(sample)
            if is_abnormal_report(report):
                abnormal_samples.append(sample)

        if oversample_factor > 0 and abnormal_samples:
            n_extra = int(len(abnormal_samples) * oversample_factor)
            self.samples.extend(random.choices(abnormal_samples, k=n_extra))
            random.shuffle(self.samples)

        n_abnormal = sum(1 for s in self.samples if is_abnormal_report(s["report"]))
        n_total = len(self.samples)
        print(f"[DualViewDataset] {n_total} samples "
              f"(abnormal: {n_abnormal}/{n_total}, {n_abnormal/max(n_total,1)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 6. 双视图 Collator (多标签疾病引导)
# ============================================================

class DualViewCollator:
    """
    训练时 Collator:
      - 从 GT 报告提取多标签疾病发现 → 生成 disease hint
      - hint_dropout: 30% 概率不提供 hint (缩小 train-test gap)
      - prompt_augment: 随机选择 prompt 变体
    """

    def __init__(self, processor,
                 use_disease_hint: bool = True,
                 prompt_augment: bool = False,
                 hint_dropout: float = 0.0,
                 max_length: int = 2048):
        self.processor = processor
        self.use_disease_hint = use_disease_hint
        self.prompt_augment = prompt_augment
        self.hint_dropout = hint_dropout
        self.max_length = max_length

    def __call__(self, features: List[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        texts = []
        all_images = []

        for f in features:
            # 选择 base prompt
            base = random.choice(USER_PROMPT_VARIANTS) if self.prompt_augment else USER_PROMPT_BASIC

            # 多标签疾病 hint
            if self.use_disease_hint:
                if self.hint_dropout > 0 and random.random() < self.hint_dropout:
                    user_text = base
                else:
                    findings = extract_disease_labels(f["report"])
                    hint = disease_findings_to_prompt(findings)
                    user_text = base + " " + hint
            else:
                user_text = base

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{f['frontal_path']}"},
                    {"type": "image", "image": f"file://{f['lateral_path']}"},
                    {"type": "text", "text": user_text},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": f["report"]}]},
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False)
            texts.append(text)
            image_inputs, _ = process_vision_info(messages)
            if image_inputs:
                all_images.extend(image_inputs)

        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

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
# 7. 辅助函数
# ============================================================

def compute_report_statistics(raw_data: list) -> dict:
    lengths = [len(clean_report_text(item.get("report", "")).split()) for item in raw_data]
    if not lengths:
        return {}
    lengths.sort()
    n = len(lengths)
    return {
        "mean": round(sum(lengths) / n, 1),
        "median": lengths[n // 2],
        "p10": lengths[int(n * 0.1)],
        "p90": lengths[int(n * 0.9)],
        "min": min(lengths),
        "max": max(lengths),
    }
