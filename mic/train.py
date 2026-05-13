"""
Disease-Aware VLM 训练 (支持 MIMIC 预训练权重)

Usage:
    # 方式 1: 从 MIMIC 预训练权重继续训练 (推荐)
    python train.py --pretrain_path /content/drive/MyDrive/medk_lora_r2gen/lora_mimic_pretrain

    # 方式 2: 从头训练 (原来的方式)
    python train.py

    # 方式 3: 从头训练 (忽略之前的 checkpoint)
    python train.py --reset

    # 快速测试
    python train.py --pretrain_path ... --max_steps 50

完整流程:
    1. python mimic_pretrain.py          # MIMIC 语言预训练
    2. python train.py --pretrain_path /content/drive/MyDrive/medk_lora_r2gen/lora_mimic_pretrain
    3. python evaluate.py                # 评估
"""
import os
import json
import argparse
import torch
import warnings
import gc
import math
from tqdm import tqdm

from peft import get_peft_model, PeftModel
from transformers import get_cosine_schedule_with_warmup

from config import TrainingConfig, DataConfig, LoRAConfig
from data_utils import (
    load_r2gen_data,
    compute_report_statistics,
    clean_report_text,
    DualViewDataset,
    DualViewCollator,
    USER_PROMPT_BASIC,
    SYSTEM_PROMPT,
)
from model import load_vlm_model, create_lora_config

warnings.filterwarnings("ignore")

CHECKPOINT_FILE = "train_state.json"


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


def save_checkpoint(save_dir, epoch, global_step, best_val_loss, patience_counter, optimizer, scheduler):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, CHECKPOINT_FILE), "w") as f:
        json.dump({
            "epoch": epoch, "global_step": global_step,
            "best_val_loss": best_val_loss, "patience_counter": patience_counter,
        }, f, indent=2)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, os.path.join(save_dir, "optim_state.pt"))


def load_checkpoint(save_dir):
    path = os.path.join(save_dir, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    optim_path = os.path.join(save_dir, "optim_state.pt")
    state["optim_state"] = torch.load(optim_path, map_location="cpu", weights_only=True) if os.path.exists(optim_path) else None
    return state


def build_dual_view_messages(frontal_path, lateral_path, user_text):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{frontal_path}"},
            {"type": "image", "image": f"file://{lateral_path}"},
            {"type": "text", "text": user_text},
        ]},
    ]


@torch.no_grad()
def quick_generation_check(model, processor, val_dataset, device, num_samples=3):
    """每个 epoch 结束后检查生成多样性"""
    from qwen_vl_utils import process_vision_info

    model.eval()
    results = []

    for idx in range(min(num_samples, len(val_dataset))):
        sample = val_dataset[idx]
        messages = build_dual_view_messages(
            sample["frontal_path"], sample["lateral_path"], USER_PROMPT_BASIC
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        try:
            out = model.generate(
                **inputs, max_new_tokens=80, num_beams=2,
                repetition_penalty=1.2, do_sample=False,
            )
            generated = processor.decode(out[0][input_len:], skip_special_tokens=True)
        except:
            generated = "[OOM]"

        results.append({"id": sample["id"], "ref": sample["report"][:100], "gen": generated[:100]})

    model.train()

    gen_texts = [r["gen"] for r in results if r["gen"] != "[OOM]"]
    if len(gen_texts) >= 2:
        unique_ratio = len(set(gen_texts)) / len(gen_texts)
        print(f"\n  [Gen Check] Diversity: {unique_ratio:.2f} ({len(set(gen_texts))}/{len(gen_texts)} unique)")
        if unique_ratio < 0.5:
            print("  ⚠️ WARNING: 可能出现模式坍塌！")
    for r in results:
        print(f"  [{r['id']}] Ref: {r['ref'][:80]}...")
        print(f"  [{r['id']}] Gen: {r['gen'][:80]}...")

    return results


# ============================================================
# VLM LoRA 训练
# ============================================================

def run_training(args, train_config, data_config, lora_config):
    save_dir = os.path.join(train_config.output_dir, "lora_discourse")

    checkpoint = None if args.reset else load_checkpoint(save_dir)
    if checkpoint:
        print(f"[Resume] Epoch {checkpoint['epoch']+1}, Step {checkpoint['global_step']}, "
              f"Best val loss: {checkpoint['best_val_loss']:.4f}")
    else:
        print("[Train] 从头开始训练")

    raw_train, raw_val, _ = load_r2gen_data(data_config)
    stats = compute_report_statistics(raw_train)
    print(f"[Train] Train: {len(raw_train)} | Val: {len(raw_val)}")
    print(f"[Train] Report stats: {stats}")

    print("[Train] Loading VLM...")
    model, processor = load_vlm_model(train_config)
    device = next(model.parameters()).device

    # ============================================================
    # LoRA 加载逻辑 (新增 MIMIC 预训练支持)
    # ============================================================
    if checkpoint and os.path.exists(save_dir):
        # 情况 1: 从之前的 IU-Xray 训练 checkpoint 恢复
        print(f"[Resume] Loading LoRA from {save_dir}")
        model = PeftModel.from_pretrained(model, save_dir, is_trainable=True)

    elif args.pretrain_path and os.path.exists(args.pretrain_path):
        # 情况 2: 从 MIMIC 预训练权重初始化 LoRA
        print(f"[Init] Loading MIMIC pre-trained LoRA from {args.pretrain_path}")
        model = PeftModel.from_pretrained(model, args.pretrain_path, is_trainable=True)

        # 验证 LoRA 配置兼容性
        pretrain_config_path = os.path.join(args.pretrain_path, "adapter_config.json")
        if os.path.exists(pretrain_config_path):
            with open(pretrain_config_path) as f:
                pretrain_lora_cfg = json.load(f)
            print(f"  Pre-trained LoRA: r={pretrain_lora_cfg.get('r')}, "
                  f"alpha={pretrain_lora_cfg.get('lora_alpha')}, "
                  f"modules={pretrain_lora_cfg.get('target_modules')}")

        print(f"[Init] ✅ MIMIC 预训练权重加载成功, 将在此基础上微调")

    else:
        # 情况 3: 从头创建 LoRA
        if args.pretrain_path:
            print(f"[Warn] 预训练路径不存在: {args.pretrain_path}")
            print(f"[Warn] 将从头创建 LoRA")
        lora_cfg = create_lora_config(train_config.discourse_lora_r, lora_config.target_modules)
        model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable: {trainable / 1e6:.1f}M")

    # 过采样异常样本 2 倍
    train_dataset = DualViewDataset(raw_train, data_config.images_dir, oversample_factor=0.0)
    val_dataset = DualViewDataset(raw_val, data_config.images_dir, oversample_factor=0.0)

    # 训练 collator: 多标签疾病 hint + prompt 多样化 + 30% hint dropout
    collator = DualViewCollator(
        processor,
        use_disease_hint=True,
        prompt_augment=True,
        hint_dropout=0.3,
        max_length=2048,
    )
    val_collator = DualViewCollator(
        processor,
        use_disease_hint=True,
        prompt_augment=False,
        hint_dropout=0.0,
        max_length=2048,
    )

    batch_size = train_config.discourse_batch_size
    grad_accum = train_config.discourse_grad_accum
    num_epochs = train_config.discourse_epochs
    steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * grad_accum))
    total_steps = steps_per_epoch * num_epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    # ============================================================
    # 学习率策略: 如果从预训练加载, 用更小的 LR
    # ============================================================
    lr = train_config.learning_rate
    if args.pretrain_path and not checkpoint:
        # 从预训练初始化时, 降低 LR 避免灾难性遗忘
        lr = train_config.learning_rate * 0.5
        print(f"[Train] 使用预训练初始化, LR 降低: {train_config.learning_rate} → {lr}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=train_config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * train_config.warmup_ratio),
        num_training_steps=total_steps,
    )

    start_epoch, global_step, best_val_loss, patience_counter = 0, 0, float("inf"), 0
    if checkpoint:
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint["patience_counter"]
        if checkpoint.get("optim_state"):
            try:
                optimizer.load_state_dict(checkpoint["optim_state"]["optimizer"])
                scheduler.load_state_dict(checkpoint["optim_state"]["scheduler"])
                print("[Resume] Optimizer restored")
            except Exception as e:
                print(f"[Resume] Optimizer restore failed: {e}")

    if start_epoch >= num_epochs:
        print(f"[Train] 已完成 {num_epochs} epochs，用 --reset 重新训练")
        return

    init_mode = "MIMIC Pre-trained" if (args.pretrain_path and not checkpoint) else \
                "Resumed" if checkpoint else "Random Init"

    print(f"\n{'='*70}")
    print(f" VLM LoRA Training (Disease-Aware Prompt Guidance)")
    print(f" Model: {train_config.model_name}")
    print(f" Init: {init_mode}")
    if args.pretrain_path and not checkpoint:
        print(f" Pre-train: {args.pretrain_path}")
    print(f" Data: R2Gen (train={len(train_dataset)}, val={len(val_dataset)})")
    print(f" Epochs: {start_epoch+1}..{num_epochs}, LR: {lr}")
    print(f" LoRA rank: {train_config.discourse_lora_r}, grad_accum: {grad_accum}")
    print(f" Disease hint: ON (multi-label) | Dropout: 0.3 | Oversample: 2x")
    print(f"{'='*70}\n")

    model.train()
    patience = 5

    for epoch in range(start_epoch, num_epochs):
        indices = torch.randperm(len(train_dataset)).tolist()
        epoch_loss, epoch_count = 0, 0
        optimizer.zero_grad()

        pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")

        for step_idx, start_idx in enumerate(pbar):
            if args.max_steps and global_step >= args.max_steps:
                break

            batch_data = [train_dataset[indices[j]] for j in range(start_idx, min(start_idx + batch_size, len(indices)))]

            try:
                batch = collator(batch_data)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
                loss.backward()
                epoch_loss += outputs.loss.item()
                epoch_count += 1
            except torch.cuda.OutOfMemoryError:
                cleanup_memory()
                optimizer.zero_grad()
                continue
            except Exception as e:
                print(f"[Error] {e}")
                continue

            if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(indices) // batch_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = epoch_loss / max(epoch_count, 1)
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}", step=global_step)

                if global_step % train_config.logging_steps == 0:
                    print(f"  Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")

        if args.max_steps and global_step >= args.max_steps:
            break

        val_loss = _eval_val(model, val_dataset, val_collator, device, batch_size)
        print(f"\n  Epoch {epoch+1} | Train: {epoch_loss/max(epoch_count,1):.4f} | Val: {val_loss:.4f}")

        quick_generation_check(model, processor, val_dataset, device, num_samples=3)

        os.makedirs(save_dir, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            save_checkpoint(save_dir, epoch, global_step, best_val_loss, patience_counter, optimizer, scheduler)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            save_checkpoint(save_dir, epoch, global_step, best_val_loss, patience_counter, optimizer, scheduler)
            print(f"  ⚠️ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n[Train] Done! Best val loss: {best_val_loss:.4f}")
    print(f"[Train] Next: python evaluate.py")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default=None,
                        help="MIMIC 预训练 LoRA 权重路径 (mimic_pretrain.py 的输出)")
    args = parser.parse_args()

    train_config = TrainingConfig()
    data_config = DataConfig()
    lora_config = LoRAConfig()

    os.makedirs(train_config.output_dir, exist_ok=True)

    lora_dir = os.path.join(train_config.output_dir, "lora_discourse")
    pretrain_dir = args.pretrain_path or ""
    print("[Main] 训练状态:")
    print(f"  MIMIC 预训练: {'✅' if os.path.exists(pretrain_dir) else '⬜'} {pretrain_dir}")
    print(f"  VLM LoRA:    {'✅' if os.path.exists(lora_dir) else '⬜'} {lora_dir}")

    run_training(args, train_config, data_config, lora_config)


if __name__ == "__main__":
    main()
