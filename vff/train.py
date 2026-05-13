"""
Disease-Aware VLM 训练

Usage:
    python train.py                        # 训练 VLM LoRA
    python train.py --max_steps 50         # 快速测试
    python train.py --reset                # 从头训练

注意: 不再需要 stage 2!
  推理时的疾病分类器用 torchxrayvision 预训练权重, 零训练成本
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
    load_cat_hints,
    load_densenet_feats,
    USER_PROMPT_BASIC,
    SYSTEM_PROMPT,
)
from model import load_vlm_model, create_lora_config
from visual_feature_fusion import VisualFeatureFusion, install_vff_hook

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
    # 根据 hint_mode / use_vff 决定 checkpoint 目录
    if args.use_vff:
        save_dir = os.path.join(train_config.output_dir, "lora_vff")
    else:
        save_dir_map = {
            "gt":    os.path.join(train_config.output_dir, "lora_discourse"),
            "cat":   os.path.join(train_config.output_dir, "lora_cat"),
            "mixed": os.path.join(train_config.output_dir, "lora_mixed"),
        }
        save_dir = save_dir_map[args.hint_mode]

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

    if checkpoint and os.path.exists(save_dir):
        print(f"[Resume] Loading LoRA from {save_dir}")
        model = PeftModel.from_pretrained(model, save_dir, is_trainable=True)
    else:
        lora_cfg = create_lora_config(train_config.discourse_lora_r, lora_config.target_modules)
        model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable: {trainable / 1e6:.1f}M")

    # VFF 模式关闭过采样 (因为过采样用 is_abnormal 判断, 和 VFF 无关; 但我们遵循 CAT 的修复: 不过采样)
    oversample = 2.0 if not args.use_vff else 0.0
    train_dataset = DualViewDataset(raw_train, data_config.images_dir, oversample_factor=oversample)
    val_dataset = DualViewDataset(raw_val, data_config.images_dir, oversample_factor=0.0)

    # ============================================================
    # VFF: 加载 DenseNet features + 创建 adapter + 装 hook
    # ============================================================
    vff_adapter = None
    train_feat_dict = None
    val_feat_dict = None
    if args.use_vff:
        feat_file = args.densenet_feat_file or os.path.join(
            train_config.output_dir, "densenet_feats.npz"
        )
        feat_data = load_densenet_feats(feat_file)
        train_feat_dict = feat_data["train"]
        val_feat_dict = feat_data["val"]

        # 创建 VFF adapter, 维度和 Qwen2-VL merger 输出对齐
        vff_adapter = VisualFeatureFusion(
            vlm_dim=3584,        # Qwen2-VL hidden_size
            medical_dim=1024,    # DenseNet penultimate
            n_heads=8,
            dropout=0.1,
            init_gate=0.0,       # tanh(0) = 0, 下界保护
        ).to(device).to(torch.bfloat16)

        vff_params = sum(p.numel() for p in vff_adapter.parameters())
        print(f"[VFF] Adapter params: {vff_params/1e6:.2f}M")
        print(f"[VFF] Initial gate (tanh): {vff_adapter.current_gate:.4f}")

        # 装 hook (会在 merger forward 时拦截)
        vff_handle = install_vff_hook(model, vff_adapter, spatial_merge_size=2)
        print(f"[VFF] Hook installed on model.model.visual.merger")

        # 如果 resume, 加载 vff adapter state
        vff_state_path = os.path.join(save_dir, "vff_adapter.pt")
        if checkpoint and os.path.exists(vff_state_path):
            vff_adapter.load_state_dict(
                torch.load(vff_state_path, map_location=device, weights_only=True)
            )
            print(f"[VFF] Loaded VFF adapter state from {vff_state_path}")

        model._vff_adapter = vff_adapter
        model._vff_handle = vff_handle

    # ============================================================
    # Hint 源选择: VFF 模式下不用 hint, 否则按 hint_mode
    # ============================================================
    hint_dict_all = None
    hint_source = args.hint_mode if not args.use_vff else "none"

    if args.use_vff:
        print(f"[Train] Mode: VFF (Visual Feature Fusion, no text hint)")
    elif hint_source in ("cat", "mixed"):
        cat_hint_file = args.cat_hint_file or os.path.join(
            train_config.output_dir, "classifier_hints.json"
        )
        hint_dict_all = load_cat_hints(cat_hint_file)
        print(f"[Train] Hint mode: {hint_source.upper()}")
        if hint_source == "mixed":
            print(f"[Train] Mixed mode: 50% GT-extracted + 50% classifier-predicted")
    else:
        print(f"[Train] Hint mode: GT (Oracle-style, extract from GT report)")

    train_hint_dict = hint_dict_all["train"] if hint_dict_all else None
    val_hint_dict = hint_dict_all["val"] if hint_dict_all else None

    # 训练 collator
    dropout_map = {"gt": 0.3, "cat": 0.0, "mixed": 0.1, "none": 0.0}
    use_hint = not args.use_vff  # VFF 模式下完全不用 hint
    collator = DualViewCollator(
        processor,
        use_disease_hint=use_hint,
        hint_source=hint_source,
        hint_dict=train_hint_dict,
        densenet_feat_dict=train_feat_dict,   # VFF
        prompt_augment=True,
        hint_dropout=dropout_map[hint_source],
        max_length=2048,
    )
    val_collator = DualViewCollator(
        processor,
        use_disease_hint=use_hint,
        hint_source=hint_source,
        hint_dict=val_hint_dict,
        densenet_feat_dict=val_feat_dict,     # VFF
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
    
    # Optimizer: LoRA 参数 + VFF adapter 参数 (如果启用)
    vff_param_ids = set()
    if args.use_vff and vff_adapter is not None:
        vff_param_ids = {id(p) for p in vff_adapter.parameters()}

    lora_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in vff_param_ids
    ]
    param_groups = [
        {"params": lora_params, "lr": train_config.learning_rate},
    ]
    if args.use_vff and vff_adapter is not None:
        gate_params = [vff_adapter.gate_raw]
        other_vff_params = [p for n, p in vff_adapter.named_parameters() if n != "gate_raw"]
        param_groups.append({
            "params": other_vff_params,
            "lr": train_config.learning_rate * 10,   # 1e-4
        })
        param_groups.append({
            "params": gate_params,
            "lr": train_config.learning_rate * 100,  # 1e-3
        })
        print(f"[VFF] LR: lora={train_config.learning_rate:.0e}, "
              f"adapter={train_config.learning_rate*10:.0e}, "
              f"gate={train_config.learning_rate*100:.0e}")
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_config.weight_decay,
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

    print(f"\n{'='*70}")
    print(f" VLM LoRA Training (Disease-Aware Prompt Guidance)")
    print(f" Model: {train_config.model_name}")
    print(f" Data: R2Gen (train={len(train_dataset)}, val={len(val_dataset)})")
    print(f" Epochs: {start_epoch+1}..{num_epochs}, LR: {train_config.learning_rate}")
    print(f" LoRA rank: {train_config.discourse_lora_r}, grad_accum: {grad_accum}")
    print(f" Hint source: {hint_source.upper()} | "
          f"Dropout: {collator.hint_dropout} | Oversample: 2x")
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
                # Pop medical_feats 出来 (不是 model forward 的参数)
                medical_feats = batch.pop("medical_feats", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                # VFF: 在 forward 前把 medical feat 和 grid_thw 挂到 model 上, hook 会读取
                if args.use_vff and medical_feats is not None:
                    model._vff_medical_feats = medical_feats.to(device)
                    model._vff_grid_thw = batch["image_grid_thw"]
                try:
                    outputs = model(**batch)
                    loss = outputs.loss / grad_accum
                    loss.backward()
                    epoch_loss += outputs.loss.item()
                    epoch_count += 1
                finally:
                    # 清理, 避免污染下一个 step
                    if hasattr(model, "_vff_medical_feats"):
                        del model._vff_medical_feats
                    if hasattr(model, "_vff_grid_thw"):
                        del model._vff_grid_thw
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
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step=global_step)

                if global_step % train_config.logging_steps == 0:
                    print(f"  Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if args.max_steps and global_step >= args.max_steps:
            break

        val_loss = _eval_val(model, val_dataset, val_collator, device, batch_size, use_vff=args.use_vff)
        print(f"\n  Epoch {epoch+1} | Train: {epoch_loss/max(epoch_count,1):.4f} | Val: {val_loss:.4f}")
        if args.use_vff and vff_adapter is not None:
            print(f"  [VFF] gate = {vff_adapter.current_gate:.4f}")

        quick_generation_check(model, processor, val_dataset, device, num_samples=3)

        os.makedirs(save_dir, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            if args.use_vff and vff_adapter is not None:
                torch.save(vff_adapter.state_dict(), os.path.join(save_dir, "vff_adapter.pt"))
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
def _eval_val(model, val_dataset, collator, device, batch_size, use_vff=False):
    model.eval()
    total_loss, count = 0, 0
    for i in range(0, len(val_dataset), batch_size):
        batch_data = [val_dataset[j] for j in range(i, min(i + batch_size, len(val_dataset)))]
        try:
            batch = collator(batch_data)
            medical_feats = batch.pop("medical_feats", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_vff and medical_feats is not None:
                model._vff_medical_feats = medical_feats.to(device)
                model._vff_grid_thw = batch["image_grid_thw"]
            try:
                total_loss += model(**batch).loss.item()
                count += 1
            finally:
                if hasattr(model, "_vff_medical_feats"):
                    del model._vff_medical_feats
                if hasattr(model, "_vff_grid_thw"):
                    del model._vff_grid_thw
        except Exception:
            continue
    model.train()
    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--use_cat_hints", action="store_true",
                        help="(DEPRECATED, 用 --hint_mode cat 代替) "
                             "Use classifier-predicted hints (CAT training)")
    parser.add_argument("--hint_mode", type=str, default=None,
                        choices=["gt", "cat", "mixed"],
                        help="Hint source: gt (Oracle), cat (CAT), mixed (50/50, 推荐)")
    parser.add_argument("--cat_hint_file", type=str, default=None,
                        help="Path to classifier_hints.json")
    parser.add_argument("--use_vff", action="store_true",
                        help="Enable Visual Feature Fusion (inject DenseNet features)")
    parser.add_argument("--densenet_feat_file", type=str, default=None,
                        help="Path to densenet_feats.npz (default: output_dir/densenet_feats.npz)")
    args = parser.parse_args()

    # 兼容旧参数
    if args.hint_mode is None:
        args.hint_mode = "cat" if args.use_cat_hints else "gt"

    train_config = TrainingConfig()
    data_config = DataConfig()
    lora_config = LoRAConfig()

    os.makedirs(train_config.output_dir, exist_ok=True)

    lora_gt    = os.path.join(train_config.output_dir, "lora_discourse")
    lora_cat   = os.path.join(train_config.output_dir, "lora_cat")
    lora_mixed = os.path.join(train_config.output_dir, "lora_mixed")
    print("[Main] 训练状态:")
    print(f"  GT-hint    (lora_discourse): {'✅' if os.path.exists(lora_gt) else '⬜'}")
    print(f"  CAT-hint   (lora_cat):       {'✅' if os.path.exists(lora_cat) else '⬜'}")
    print(f"  Mixed-hint (lora_mixed):     {'✅' if os.path.exists(lora_mixed) else '⬜'}")
    print(f"  当前模式: {args.hint_mode.upper()}")

    run_training(args, train_config, data_config, lora_config)


if __name__ == "__main__":
    main()
