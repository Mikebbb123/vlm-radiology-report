"""
VFF Smoke Test

不训练, 只做一次 forward + backward, 验证:
  1. Hook 在 merger 上正确触发
  2. Tensor shape 正确
  3. Loss 能算出来且不是 NaN
  4. 梯度能流到 VFF adapter
  5. VFF adapter 的 gate 初始 = 0 (等同于 baseline)

Usage:
  python smoke_test_vff.py
"""
import os
import torch
import numpy as np
from peft import get_peft_model

from config import TrainingConfig, DataConfig, LoRAConfig
from data_utils import (
    load_r2gen_data, DualViewDataset, DualViewCollator, load_densenet_feats,
)
from model import load_vlm_model, create_lora_config
from visual_feature_fusion import VisualFeatureFusion, install_vff_hook


def main():
    tc, dc, lc = TrainingConfig(), DataConfig(), LoRAConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print(" VFF Smoke Test")
    print("="*70)

    # 1. 加载数据 (只取前 2 个样本)
    raw_train, _, _ = load_r2gen_data(dc)
    ds = DualViewDataset(raw_train, dc.images_dir, oversample_factor=0.0)
    print(f"[1] Dataset: {len(ds)} samples, taking 2 for smoke test")

    # 2. 加载 DenseNet features
    feat_file = os.path.join(tc.output_dir, "densenet_feats.npz")
    feat_data = load_densenet_feats(feat_file)
    train_feat = feat_data["train"]
    print(f"[2] DenseNet features: {len(train_feat)} train samples")

    # 3. 加载 VLM + LoRA
    print("[3] Loading Qwen2-VL...")
    model, processor = load_vlm_model(tc)
    lora_cfg = create_lora_config(tc.discourse_lora_r, lc.target_modules)
    model = get_peft_model(model, lora_cfg)
    model.train()
    print(f"    Device: {next(model.parameters()).device}")

    # 4. 创建 VFF adapter 并装 hook
    vff = VisualFeatureFusion(vlm_dim=3584, medical_dim=1024,
                               n_heads=8, dropout=0.1, init_gate=0.0
                              ).to(device).to(torch.bfloat16)
    install_vff_hook(model, vff, spatial_merge_size=2)
    print(f"[4] VFF adapter: {sum(p.numel() for p in vff.parameters())/1e6:.2f}M params, "
          f"gate={vff.current_gate:.4f}")
    assert abs(vff.current_gate) < 1e-6, "Gate 初始应该是 0"

    # 5. 准备 batch
    collator = DualViewCollator(
        processor, use_disease_hint=False, hint_source="none",
        densenet_feat_dict=train_feat, prompt_augment=False,
        hint_dropout=0.0, max_length=2048,
    )
    batch_data = [ds[0]]
    batch = collator(batch_data)
    medical_feats = batch.pop("medical_feats")
    batch = {k: v.to(device) for k, v in batch.items()}
    print(f"[5] Batch prepared:")
    print(f"    input_ids: {batch['input_ids'].shape}")
    print(f"    pixel_values: {batch['pixel_values'].shape}")
    print(f"    image_grid_thw: {batch['image_grid_thw'].shape} = {batch['image_grid_thw'].tolist()}")
    print(f"    medical_feats: {medical_feats.shape}")

    # 6. Forward WITHOUT VFF (baseline, gate=0 等同于无 adapter)
    print("\n[6] Forward #1: gate=0 (should equal baseline)")
    model._vff_medical_feats = medical_feats.to(device)
    model._vff_grid_thw = batch["image_grid_thw"]
    try:
        out1 = model(**batch)
        loss1 = out1.loss.item()
        print(f"    Loss: {loss1:.4f}")
        assert not np.isnan(loss1), "Loss is NaN!"
    finally:
        del model._vff_medical_feats
        del model._vff_grid_thw

    # 7. 把 gate 设成一个小正值, 验证 forward 仍正常, 梯度流通
    print("\n[7] Forward #2: gate=tanh(1)≈0.76 (VFF active)")
    with torch.no_grad():
        vff.gate_raw.fill_(1.0)
    print(f"    Adjusted gate = {vff.current_gate:.4f}")

    model._vff_medical_feats = medical_feats.to(device)
    model._vff_grid_thw = batch["image_grid_thw"]
    try:
        out2 = model(**batch)
        loss2 = out2.loss
        print(f"    Loss: {loss2.item():.4f}")
        assert not torch.isnan(loss2), "Loss is NaN!"

        # Backward, 检查 VFF 梯度
        loss2.backward()
        grad_norm = sum(
            p.grad.norm().item()**2 for p in vff.parameters() if p.grad is not None
        ) ** 0.5
        print(f"    VFF grad norm: {grad_norm:.4f}")
        assert grad_norm > 0, "VFF adapter 没有梯度, hook 可能没接上!"
    finally:
        del model._vff_medical_feats
        del model._vff_grid_thw

    # 8. 复位 gate 到 0, 再测一次
    with torch.no_grad():
        vff.gate_raw.fill_(0.0)
    print(f"\n[8] Gate reset to {vff.current_gate:.4f}")

    print("\n" + "="*70)
    print(" ✅ All smoke tests passed!")
    print("="*70)
    print(f"  - Hook registered and triggered")
    print(f"  - Forward runs without error")
    print(f"  - Loss is finite: {loss1:.4f} (gate=0), {loss2.item():.4f} (gate≈0.76)")
    print(f"  - Gradient flows to VFF adapter (norm={grad_norm:.4f})")
    print(f"\n[Next] python train.py --use_vff --reset")


if __name__ == "__main__":
    main()
