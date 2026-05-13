"""
CAT: 预计算分类器 hint

一次性跑训练好的 DenseNet 分类器, 对 train/val/test 所有样本生成预测 hint,
存到一个 JSON 文件. 后续训练和推理都从这个 JSON 查表, 保证 train-test
的 hint 分布严格一致.

Usage:
  python precompute_classifier_hints.py
  python precompute_classifier_hints.py --threshold 0.5  # 覆盖默认阈值
  python precompute_classifier_hints.py --use_fixed_threshold  # 用 0.5 而不是 optimal

输出:
  /content/drive/MyDrive/medk_lora_r2gen/classifier_hints.json
  格式: {
    "train": {sample_id: [disease1, disease2, ...]},
    "val":   {sample_id: [...]},
    "test":  {sample_id: [...]},
    "meta":  {
      "threshold_mode": "optimal" | "fixed",
      "thresholds":     {disease: float},
      "stats":          {split: {"n_samples": int, "n_with_hint": int, ...}}
    }
  }
"""
import os
import json
import argparse
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm

from torchvision import transforms

from config import DataConfig, TrainingConfig
from data_utils import load_r2gen_data, clean_report_text
from finetune_classifier import (
    FineTunedDiseaseClassifier,
    DISEASE_NAMES,
    NUM_DISEASES,
)


def load_classifier_and_thresholds(train_config, use_fixed=False, override_thresh=None):
    """加载分类器权重和 per-class 最优阈值"""
    save_dir = os.path.join(train_config.output_dir, "disease_classifier")
    ckpt_path = os.path.join(save_dir, "finetuned_classifier.pt")
    meta_path = os.path.join(save_dir, "classifier_meta.json")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {ckpt_path}\n"
            f"Run `python finetune_classifier.py` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FineTunedDiseaseClassifier().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[CAT] Loaded classifier from {ckpt_path}")

    # 决定阈值
    thresholds = {}
    if override_thresh is not None:
        thresholds = {d: override_thresh for d in DISEASE_NAMES}
        mode = f"fixed={override_thresh}"
    elif use_fixed:
        thresholds = {d: 0.5 for d in DISEASE_NAMES}
        mode = "fixed=0.5"
    elif os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "optimal_thresholds" in meta:
            thresholds = meta["optimal_thresholds"]
            mode = "optimal (from classifier_meta.json)"
        else:
            thresholds = {d: 0.5 for d in DISEASE_NAMES}
            mode = "fixed=0.5 (no optimal found)"
    else:
        thresholds = {d: 0.5 for d in DISEASE_NAMES}
        mode = "fixed=0.5 (no meta file)"

    print(f"[CAT] Threshold mode: {mode}")
    for d in DISEASE_NAMES:
        print(f"  {d:<22} threshold={thresholds[d]:.2f}")

    return model, thresholds, device


def load_image_for_classifier(image_path, xrv_transform):
    """用和 finetune_classifier.py 完全一致的方式加载图像"""
    import skimage.io
    import torchxrayvision as xrv

    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)
    if img.ndim == 3:
        img = img.mean(2)
    img = img[None, ...]  # [1, H, W]
    img = xrv_transform(img)
    return torch.from_numpy(img).float()


def predict_hint_for_sample(model, device, xrv_transform,
                             frontal_path, lateral_path, thresholds):
    """
    对一个样本 (双视图) 预测疾病 hint.
    双视图取 max 概率, 再应用每类阈值.
    """
    frontal_tensor = load_image_for_classifier(frontal_path, xrv_transform)
    frontal_batch = frontal_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        frontal_probs = model.predict_probs(frontal_batch)[0].cpu().numpy()

    probs = frontal_probs
    if lateral_path is not None and os.path.exists(lateral_path):
        try:
            lateral_tensor = load_image_for_classifier(lateral_path, xrv_transform)
            lateral_batch = lateral_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                lateral_probs = model.predict_probs(lateral_batch)[0].cpu().numpy()
            # 双视图 max
            probs = np.maximum(frontal_probs, lateral_probs)
        except Exception:
            pass

    # 应用每类阈值
    predicted = []
    for i, disease in enumerate(DISEASE_NAMES):
        if probs[i] > thresholds[disease]:
            predicted.append(disease)
    return predicted, probs.tolist()


def process_split(split_name, raw_data, images_dir, model, device,
                   xrv_transform, thresholds):
    """对一个 split 所有样本预测 hint"""
    hints = {}
    label_counter = Counter()
    n_with_hint = 0
    n_skipped = 0

    for item in tqdm(raw_data, desc=f"[{split_name}]"):
        sample_id = item.get("id", "")
        img_paths = item.get("image_path", [])
        report = clean_report_text(item.get("report", ""))

        if not sample_id or len(img_paths) < 1 or len(report) < 15:
            n_skipped += 1
            continue

        frontal = os.path.join(images_dir, img_paths[0])
        if not os.path.exists(frontal):
            n_skipped += 1
            continue

        lateral = None
        if len(img_paths) >= 2:
            lateral_candidate = os.path.join(images_dir, img_paths[1])
            if os.path.exists(lateral_candidate):
                lateral = lateral_candidate

        try:
            predicted, _ = predict_hint_for_sample(
                model, device, xrv_transform, frontal, lateral, thresholds
            )
        except Exception as e:
            print(f"[Warn] Failed on {sample_id}: {e}")
            predicted = []

        hints[sample_id] = predicted
        if predicted:
            n_with_hint += 1
        for d in predicted:
            label_counter[d] += 1

    stats = {
        "n_samples": len(hints),
        "n_skipped": n_skipped,
        "n_with_hint": n_with_hint,
        "pct_with_hint": round(n_with_hint / max(len(hints), 1) * 100, 1),
        "label_distribution": dict(label_counter),
    }
    print(f"  {split_name}: {len(hints)} samples, "
          f"{n_with_hint} with hint ({stats['pct_with_hint']}%)")
    for d in DISEASE_NAMES:
        c = label_counter.get(d, 0)
        print(f"    {d:<22} {c:>4} ({c/max(len(hints),1)*100:4.1f}%)")

    return hints, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON path (default: output_dir/classifier_hints.json)")
    parser.add_argument("--use_fixed_threshold", action="store_true",
                        help="Use fixed threshold=0.5 instead of per-class optimal")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override with a single fixed threshold for all classes")
    args = parser.parse_args()

    import torchxrayvision as xrv

    train_config = TrainingConfig()
    data_config = DataConfig()

    output_file = args.output_file or os.path.join(
        train_config.output_dir, "classifier_hints.json"
    )

    # 1. 加载分类器
    model, thresholds, device = load_classifier_and_thresholds(
        train_config,
        use_fixed=args.use_fixed_threshold,
        override_thresh=args.threshold,
    )

    # 2. 准备 xrv transform (和 finetune_classifier.py 一致)
    xrv_transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    # 3. 加载数据
    raw_train, raw_val, raw_test = load_r2gen_data(data_config)

    # 4. 对三个 split 预测 hint
    print(f"\n{'='*70}")
    print(f" Pre-computing classifier hints for all splits")
    print(f"{'='*70}")

    all_hints = {}
    all_stats = {}

    for name, data in [("train", raw_train), ("val", raw_val), ("test", raw_test)]:
        print(f"\n[{name}]")
        hints, stats = process_split(
            name, data, data_config.images_dir, model, device,
            xrv_transform, thresholds
        )
        all_hints[name] = hints
        all_stats[name] = stats

    # 5. 保存
    output = {
        "train": all_hints["train"],
        "val":   all_hints["val"],
        "test":  all_hints["test"],
        "meta": {
            "threshold_mode": "fixed" if args.use_fixed_threshold or args.threshold else "optimal",
            "thresholds": thresholds,
            "disease_names": DISEASE_NAMES,
            "stats": all_stats,
            "classifier_ckpt": os.path.join(
                train_config.output_dir, "disease_classifier", "finetuned_classifier.pt"
            ),
        }
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f" Saved hints to {output_file}")
    print(f"{'='*70}")
    print(f"  train: {all_stats['train']['n_samples']} samples, "
          f"{all_stats['train']['pct_with_hint']}% with hint")
    print(f"  val:   {all_stats['val']['n_samples']} samples, "
          f"{all_stats['val']['pct_with_hint']}% with hint")
    print(f"  test:  {all_stats['test']['n_samples']} samples, "
          f"{all_stats['test']['pct_with_hint']}% with hint")

    print(f"\n[Next steps]")
    print(f"  1. Retrain VLM with CAT hints:")
    print(f"     python train.py --reset --use_cat_hints")
    print(f"  2. Evaluate with same CAT hints:")
    print(f"     python evaluate.py --use_cat_hints")


if __name__ == "__main__":
    main()
