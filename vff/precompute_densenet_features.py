"""
Visual Feature Fusion: 预计算 DenseNet penultimate features

一次性把 DenseNet 的 1024-d penultimate feature 对 train/val/test 所有样本
算好, 存到 .npz 文件. 训练时直接查表, 不用每 step 重新 forward DenseNet.

只用 frontal view (fusion 策略 A).

输出:
  /content/drive/MyDrive/medk_lora_r2gen/densenet_feats.npz

Usage:
  python precompute_densenet_features.py
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms

from config import DataConfig, TrainingConfig
from data_utils import load_r2gen_data, clean_report_text
from finetune_classifier import FineTunedDiseaseClassifier


class DenseNetFeatureExtractor(nn.Module):
    """包装 FineTunedDiseaseClassifier, 返回 penultimate feature [B, 1024]"""
    def __init__(self, clf: FineTunedDiseaseClassifier):
        super().__init__()
        self.features = clf.features
        self.pool = clf.pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return pooled  # [B, 1024]


def load_classifier(train_config):
    save_dir = os.path.join(train_config.output_dir, "disease_classifier")
    ckpt_path = os.path.join(save_dir, "finetuned_classifier.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {ckpt_path}\n"
            f"Run `python finetune_classifier.py` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = FineTunedDiseaseClassifier().to(device)
    clf.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    clf.eval()
    print(f"[VFF] Loaded classifier from {ckpt_path}")

    extractor = DenseNetFeatureExtractor(clf).to(device)
    extractor.eval()
    return extractor, device


def load_image_for_densenet(image_path, xrv_transform):
    """和 finetune_classifier.py 一致的图像加载"""
    import skimage.io
    import torchxrayvision as xrv

    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)
    if img.ndim == 3:
        img = img.mean(2)
    img = img[None, ...]
    img = xrv_transform(img)
    return torch.from_numpy(img).float()


def extract_features_for_split(split_name, raw_data, images_dir,
                                extractor, device, xrv_transform):
    """对一个 split 的所有样本提取 frontal view 的 penultimate feature"""
    ids = []
    feats = []
    n_skipped = 0

    for item in tqdm(raw_data, desc=f"[{split_name}]"):
        sample_id = item.get("id", "")
        img_paths = item.get("image_path", [])
        report = clean_report_text(item.get("report", ""))

        # 和 DualViewDataset 一致的过滤
        if not sample_id or len(img_paths) < 2 or len(report) < 15:
            n_skipped += 1
            continue

        frontal = os.path.join(images_dir, img_paths[0])
        lateral = os.path.join(images_dir, img_paths[1])

        if not os.path.exists(frontal) or not os.path.exists(lateral):
            n_skipped += 1
            continue

        try:
            img_tensor = load_image_for_densenet(frontal, xrv_transform)
            img_batch = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                feat = extractor(img_batch)[0].cpu().numpy()  # [1024]
        except Exception as e:
            print(f"[Warn] Failed on {sample_id}: {e}")
            n_skipped += 1
            continue

        ids.append(sample_id)
        feats.append(feat)

    print(f"  {split_name}: {len(ids)} samples extracted, {n_skipped} skipped")
    if not feats:
        return ids, np.zeros((0, 1024), dtype=np.float32)
    return ids, np.stack(feats).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    import torchxrayvision as xrv

    train_config = TrainingConfig()
    data_config = DataConfig()

    output_file = args.output_file or os.path.join(
        train_config.output_dir, "densenet_feats.npz"
    )

    extractor, device = load_classifier(train_config)
    n_params = sum(p.numel() for p in extractor.parameters())
    print(f"[VFF] Extractor params: {n_params/1e6:.2f}M")

    xrv_transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    raw_train, raw_val, raw_test = load_r2gen_data(data_config)

    print(f"\n{'='*70}")
    print(f" Pre-computing DenseNet penultimate features (1024-d)")
    print(f" Strategy: frontal view only")
    print(f"{'='*70}\n")

    train_ids, train_feats = extract_features_for_split(
        "train", raw_train, data_config.images_dir, extractor, device, xrv_transform)
    val_ids, val_feats = extract_features_for_split(
        "val", raw_val, data_config.images_dir, extractor, device, xrv_transform)
    test_ids, test_feats = extract_features_for_split(
        "test", raw_test, data_config.images_dir, extractor, device, xrv_transform)

    print(f"\n{'='*70}")
    print(f" Feature statistics")
    print(f"{'='*70}")
    for name, feats in [("train", train_feats), ("val", val_feats), ("test", test_feats)]:
        if len(feats) == 0:
            continue
        print(f"  {name}: shape={feats.shape}, "
              f"mean={feats.mean():.4f}, std={feats.std():.4f}, "
              f"min={feats.min():.4f}, max={feats.max():.4f}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(
        output_file,
        train_ids=np.array(train_ids),
        train_feats=train_feats,
        val_ids=np.array(val_ids),
        val_feats=val_feats,
        test_ids=np.array(test_ids),
        test_feats=test_feats,
    )

    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\n[VFF] Saved to {output_file} ({size_mb:.1f} MB)")
    print(f"\n[Next] python train.py --use_vff --reset")


if __name__ == "__main__":
    main()
