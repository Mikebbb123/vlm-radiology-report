"""
在 IU-Xray 上微调 DenseNet 疾病分类器 (改进版)

改进点 vs 旧版:
  1. pos_weight 上限 capped at 5.0 (旧版无上限, 导致假阳性爆炸)
  2. 训练数据增强 (翻转, 旋转, 平移)
  3. Label smoothing 0.05 (容忍关键词提取噪声)
  4. 验证集上自动搜索每类最优 threshold (替代固定 0.5)
  5. 增强标签提取模式 (更多同义表达)
  6. Focal loss 选项 (处理极端不平衡)

Usage:
  python finetune_classifier.py
  python finetune_classifier.py --epochs 20 --lr 1e-4
  python finetune_classifier.py --eval_only
  python finetune_classifier.py --use_focal_loss

安装:
  pip install torchxrayvision scikit-learn
"""
import os
import json
import argparse
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from config import DataConfig, TrainingConfig
from data_utils import (
    load_r2gen_data, clean_report_text,
    extract_disease_labels, DualViewDataset, DISEASE_PATTERNS,
)


# ============================================================
# 1. 疾病标签定义
# ============================================================

DISEASE_NAMES = list(DISEASE_PATTERNS.keys())
NUM_DISEASES = len(DISEASE_NAMES)
print(f"[Classifier] {NUM_DISEASES} disease categories: {DISEASE_NAMES}")


# ============================================================
# 2. 图像数据集 (支持增强)
# ============================================================

class XrayDiseaseDataset(Dataset):
    """X 光图像 + 多标签疾病标签数据集"""

    def __init__(self, samples: list, transform=None, augment: bool = False):
        self.samples = samples
        self.transform = transform  # xrv transforms (numpy → numpy)
        self.augment = augment      # torch-based augmentation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        import skimage.io
        import torchxrayvision as xrv

        img = skimage.io.imread(sample["image_path"])
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3:
            img = img.mean(2)
        img = img[None, ...]  # [1, H, W]

        # xrv transforms: center crop + resize (numpy → numpy)
        if self.transform:
            img = self.transform(img)

        img = torch.from_numpy(img).float()

        # torch-based augmentation (tensor → tensor)
        if self.augment:
            # 水平翻转 50%
            if random.random() < 0.5:
                img = torch.flip(img, dims=[-1])
            # 轻微旋转 + 平移: 用 affine_grid + grid_sample
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                tx = random.uniform(-0.05, 0.05)
                ty = random.uniform(-0.05, 0.05)
                scale = random.uniform(0.95, 1.05)
                rad = angle * 3.14159 / 180
                cos_a, sin_a = scale * np.cos(rad), scale * np.sin(rad)
                theta = torch.tensor([
                    [cos_a, -sin_a, tx],
                    [sin_a,  cos_a, ty],
                ], dtype=torch.float32).unsqueeze(0)
                grid = F.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)
                img = F.grid_sample(img.unsqueeze(0), grid, align_corners=False, padding_mode='zeros')[0]

        labels = torch.tensor(sample["labels"], dtype=torch.float32)
        return img, labels


def prepare_samples(raw_data: list, images_dir: str) -> list:
    """从原始数据准备分类样本"""
    samples = []
    label_counts = Counter()

    for item in raw_data:
        report = clean_report_text(item.get("report", ""))
        image_paths = item.get("image_path", [])

        if len(report) < 15 or len(image_paths) < 1:
            continue

        frontal_path = os.path.join(images_dir, image_paths[0])
        if not os.path.exists(frontal_path):
            continue

        findings = extract_disease_labels(report)
        labels = [1 if d in findings else 0 for d in DISEASE_NAMES]

        samples.append({
            "image_path": frontal_path,
            "labels": labels,
            "findings": findings,
            "report": report,
        })

        for f in findings:
            label_counts[f] += 1

    print(f"\n[Data] {len(samples)} samples, label distribution:")
    for disease in DISEASE_NAMES:
        count = label_counts.get(disease, 0)
        pct = count / len(samples) * 100
        bar = "█" * int(pct)
        print(f"  {disease:<22} {count:>4}/{len(samples)} ({pct:5.1f}%) {bar}")

    n_normal = sum(1 for s in samples if sum(s["labels"]) == 0)
    print(f"  {'(normal/no findings)':<22} {n_normal:>4}/{len(samples)} ({n_normal/len(samples)*100:5.1f}%)")

    return samples


# ============================================================
# 3. Focal Loss (可选, 替代 BCE)
# ============================================================

class FocalLossWithLogits(nn.Module):
    """
    Focal Loss: 减少对 easy negatives 的关注, 专注于 hard examples
    比 pos_weight 更稳定地处理类别不平衡
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


# ============================================================
# 4. 模型
# ============================================================

class FineTunedDiseaseClassifier(nn.Module):
    """
    基于 torchxrayvision DenseNet-121 的微调分类器

    改进: Dropout 0.3 → 0.5, 增加一层 BatchNorm
    """

    def __init__(self, pretrained_weights: str = "densenet121-res224-all"):
        super().__init__()
        import torchxrayvision as xrv

        base = xrv.models.DenseNet(weights=pretrained_weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_DISEASES),
        )

        # 冻结策略: 只训练 denseblock3, denseblock4, norm5, classifier
        for param in self.features.parameters():
            param.requires_grad = False

        for name, param in self.features.named_parameters():
            if any(k in name for k in ["denseblock3", "denseblock4", "norm5"]):
                param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Classifier] Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ============================================================
# 5. 每类最优阈值搜索 (关键改进)
# ============================================================

def find_optimal_thresholds(model, loader, device, search_range=None):
    """
    在验证集上为每个疾病类别搜索最优阈值

    策略: 优化 F1, 但对 precision 加权更高 (减少假阳性)
    因为对 VLM 来说, 错误的 hint 比没有 hint 伤害大
    """
    if search_range is None:
        search_range = np.arange(0.3, 0.95, 0.05)

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = model.predict_probs(images).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    optimal_thresholds = {}
    for i, disease in enumerate(DISEASE_NAMES):
        n_pos = int(all_labels[:, i].sum())
        if n_pos == 0:
            optimal_thresholds[disease] = 0.5
            continue

        best_score = -1
        best_thresh = 0.5

        for thresh in search_range:
            preds = (all_probs[:, i] > thresh).astype(int)
            prec = precision_score(all_labels[:, i], preds, zero_division=0)
            rec = recall_score(all_labels[:, i], preds, zero_division=0)

            # Fbeta with beta=0.5: 更重视 precision (减少假阳性)
            if prec + rec > 0:
                fbeta = (1 + 0.5**2) * prec * rec / (0.5**2 * prec + rec)
            else:
                fbeta = 0

            if fbeta > best_score:
                best_score = fbeta
                best_thresh = thresh

        optimal_thresholds[disease] = round(best_thresh, 2)

    print("\n[Threshold] Per-class optimal thresholds (optimized for F0.5):")
    for disease, thresh in optimal_thresholds.items():
        n_pos = int(all_labels[:, DISEASE_NAMES.index(disease)].sum())
        print(f"  {disease:<22} threshold={thresh:.2f} (GT positives={n_pos})")

    return optimal_thresholds


# ============================================================
# 6. 训练
# ============================================================

def train_classifier(args):
    import torchxrayvision as xrv

    data_config = DataConfig()
    train_config = TrainingConfig()

    raw_train, raw_val, raw_test = load_r2gen_data(data_config)

    print("\n[Preparing training data]")
    train_samples = prepare_samples(raw_train, data_config.images_dir)
    print("\n[Preparing validation data]")
    val_samples = prepare_samples(raw_val, data_config.images_dir)
    print("\n[Preparing test data]")
    test_samples = prepare_samples(raw_test, data_config.images_dir)

    # ---- xrv transforms only (numpy-compatible) ----
    # 数据增强在 XrayDiseaseDataset 中用 torch 操作实现
    base_transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    val_transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    train_dataset = XrayDiseaseDataset(train_samples, base_transform, augment=True)
    val_dataset = XrayDiseaseDataset(val_samples, val_transform, augment=False)
    test_dataset = XrayDiseaseDataset(test_samples, val_transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FineTunedDiseaseClassifier().to(device)

    # ---- 改进 2: pos_weight 上限 capped at 5.0 ----
    if args.use_focal_loss:
        criterion = FocalLossWithLogits(alpha=0.25, gamma=2.0)
        print(f"\n[Classifier] Using Focal Loss (alpha=0.25, gamma=2.0)")
    else:
        all_labels = np.array([s["labels"] for s in train_samples])
        pos_counts = all_labels.sum(axis=0)
        neg_counts = len(train_samples) - pos_counts
        raw_weight = neg_counts / np.maximum(pos_counts, 1)

        # 关键改进: cap pos_weight
        MAX_POS_WEIGHT = 5.0
        capped_weight = np.minimum(raw_weight, MAX_POS_WEIGHT)

        print(f"\n[Classifier] pos_weight (raw → capped at {MAX_POS_WEIGHT}):")
        for i, disease in enumerate(DISEASE_NAMES):
            print(f"  {disease:<22} {raw_weight[i]:>6.1f} → {capped_weight[i]:>4.1f}")

        pos_weight = torch.tensor(capped_weight, dtype=torch.float32).to(device)

        # ---- 改进 3: Label smoothing ----
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = os.path.join(train_config.output_dir, "disease_classifier")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "finetuned_classifier.pt")

    label_smoothing = args.label_smoothing

    print(f"\n{'='*70}")
    print(f" Fine-tuning Disease Classifier on IU-Xray (Improved)")
    print(f" Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    print(f" Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f" Label smoothing: {label_smoothing}")
    print(f" Data augmentation: flip + rotate + translate")
    print(f"{'='*70}\n")

    best_val_f1 = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # ---- Label smoothing: 0/1 → 0.05/0.95 ----
            if label_smoothing > 0:
                labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate_classifier(model, val_loader, device)
            print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"Val F1: {val_metrics['macro_f1']:.3f} | "
                  f"Val Acc: {val_metrics['accuracy']:.3f} | "
                  f"Abnormal Recall: {val_metrics['abnormal_recall']:.3f}")

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"    -> Best model (F1={best_val_f1:.3f})")

    if best_state:
        model.load_state_dict(best_state)

    # ---- 改进 4: 每类最优阈值搜索 ----
    print(f"\n{'='*70}")
    print(f" SEARCHING OPTIMAL THRESHOLDS ON VALIDATION SET")
    print(f"{'='*70}")
    optimal_thresholds = find_optimal_thresholds(model, val_loader, device)

    # 在测试集上用最优阈值评估
    print(f"\n{'='*70}")
    print(f" TEST SET EVALUATION (with optimal thresholds)")
    print(f"{'='*70}")
    threshold_list = [optimal_thresholds[d] for d in DISEASE_NAMES]
    test_metrics = evaluate_classifier(
        model, test_loader, device,
        threshold=threshold_list,
        verbose=True,
    )

    # 也跑一次固定阈值的评估作为对比
    print(f"\n{'='*70}")
    print(f" TEST SET EVALUATION (fixed threshold=0.5, for comparison)")
    print(f"{'='*70}")
    test_metrics_fixed = evaluate_classifier(
        model, test_loader, device,
        threshold=0.5,
        verbose=True,
    )

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"\n[Classifier] Saved model to {save_path}")

    # 保存阈值和元信息
    meta = {
        "disease_names": DISEASE_NAMES,
        "num_diseases": NUM_DISEASES,
        "best_val_f1": round(best_val_f1, 4),
        "optimal_thresholds": optimal_thresholds,
        "test_metrics_optimal": {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in test_metrics.items()},
        "test_metrics_fixed05": {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in test_metrics_fixed.items()},
        "train_samples": len(train_samples),
        "epochs": args.epochs,
        "lr": args.lr,
        "label_smoothing": label_smoothing,
        "pos_weight_cap": 5.0,
        "use_focal_loss": args.use_focal_loss,
    }
    meta_path = os.path.join(save_dir, "classifier_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Classifier] Saved metadata to {meta_path}")

    print(f"\n[Classifier] Next steps:")
    print(f"  python evaluate.py                    # auto-loads optimal thresholds")
    print(f"  python evaluate.py --no_disease       # baseline comparison")
    print(f"  python evaluate.py --ablation         # ablation study")


def evaluate_classifier(model, loader, device, threshold=0.5, verbose=False):
    """
    评估分类器性能

    threshold: float 或 list[float] (每类不同阈值)
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = model.predict_probs(images).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 支持每类不同阈值
    if isinstance(threshold, (list, np.ndarray)):
        threshold_arr = np.array(threshold)
        all_preds = (all_probs > threshold_arr[None, :]).astype(int)
    else:
        all_preds = (all_probs > threshold).astype(int)

    accuracy = (all_preds == all_labels).mean()

    # 异常样本 recall
    abnormal_mask = all_labels.sum(axis=1) > 0
    if abnormal_mask.sum() > 0:
        abnormal_preds = all_preds[abnormal_mask].sum(axis=1) > 0
        abnormal_recall = abnormal_preds.mean()
    else:
        abnormal_recall = 0.0

    # 每类 F1
    per_class_f1 = []
    for i, disease in enumerate(DISEASE_NAMES):
        thresh_i = threshold[i] if isinstance(threshold, (list, np.ndarray)) else threshold
        if all_labels[:, i].sum() > 0:
            f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            prec = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            rec = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            per_class_f1.append(f1)

            if verbose:
                n_pos = int(all_labels[:, i].sum())
                n_pred = int(all_preds[:, i].sum())
                print(f"  {disease:<22} F1={f1:.3f} P={prec:.3f} R={rec:.3f} "
                      f"(GT={n_pos}, Pred={n_pred}, thresh={thresh_i:.2f})")
        else:
            per_class_f1.append(0.0)
            if verbose:
                print(f"  {disease:<22} (no positive samples, thresh={thresh_i:.2f})")

    # macro F1 只计算有阳性样本的类别
    f1_with_positives = [per_class_f1[i] for i in range(len(DISEASE_NAMES))
                         if all_labels[:, i].sum() > 0]
    macro_f1 = np.mean(f1_with_positives) if f1_with_positives else 0.0

    # 总假阳性数
    total_fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    total_fn = int(((all_preds == 0) & (all_labels == 1)).sum())

    if verbose:
        print(f"\n  Overall: Acc={accuracy:.3f} | Macro F1={macro_f1:.3f} | "
              f"Abnormal Recall={abnormal_recall:.3f}")
        print(f"  Total FP={total_fp} | Total FN={total_fn}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "abnormal_recall": abnormal_recall,
        "per_class_f1": per_class_f1,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


# ============================================================
# 7. 仅评估模式
# ============================================================

def eval_only(args):
    import torchxrayvision as xrv

    data_config = DataConfig()
    train_config = TrainingConfig()

    save_dir = os.path.join(train_config.output_dir, "disease_classifier")
    save_path = os.path.join(save_dir, "finetuned_classifier.pt")
    meta_path = os.path.join(save_dir, "classifier_meta.json")

    if not os.path.exists(save_path):
        print(f"[Error] No model found at {save_path}. Run training first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FineTunedDiseaseClassifier().to(device)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"[Classifier] Loaded from {save_path}")

    # 加载最优阈值
    optimal_thresholds = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "optimal_thresholds" in meta:
            optimal_thresholds = meta["optimal_thresholds"]
            print(f"[Classifier] Loaded optimal thresholds from {meta_path}")

    _, raw_val, raw_test = load_r2gen_data(data_config)
    test_samples = prepare_samples(raw_test, data_config.images_dir)

    val_transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    test_dataset = XrayDiseaseDataset(test_samples, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    # 用最优阈值评估
    if optimal_thresholds:
        print(f"\n{'='*70}")
        print(f" TEST SET (optimal per-class thresholds)")
        print(f"{'='*70}")
        threshold_list = [optimal_thresholds.get(d, 0.5) for d in DISEASE_NAMES]
        evaluate_classifier(model, test_loader, device, threshold=threshold_list, verbose=True)

    # 固定阈值对比
    print(f"\n{'='*70}")
    print(f" TEST SET (fixed threshold=0.5)")
    print(f"{'='*70}")
    evaluate_classifier(model, test_loader, device, threshold=0.5, verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use Focal Loss instead of BCE (better for imbalance)")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train_classifier(args)


if __name__ == "__main__":
    main()
