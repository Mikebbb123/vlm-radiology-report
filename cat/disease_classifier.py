"""
推理时疾病分类器 (基于 torchxrayvision 预训练 DenseNet-121)

安装: pip install torchxrayvision

零训练成本, 直接对 X 光图像预测 18 种病理概率
"""
import torch
import numpy as np
from typing import List, Dict, Optional


class DiseaseClassifier:
    """
    预训练 CheXNet 多标签疾病分类器

    推理时使用: X光图 → 疾病概率 → prompt hint
    """

    RELEVANT_PATHOLOGIES = {
        "Atelectasis":         "atelectasis",
        "Cardiomegaly":        "cardiomegaly",
        "Consolidation":       "consolidation",
        "Edema":               "pulmonary edema",
        "Effusion":            "pleural effusion",
        "Emphysema":           "emphysema",
        "Enlarged Cardiomediastinum": "enlarged cardiomediastinal silhouette",
        "Fracture":            "fracture",
        "Lung Opacity":        "lung opacity",
        "Pleural_Thickening":  "pleural thickening",
        "Pneumonia":           "pneumonia",
        "Pneumothorax":        "pneumothorax",
    }

    def __init__(self, device: str = "cuda",
                 weights: str = "densenet121-res224-all",
                 threshold: float = 0.5):
        import torchxrayvision as xrv
        import torchvision

        self.device = device
        self.threshold = threshold

        self.model = xrv.models.DenseNet(weights=weights).to(device)
        self.model.eval()

        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])

        self.pathology_indices = {}
        for i, name in enumerate(self.model.pathologies):
            if name in self.RELEVANT_PATHOLOGIES:
                self.pathology_indices[name] = i

        print(f"[DiseaseClassifier] Loaded {weights}, "
              f"tracking {len(self.pathology_indices)} pathologies")

    def _load_image(self, image_path: str) -> torch.Tensor:
        import skimage.io
        import torchxrayvision as xrv

        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3:
            img = img.mean(2)
        img = img[None, ...]
        img = self.transform(img)
        return torch.from_numpy(img).float()

    def predict(self, image_path: str) -> Dict[str, float]:
        img = self._load_image(image_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img)
        probs = {}
        for name, idx in self.pathology_indices.items():
            probs[name] = round(torch.sigmoid(outputs[0][idx]).item(), 3)
        return probs

    def get_prompt_hint(self, frontal_path: str,
                        lateral_path: str = None,
                        threshold: float = None) -> str:
        """生成 disease-aware prompt hint"""
        threshold = threshold or self.threshold

        probs = self.predict(frontal_path)
        if lateral_path:
            probs_lat = self.predict(lateral_path)
            for name in probs:
                probs[name] = max(probs[name], probs_lat.get(name, 0))

        findings = []
        for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
            if prob >= threshold:
                findings.append(self.RELEVANT_PATHOLOGIES[name])

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
