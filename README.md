[README.md](https://github.com/user-attachments/files/27684410/README.md)
# An Empirical Study of VLM Fine-tuning for Radiology Report Generation
### Why Retrieval Baselines Win on Small-Scale Medical Benchmarks

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-findings">Key Findings</a> •
  <a href="#results">Results</a> •
  <a href="#setup">Setup</a> •
  <a href="#reproducing-results">Reproducing Results</a> •
  <a href="#citation">Citation</a>
</p>

**Authors:** Shen Zhengbo, Lai Bangheng  
**Institution:** College of Science, Mathematics and Technology, Wenzhou-Kean University  
**Year:** 2025

> 📄 [Technical Report (Zenodo)](YOUR_ZENODO_LINK) · 📄 [arXiv](YOUR_ARXIV_LINK) · 🤗 [Base Model (Qwen2-VL-7B)](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

---

## Overview

Recent advances in general-purpose vision-language models (VLMs) such as Qwen2-VL suggest an attractive alternative to task-specific architectures for medical image-to-text generation: simply fine-tune a pretrained VLM with LoRA on a small medical dataset, optionally augmented with classifier-derived hints or domain pre-training.

**This work asks a different question.** Rather than proposing a new method, we systematically investigate whether VLM fine-tuning on IU-Xray can reliably improve over trivial baselines. We hold the base model (Qwen2-VL-7B) and dataset (IU-Xray, R2Gen split) fixed, and compare **seven approaches** spanning four design axes:

| Design Axis | Methods |
|------------|---------|
| Pure fine-tuning | LoRA Baseline |
| Prompt-channel enhancement | Disease-Hint, CAT |
| Visual-channel enhancement | VFF (Visual Feature Fusion) |
| Domain pre-training | MIMIC-CXR Text-Only Pre-training |
| Reference points | NN Retrieval, Oracle (GT leak) |

---

## Key Findings

**Finding 1: All fine-tuning methods plateau around BLEU-4 ≈ 8.7**

Despite extensive experimentation, every learned method — including elaborate enhancements — converges to the no-hint LoRA baseline on BLEU-4.

**Finding 2: A 30-second retrieval baseline outperforms all fine-tuned models**

A zero-training nearest-neighbor retrieval baseline (cosine similarity in DenseNet feature space) achieves BLEU-4 = 9.08, exceeding every fine-tuned configuration. Ten days of GPU training cannot beat 30 seconds of retrieval.

**Finding 3: BLEU and clinical metrics tell opposite stories**

VFF achieves the *lowest* BLEU-4 among fine-tuned methods but the *highest* CheXbert-14 micro-F1 (54.17) and RadGraph scores. This divergence suggests BLEU on IU-Xray primarily rewards template matching rather than diagnostic accuracy.

**Core diagnosis:** The fundamental bottleneck is data scarcity (2,069 training samples). The language prior from pre-training dominates the visual signal — zero-shot Qwen2-VL generates the same "cardiomegaly is noted..." template for visually distinct images, including normal cases.

---

## Results

### Main Results on IU-Xray Test Set (590 samples, R2Gen split)

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | Unique | CheXbert-14 µF1 | RadGraph-S |
|--------|--------|--------|--------|--------|---------|--------|--------|-----------------|------------|
| LoRA Baseline | 29.97 | 17.71 | 11.85 | 8.71 | 30.41 | 23.95 | 30 | 51.63 | 35.75 |
| Disease-Hint *(main)* | — | — | — | ~9.0 | — | — | — | — | — |
| CAT | 22.14 | 13.27 | 9.20 | 6.63 | 27.50 | 20.19 | 29 | 53.66 | 34.37 |
| VFF | 27.06 | 16.54 | 11.47 | 8.39 | 29.81 | 23.23 | **43** | **54.17** | **37.11** |
| MIMIC Pre-training | 22.46 | 13.60 | 9.38 | 6.80 | 27.15 | 20.02 | 17 ⚠️ | 36.35 | 32.57 |
| **NN Retrieval** *(zero training)* | **29.83** | **18.11** | **12.38** | **9.08** | 27.62 | — | — | 41.45 | 33.06 |
| Oracle *(GT leak, ref only)* | 37.54 | 25.83 | 19.86 | 16.14 | 35.87 | 32.28 | — | 54.51 | 43.08 |

> ⚠️ MIMIC Pre-training shows severe mode collapse (only 17 unique outputs across 590 test samples).  
> Oracle uses ground-truth disease keywords leaked into the prompt — this is a sanity check, not a real comparison target.

### Comparison with Literature

| Model | BLEU-4 | Notes |
|-------|--------|-------|
| R2Gen (2020) | 16.5 | Task-specific encoder-decoder |
| R2GenCMN (2021) | 17.6 | Task-specific + memory |
| LLaVA-Med | 18.6 | Pre-trained on 213K MIMIC-CXR images |
| SERPENT-VLM (2024) | 19.0 | Self-refining alignment loss |
| **Ours (Baseline)** | **8.71** | Qwen2-VL-7B + LoRA, IU-Xray only |

The primary gap vs. LLaVA-Med: they pre-train on 213K MIMIC-CXR samples; we use 2,069 IU-Xray samples.

---

## Mechanistic Analysis

### Why prompt-channel enhancement fails (CAT)
- DenseNet classifier predictions are noisy (~55% match with GT labels)
- With only 2,069 training samples, LoRA cannot learn precise "noisy hint → specific word" mappings
- The model learns a shortcut: "hint present → switch to short template", ignoring hint content

### Why visual-channel enhancement fails (VFF)
- The VFF gate converges to 0.0327 regardless of learning rate schedule
- Language-side LoRA and visual-side adapter compete for the same gradient budget
- Result: visual features are effectively ignored despite architectural changes

### Why MIMIC text pre-training fails
- Text-only pre-training on 148K MIMIC-CXR reports causes template memorization
- After fine-tuning, the model outputs "the heart is normal in size. the mediastinum is unremarkable." for diverse images
- Mode collapse: only 17 unique outputs across 590 test samples

### Why NN retrieval wins
- IU-Xray train/test sets are similar in DenseNet feature space
- 92.6% of training reports contain normal-finding keywords → BLEU rewards template overlap
- Retrieval directly exploits this structural homogeneity without any learned parameters

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md               # Dataset download instructions
│   └── preprocess.py           # Data preprocessing and R2Gen split
│
├── models/
│   ├── densenet_classifier.py  # DenseNet-121 disease classifier
│   └── vff_module.py           # Gated cross-attention VFF module
│
├── experiments/
│   ├── baseline_lora.py        # Experiment 1: LoRA baseline
│   ├── disease_hint.py         # Experiment 2: Disease-hint prompting
│   ├── cat_training.py         # Experiment 3: Classifier-Aligned Training
│   ├── vff_training.py         # Experiment 4: Visual Feature Fusion
│   ├── mimic_pretrain.py       # Experiment 5: MIMIC text pre-training
│   └── nn_retrieval.py         # Experiment 6: NN retrieval baseline
│
├── evaluation/
│   ├── compute_nlg.py          # BLEU, ROUGE-L, METEOR
│   └── compute_clinical.py     # CheXbert-5/14, RadGraph
│
├── results/
│   ├── main_results.csv        # Full results table
│   └── predictions/            # Model outputs per experiment
│
└── docs/
    └── technical_report.pdf    # Full technical report
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers >= 4.45.0`
- `peft >= 0.12.0` (LoRA)
- `torch >= 2.0.0`
- `torchxrayvision` (DenseNet-121 pre-trained weights)
- `evaluate` (BLEU, ROUGE)

### Hardware

All experiments were run on **Google Colab A100 (40GB / 80GB)** with bfloat16 training.  
Estimated compute per experiment: 8–12 hours on A100 40GB.

### Dataset

We use the **IU-Xray** dataset with the standard **R2Gen split**:
- Train: 2,069 samples | Val: 296 samples | Test: 590 samples
- Each sample: dual-view chest X-rays (frontal + lateral) paired with findings paragraph

**Download:**
```
Academic Torrents: https://academictorrents.com/details/5a3a439df24931f410fac2698b87b050203d9467d
HuggingFace: https://huggingface.co/datasets/ChayanM/IUXray-Data-Train-Test
```

Place data under `data/` following the structure in `data/README.md`.

### Base Models

```python
# Qwen2-VL-7B (primary base model)
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct"
)

# DenseNet-121 classifier (auxiliary)
import torchxrayvision as xrv
classifier = xrv.models.DenseNet(weights="densenet121-res224-all")
```

---

## Reproducing Results

### Experiment 1: LoRA Baseline

```bash
python experiments/baseline_lora.py \
  --data_dir ./data \
  --output_dir ./checkpoints/baseline \
  --base_model Qwen/Qwen2-VL-7B-Instruct \
  --lora_rank 16 \
  --lora_alpha 32 \
  --epochs 10 \
  --batch_size 4
```

Expected: BLEU-4 ≈ 8.71, ROUGE-L ≈ 30.41

### Experiment 2: NN Retrieval (Zero Training, ~30 seconds)

```bash
python experiments/nn_retrieval.py \
  --data_dir ./data \
  --output_file ./results/nn_retrieval_predictions.json
```

Expected: BLEU-4 ≈ 9.08 — **no training required**

### Experiments 3–5

See individual scripts in `experiments/` for CAT, VFF, and MIMIC pre-training configurations.

### Evaluation

```bash
# NLG metrics (BLEU, ROUGE, METEOR)
python evaluation/compute_nlg.py \
  --predictions ./results/predictions.json \
  --references ./data/test_references.json

# Clinical metrics (CheXbert, RadGraph)
python evaluation/compute_clinical.py \
  --predictions ./results/predictions.json \
  --references ./data/test_references.json
```

---

## LoRA Configuration

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## Implications

1. **Report NN retrieval baselines.** On IU-Xray, any model claiming improvement should be compared against NN retrieval. This is a low-cost, high-information sanity check.

2. **Supplement BLEU with clinical metrics.** BLEU rewards template matching; CheXbert and RadGraph measure diagnostic entity overlap. These metrics can disagree substantially (VFF case: worst BLEU, best clinical scores).

3. **Acknowledge data scale requirements.** General-purpose VLMs fine-tuned on 2,069 samples cannot match task-specific models pre-trained on 213K+ samples. Scale matters more than fine-tuning method.

---

## Citation

If you find this work useful, please cite:

```bibtex
@techreport{shen2025vlm,
  title     = {An Empirical Study of Vision-Language Model Fine-tuning 
               for Radiology Report Generation: Why Retrieval Baselines 
               Win on Small-Scale Medical Benchmarks},
  author    = {Shen, Zhengbo and Lai, Bangheng},
  institution = {Wenzhou-Kean University},
  year      = {2025},
  url       = {YOUR_ZENODO_OR_ARXIV_LINK}
}

@inproceedings{chen2020generating,
  title     = {Generating Radiology Reports via 
               Memory-driven Transformer},
  author    = {Chen, Zhihong and Song, Yan and 
               Chang, Tsung-Hui and Wan, Xiang},
  booktitle = {Proceedings of the 2020 Conference on 
               Empirical Methods in Natural Language 
               Processing (EMNLP)},
  year      = {2020}
}
```

---

## Acknowledgments

We thank the IAINLP Institute at Wenzhou-Kean University, where this research direction was originally proposed. All experiments were conducted independently by the authors using personal computational resources (Google Colab).

We also thank the authors of [R2Gen](https://github.com/cuhksz-nlp/R2Gen) for the dataset split, [torchxrayvision](https://github.com/mlmed/torchxrayvision) for pre-trained DenseNet weights, and [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) for the base model.

---

## License

This project is released under the [MIT License](LICENSE).

The IU-Xray dataset is subject to its own terms of use. Please refer to the original dataset source for licensing information.
