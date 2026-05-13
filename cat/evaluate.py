"""
Disease-Aware VLM 评估

Usage:
    python evaluate.py                          # 自动用 CheXNet 疾病引导
    python evaluate.py --no_disease             # 纯 baseline (无引导)
    python evaluate.py --ablation               # 消融实验 (4组对比)
    python evaluate.py --max_samples 50         # 快速测试
    python evaluate.py --threshold 0.3          # 调整疾病预测阈值
"""
import os
import json
import argparse
import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from config import TrainingConfig, DataConfig, GenerationConfig
from data_utils import (
    load_r2gen_data, clean_report_text,
    load_cat_hints, disease_findings_to_prompt,
    SYSTEM_PROMPT, USER_PROMPT_BASIC,
)


# ============================================================
# 1. 指标
# ============================================================

def compute_bleu(refs, hyps):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    r = [[ref.split()] for ref in refs]
    h = [hyp.split() for hyp in hyps]
    smooth = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        w = tuple([1.0/n]*n + [0.0]*(4-n))
        try: scores[f"BLEU-{n}"] = round(corpus_bleu(r, h, weights=w, smoothing_function=smooth)*100, 2)
        except: scores[f"BLEU-{n}"] = 0.0
    return scores

def compute_rouge(refs, hyps):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    s = {"ROUGE-1":[],"ROUGE-2":[],"ROUGE-L":[]}
    for ref, hyp in zip(refs, hyps):
        r = scorer.score(ref, hyp)
        s["ROUGE-1"].append(r["rouge1"].fmeasure)
        s["ROUGE-2"].append(r["rouge2"].fmeasure)
        s["ROUGE-L"].append(r["rougeL"].fmeasure)
    return {k: round(np.mean(v)*100, 2) for k,v in s.items()}

def compute_meteor(refs, hyps):
    try:
        from nltk.translate.meteor_score import meteor_score as ms
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        scores = [ms([r.split()], h.split()) for r,h in zip(refs, hyps)]
        return {"METEOR": round(np.mean(scores)*100, 2)}
    except: return {"METEOR": 0.0}

def compute_all_metrics(refs, hyps):
    m = {}
    m.update(compute_bleu(refs, hyps))
    m.update(compute_rouge(refs, hyps))
    m.update(compute_meteor(refs, hyps))
    return m


# ============================================================
# 2. 模型加载
# ============================================================

def load_model(model_path, train_config):
    print(f"[Eval] Loading VLM + LoRA from {model_path}...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        train_config.model_name, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        train_config.model_name, trust_remote_code=True,
        min_pixels=train_config.image_min_pixels,
        max_pixels=train_config.image_max_pixels,
    )
    model = PeftModel.from_pretrained(base, model_path, is_trainable=False)
    model.eval()
    return model, processor


def load_disease_classifier(device):
    """加载 torchxrayvision 预训练分类器"""
    try:
        from disease_classifier import DiseaseClassifier
        return DiseaseClassifier(device=str(device))
    except ImportError:
        print("[Eval] ⚠️ torchxrayvision not installed. Run: pip install torchxrayvision")
        print("[Eval] Falling back to baseline (no disease guidance)")
        return None
    except Exception as e:
        print(f"[Eval] ⚠️ Failed to load disease classifier: {e}")
        return None


# ============================================================
# 3. 生成
# ============================================================

def build_dual_view_messages(frontal_path, lateral_path, user_text):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{frontal_path}"},
            {"type": "image", "image": f"file://{lateral_path}"},
            {"type": "text", "text": user_text},
        ]},
    ]


def generate_single(
    model, processor, frontal_path, lateral_path,
    gen_kwargs, disease_classifier=None, threshold=0.5,
    cat_hint_findings=None,
):
    user_text = USER_PROMPT_BASIC

    # 优先级: CAT hint (预计算) > CheXNet 即时分类 > 无 hint
    if cat_hint_findings is not None:
        # CAT 模式: 用预计算的分类器 hint, 和训练时完全一致
        hint = disease_findings_to_prompt(cat_hint_findings)
        user_text = USER_PROMPT_BASIC + " " + hint
    elif disease_classifier is not None:
        # 老模式: CheXNet 即时预测
        hint = disease_classifier.get_prompt_hint(
            frontal_path, lateral_path, threshold=threshold
        )
        user_text = USER_PROMPT_BASIC + " " + hint

    messages = build_dual_view_messages(frontal_path, lateral_path, user_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    return processor.decode(out[0][input_len:], skip_special_tokens=True)


def generate_reports(
    model, processor, test_data, images_dir, gen_config,
    disease_classifier=None, threshold=0.5, max_samples=None,
    cat_hint_dict=None,
):
    gen_kwargs = {
        "num_beams": gen_config.num_beams,
        "length_penalty": gen_config.length_penalty,
        "repetition_penalty": gen_config.repetition_penalty,
        "no_repeat_ngram_size": gen_config.no_repeat_ngram_size,
        "early_stopping": gen_config.early_stopping,
        "max_new_tokens": gen_config.max_new_tokens,
        "min_new_tokens": gen_config.min_new_tokens,
        "do_sample": False,
    }

    ids, refs, hyps = [], [], []
    samples = test_data[:max_samples] if max_samples else test_data
    skipped = 0

    for item in tqdm(samples, desc="Generating"):
        report = clean_report_text(item.get("report", ""))
        img_paths = item.get("image_path", [])
        sample_id = item.get("id", "")

        if len(report) < 15 or len(img_paths) < 2:
            skipped += 1
            continue

        frontal = os.path.join(images_dir, img_paths[0])
        lateral = os.path.join(images_dir, img_paths[1])

        if not os.path.exists(frontal) or not os.path.exists(lateral):
            skipped += 1
            continue

        # CAT: 从预计算字典查 hint
        cat_hint_findings = None
        if cat_hint_dict is not None:
            cat_hint_findings = list(cat_hint_dict.get(sample_id, []))

        try:
            hyp = generate_single(
                model, processor, frontal, lateral, gen_kwargs,
                disease_classifier=disease_classifier, threshold=threshold,
                cat_hint_findings=cat_hint_findings,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hyp = "normal findings."
        except Exception as e:
            hyp = "normal findings."

        ids.append(sample_id)
        refs.append(report)
        hyps.append(clean_report_text(hyp))

    if skipped > 0:
        print(f"[Eval] Skipped {skipped}")

    return ids, refs, hyps


# ============================================================
# 4. 消融实验
# ============================================================

def run_ablation(model, processor, test_data, images_dir, device, disease_classifier=None):
    """
    消融实验 (4 组):
    1. Full: CheXNet disease guidance + beam4
    2. Baseline: 无引导 + beam4
    3. Greedy: 无引导 + beam1
    4. Disease (threshold=0.3): 更低阈值 → 更多发现
    """
    results = {}
    max_samples = min(200, len(test_data))

    configs = [
        ("Full (CheXNet+beam4)",     disease_classifier, 0.5, {}),
        ("Disease (threshold=0.3)",  disease_classifier, 0.3, {}),
        ("Baseline (beam=4)",        None,               0.5, {}),
        ("Greedy (beam=1)",          None,               0.5, {"num_beams": 1}),
    ]

    for i, (name, dc, thresh, overrides) in enumerate(configs):
        if dc is None and "CheXNet" in name:
            continue
        print(f"\n[Ablation {i+1}/{len(configs)}] {name}")
        gen_config = GenerationConfig()
        for k, v in overrides.items():
            setattr(gen_config, k, v)

        _, r, h = generate_reports(
            model, processor, test_data, images_dir, gen_config,
            disease_classifier=dc, threshold=thresh,
            max_samples=max_samples,
        )
        results[name] = compute_all_metrics(r, h)

    return results


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/content/drive/MyDrive/medk_lora_r2gen/lora_cat",
                        help="CAT 模型默认路径 lora_cat; 老 oracle 模型用 lora_discourse")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="CheXNet 阈值 (CAT 模式下忽略)")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no_disease", action="store_true",
                        help="Disable all disease guidance (baseline)")
    parser.add_argument("--use_cat_hints", action="store_true",
                        help="Use pre-computed CAT classifier hints (推荐)")
    parser.add_argument("--cat_hint_file", type=str, default=None,
                        help="Path to classifier_hints.json")
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    args = parser.parse_args()

    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except: pass

    train_config = TrainingConfig()
    data_config = DataConfig()

    _, _, test_data = load_r2gen_data(data_config)
    print(f"[Eval] Test: {len(test_data)} (R2Gen standard split)")

    model, processor = load_model(args.model_path, train_config)
    device = next(model.parameters()).device

    # ========================================================
    # Hint 源选择: CAT (优先) > CheXNet > 无
    # ========================================================
    cat_hint_dict = None
    disease_classifier = None

    if args.use_cat_hints:
        cat_file = args.cat_hint_file or os.path.join(
            train_config.output_dir, "classifier_hints.json"
        )
        cat_data = load_cat_hints(cat_file)
        cat_hint_dict = cat_data["test"]
        print(f"[Eval] Using CAT hints (classifier-predicted, train-test aligned)")
        print(f"[Eval] CAT hints loaded for {len(cat_hint_dict)} test samples")
        mode = "CAT (Classifier-Aligned Training, same hint source as training)"
    elif not args.no_disease:
        disease_classifier = load_disease_classifier(device)
        if disease_classifier:
            mode = f"CheXNet (threshold={args.threshold}, ⚠️ train-test mismatch)"
        else:
            mode = "Baseline (CheXNet unavailable)"
    else:
        mode = "Baseline (no hint)"

    print(f"\n[Eval] Mode: {mode}")

    gen_config = GenerationConfig()
    print(f"[Eval] Config: beam={gen_config.num_beams}, max_tokens={gen_config.max_new_tokens}")

    # 主评估
    sample_ids, refs, hyps = generate_reports(
        model, processor, test_data, data_config.images_dir, gen_config,
        disease_classifier=disease_classifier, threshold=args.threshold,
        max_samples=args.max_samples,
        cat_hint_dict=cat_hint_dict,
    )

    metrics = compute_all_metrics(refs, hyps)

    ref_len = np.mean([len(r.split()) for r in refs])
    hyp_len = np.mean([len(h.split()) for h in hyps])

    print(f"\n{'='*70}")
    print(f" EVALUATION RESULTS ({len(refs)} samples)")
    print(f" Data: R2Gen standard test split")
    print(f" Mode: {mode}")
    print(f"{'='*70}")
    for m, s in sorted(metrics.items()):
        print(f"  {m:<15} {s:>10.2f}")
    print(f"\n  Ref length: {ref_len:.1f} | Gen length: {hyp_len:.1f}")

    examples = []
    for i in range(min(5, len(refs))):
        examples.append({"id": sample_ids[i], "reference": refs[i], "generated": hyps[i]})
        print(f"\n--- {sample_ids[i]} ---")
        print(f"  Ref: {refs[i][:120]}")
        print(f"  Gen: {hyps[i][:120]}")

    # 消融实验
    ablation = {}
    if args.ablation:
        ablation = run_ablation(
            model, processor, test_data, data_config.images_dir, device,
            disease_classifier=disease_classifier,
        )
        print(f"\n{'='*70}")
        print(f" ABLATION RESULTS")
        print(f"{'='*70}")
        header = f"  {'Method':<30}"
        for m in ["BLEU-4", "ROUGE-L", "METEOR"]:
            header += f" {m:>10}"
        print(header)
        print(f"  {'-'*60}")
        for method, scores in ablation.items():
            row = f"  {method:<30}"
            for m in ["BLEU-4", "ROUGE-L", "METEOR"]:
                row += f" {scores.get(m,0):>10.2f}"
            print(row)

    # 保存
    output = {
        "metrics": metrics,
        "mode": mode,
        "data_split": "R2Gen standard",
        "ablation": ablation,
        "examples": examples,
        "config": {
            "model_path": args.model_path,
            "model_name": train_config.model_name,
            "num_samples": len(refs),
            "generation_config": vars(gen_config),
            "disease_threshold": args.threshold if disease_classifier else None,
            "ref_avg_length": round(ref_len, 1),
            "hyp_avg_length": round(hyp_len, 1),
        }
    }
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Saved to {args.output_file}")


if __name__ == "__main__":
    main()
