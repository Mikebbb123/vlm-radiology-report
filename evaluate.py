"""
Disease-Aware VLM 评估 (VFF / CAT / CheXNet / Baseline)

Hint sources (priority): VFF > CAT > CheXNet > None

Usage:
    # Baseline (LoRA only, no hint)
    python evaluate.py --no_disease --model_path .../lora_discourse \
                       --output_file eval_baseline.json

    # CAT (classifier-aligned hints via precomputed dict)
    python evaluate.py --use_cat_hints --model_path .../lora_cat \
                       --output_file eval_cat.json

    # VFF (visual feature fusion via gated cross-attention)
    python evaluate.py --use_vff --model_path .../lora_vff \
                       --output_file eval_vff.json

    # CheXNet live hints (default if no flag given)
    python evaluate.py --model_path .../lora_discourse \
                       --output_file eval_chexnet.json

    # Ablation / quick smoke test
    python evaluate.py --ablation
    python evaluate.py --max_samples 50

Output files per run:
    <output_file>                 — summary (metrics + config + first 5 examples)
    <output_file>_preds.json      — ALL per-sample {id, reference, generated, is_error}
                                    Feed this directly to clinical_metrics.py:
                                    python clinical_metrics.py --input <output_file>_preds.json
"""
import os
import json
import argparse
import traceback
from collections import Counter
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from config import TrainingConfig, DataConfig, GenerationConfig
from data_utils import (
    load_r2gen_data, clean_report_text,
    load_cat_hints, load_densenet_feats, disease_findings_to_prompt,
    SYSTEM_PROMPT, USER_PROMPT_BASIC,
)
from visual_feature_fusion import VisualFeatureFusion, install_vff_hook


# ============================================================
# 1. Metrics
# ============================================================

def compute_bleu(refs, hyps):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    r = [[ref.split()] for ref in refs]
    h = [hyp.split() for hyp in hyps]
    smooth = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        w = tuple([1.0 / n] * n + [0.0] * (4 - n))
        try:
            scores[f"BLEU-{n}"] = round(
                corpus_bleu(r, h, weights=w, smoothing_function=smooth) * 100, 2
            )
        except Exception:
            scores[f"BLEU-{n}"] = 0.0
    return scores


def compute_rouge(refs, hyps):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    s = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": []}
    for ref, hyp in zip(refs, hyps):
        r = scorer.score(ref, hyp)
        s["ROUGE-1"].append(r["rouge1"].fmeasure)
        s["ROUGE-2"].append(r["rouge2"].fmeasure)
        s["ROUGE-L"].append(r["rougeL"].fmeasure)
    return {k: round(np.mean(v) * 100, 2) for k, v in s.items()}


def compute_meteor(refs, hyps):
    try:
        from nltk.translate.meteor_score import meteor_score as ms
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        scores = [ms([r.split()], h.split()) for r, h in zip(refs, hyps)]
        return {"METEOR": round(np.mean(scores) * 100, 2)}
    except Exception:
        return {"METEOR": 0.0}


def compute_all_metrics(refs, hyps):
    m = {}
    m.update(compute_bleu(refs, hyps))
    m.update(compute_rouge(refs, hyps))
    m.update(compute_meteor(refs, hyps))
    return m


# ============================================================
# 2. Model loading
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
    """Load torchxrayvision-based DenseNet classifier (for CheXNet live hints)."""
    try:
        from disease_classifier import DiseaseClassifier
        return DiseaseClassifier(device=str(device))
    except ImportError:
        print("[Eval] ⚠️ torchxrayvision not installed. "
              "Run: pip install torchxrayvision")
        print("[Eval] Falling back to baseline (no disease guidance)")
        return None
    except Exception as e:
        print(f"[Eval] ⚠️ Failed to load disease classifier: {e}")
        return None


# ============================================================
# 3. Generation
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
    vff_medical_feat=None,
):
    user_text = USER_PROMPT_BASIC

    # Priority: CAT hint > CheXNet > no hint. VFF is orthogonal (visual channel).
    if cat_hint_findings is not None:
        hint = disease_findings_to_prompt(cat_hint_findings)
        user_text = USER_PROMPT_BASIC + " " + hint
    elif disease_classifier is not None:
        hint = disease_classifier.get_prompt_hint(
            frontal_path, lateral_path, threshold=threshold
        )
        user_text = USER_PROMPT_BASIC + " " + hint

    messages = build_dual_view_messages(frontal_path, lateral_path, user_text)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # VFF: stash medical feature + grid_thw on model instance for the hook to pick up
    if vff_medical_feat is not None:
        feat_tensor = torch.from_numpy(
            np.asarray(vff_medical_feat, dtype=np.float32)
        ).unsqueeze(0).to(model.device)  # [1, 1024]
        model._vff_medical_feats = feat_tensor
        model._vff_grid_thw = inputs["image_grid_thw"]

    try:
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    finally:
        if hasattr(model, "_vff_medical_feats"):
            del model._vff_medical_feats
        if hasattr(model, "_vff_grid_thw"):
            del model._vff_grid_thw

    return processor.decode(out[0][input_len:], skip_special_tokens=True)


ERROR_FALLBACK = "normal findings."


def generate_reports(
    model, processor, test_data, images_dir, gen_config,
    disease_classifier=None, threshold=0.5, max_samples=None,
    cat_hint_dict=None, vff_feat_dict=None,
) -> Tuple[List[str], List[str], List[str], List[bool], Dict[str, int]]:
    """
    Generate reports for every eligible test sample.

    Returns:
        ids:      sample ids for successfully processed samples (skipped ones excluded)
        refs:     ground-truth reports
        hyps:     generated reports (or ERROR_FALLBACK if generation failed)
        is_error: True if this sample hit an exception and used the fallback
        stats:    {skipped, oom_count, other_error_count, missing_vff_feat}
    """
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

    ids, refs, hyps, is_error = [], [], [], []
    samples = test_data[:max_samples] if max_samples else test_data
    stats = {
        "skipped": 0,
        "oom_count": 0,
        "other_error_count": 0,
        "missing_vff_feat": 0,
    }

    for item in tqdm(samples, desc="Generating"):
        report = clean_report_text(item.get("report", ""))
        img_paths = item.get("image_path", [])
        sample_id = item.get("id", "")

        if len(report) < 15 or len(img_paths) < 2:
            stats["skipped"] += 1
            continue

        frontal = os.path.join(images_dir, img_paths[0])
        lateral = os.path.join(images_dir, img_paths[1])
        if not os.path.exists(frontal) or not os.path.exists(lateral):
            stats["skipped"] += 1
            continue

        # CAT hint lookup
        cat_hint_findings = None
        if cat_hint_dict is not None:
            cat_hint_findings = list(cat_hint_dict.get(sample_id, []))

        # VFF feature lookup
        vff_feat = None
        if vff_feat_dict is not None:
            vff_feat = vff_feat_dict.get(sample_id)
            if vff_feat is None:
                stats["missing_vff_feat"] += 1
                vff_feat = np.zeros(1024, dtype=np.float32)

        err = False
        try:
            hyp = generate_single(
                model, processor, frontal, lateral, gen_kwargs,
                disease_classifier=disease_classifier, threshold=threshold,
                cat_hint_findings=cat_hint_findings,
                vff_medical_feat=vff_feat,
            )
        except torch.cuda.OutOfMemoryError:
            stats["oom_count"] += 1
            torch.cuda.empty_cache()
            hyp = ERROR_FALLBACK
            err = True
            tqdm.write(f"[Eval] OOM on {sample_id}, using fallback")
        except Exception as e:
            stats["other_error_count"] += 1
            hyp = ERROR_FALLBACK
            err = True
            tqdm.write(f"[Eval] Exception on {sample_id}: {type(e).__name__}: {e}")

        ids.append(sample_id)
        refs.append(report)
        hyps.append(clean_report_text(hyp))
        is_error.append(err)

    # Summary printouts
    if stats["skipped"]:
        print(f"[Eval] Skipped {stats['skipped']} samples "
              f"(missing images or too-short report)")
    if stats["oom_count"]:
        print(f"[Eval] OOM fallbacks: {stats['oom_count']}")
    if stats["other_error_count"]:
        print(f"[Eval] Other-exception fallbacks: {stats['other_error_count']}")
    if stats["missing_vff_feat"]:
        print(f"[Eval] VFF feature missing (→ zeros): {stats['missing_vff_feat']}")

    return ids, refs, hyps, is_error, stats


def print_generation_diagnostics(refs, hyps, is_error):
    """Print sanity stats on generated outputs (mode-collapse detector etc)."""
    n = len(hyps)
    ref_len = np.mean([len(r.split()) for r in refs]) if refs else 0
    hyp_len = np.mean([len(h.split()) for h in hyps]) if hyps else 0
    unique_hyps = len(set(hyps))
    unique_frac = unique_hyps / n if n else 0
    error_frac = sum(is_error) / n if n else 0
    top3 = Counter(hyps).most_common(3)

    print(f"\n{'-' * 60}")
    print(f" GENERATION DIAGNOSTICS")
    print(f"{'-' * 60}")
    print(f"  Samples:           {n}")
    print(f"  Ref avg length:    {ref_len:.1f} words")
    print(f"  Gen avg length:    {hyp_len:.1f} words")
    print(f"  Unique outputs:    {unique_hyps} ({unique_frac * 100:.1f}% of samples)")
    print(f"  Fallback rate:     {error_frac * 100:.1f}%")
    if n and unique_frac < 0.5:
        print(f"  ⚠️  WARNING: <50% unique outputs — possible mode collapse.")
    print(f"  Top-3 most common outputs:")
    for i, (text, count) in enumerate(top3, 1):
        print(f"    {i}. [{count}x] {text[:100]}")


# ============================================================
# 4. Persistence
# ============================================================

def save_predictions(
    path: str, ids: List[str], refs: List[str], hyps: List[str],
    is_error: List[bool] = None,
) -> None:
    """Save per-sample preds in the format clinical_metrics.py expects."""
    records = []
    for i, (sid, r, h) in enumerate(zip(ids, refs, hyps)):
        rec = {"id": sid, "reference": r, "generated": h}
        if is_error is not None:
            rec["is_error"] = bool(is_error[i])
        records.append(rec)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[Eval] Saved {len(records)} predictions to {path}")


def preds_path_for(output_file: str, suffix: str = "") -> str:
    """eval_baseline.json → eval_baseline_preds.json (+ optional _suffix)"""
    root, ext = os.path.splitext(output_file)
    tag = f"_{suffix}" if suffix else ""
    return f"{root}_preds{tag}{ext or '.json'}"


# ============================================================
# 5. Ablation
# ============================================================

def run_ablation(
    model, processor, test_data, images_dir, device,
    disease_classifier=None, preds_base: str = "eval_results.json",
):
    """4-config ablation on a 200-sample subset. Each config saves its own preds."""
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
        print(f"\n[Ablation {i + 1}/{len(configs)}] {name}")
        gen_config = GenerationConfig()
        for k, v in overrides.items():
            setattr(gen_config, k, v)

        ids, r, h, is_err, _ = generate_reports(
            model, processor, test_data, images_dir, gen_config,
            disease_classifier=dc, threshold=thresh,
            max_samples=max_samples,
        )
        results[name] = compute_all_metrics(r, h)

        tag = name.split()[0].lower().replace("(", "").replace(")", "")
        save_predictions(preds_path_for(preds_base, f"abl_{tag}"), ids, r, h, is_err)

    return results


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/content/drive/MyDrive/medk_lora_r2gen/lora_cat",
                        help="默认 lora_cat; baseline 用 lora_discourse; "
                             "VFF 模式会自动切到 lora_vff")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="CheXNet threshold (忽略于 CAT/VFF 模式)")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no_disease", action="store_true",
                        help="Disable all disease guidance (pure baseline)")
    parser.add_argument("--use_cat_hints", action="store_true")
    parser.add_argument("--cat_hint_file", type=str, default=None)
    parser.add_argument("--use_vff", action="store_true",
                        help="Enable Visual Feature Fusion")
    parser.add_argument("--densenet_feat_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    args = parser.parse_args()

    # VFF 模式自动切 model_path (除非用户显式指定了别的)
    if args.use_vff and args.model_path.endswith("lora_cat"):
        default_vff = "/content/drive/MyDrive/medk_lora_r2gen/lora_vff"
        args.model_path = default_vff
        print(f"[Eval] --use_vff 自动切换 model_path → {default_vff}")

    try:
        import nltk
        for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
            nltk.download(pkg, quiet=True)
    except Exception:
        pass

    train_config = TrainingConfig()
    data_config = DataConfig()

    _, _, test_data = load_r2gen_data(data_config)
    print(f"[Eval] Test: {len(test_data)} samples (R2Gen standard split)")

    model, processor = load_model(args.model_path, train_config)
    device = next(model.parameters()).device

    # ========================================================
    # Hint source routing: VFF > CAT > CheXNet > None
    # ========================================================
    cat_hint_dict = None
    disease_classifier = None
    vff_feat_dict = None

    if args.use_vff:
        feat_file = args.densenet_feat_file or os.path.join(
            train_config.output_dir, "densenet_feats.npz"
        )
        feat_data = load_densenet_feats(feat_file)
        vff_feat_dict = feat_data["test"]
        print(f"[Eval] VFF: loaded {len(vff_feat_dict)} test features")

        vff_adapter = VisualFeatureFusion(
            vlm_dim=3584, medical_dim=1024,
            n_heads=8, dropout=0.1, init_gate=0.0,
        ).to(device).to(torch.bfloat16)
        vff_state = os.path.join(args.model_path, "vff_adapter.pt")
        if not os.path.exists(vff_state):
            raise FileNotFoundError(f"VFF adapter state not found: {vff_state}")
        vff_adapter.load_state_dict(
            torch.load(vff_state, map_location=device, weights_only=True)
        )
        vff_adapter.eval()
        install_vff_hook(model, vff_adapter, spatial_merge_size=2)
        print(f"[Eval] VFF: loaded adapter, gate={vff_adapter.current_gate:.4f}")
        mode = f"VFF (Visual Feature Fusion, gate={vff_adapter.current_gate:.4f})"

    elif args.use_cat_hints:
        cat_file = args.cat_hint_file or os.path.join(
            train_config.output_dir, "classifier_hints.json"
        )
        cat_data = load_cat_hints(cat_file)
        cat_hint_dict = cat_data["test"]
        mode = "CAT (Classifier-Aligned Training)"
    elif not args.no_disease:
        disease_classifier = load_disease_classifier(device)
        mode = (f"CheXNet (threshold={args.threshold})"
                if disease_classifier else "Baseline (classifier unavailable)")
    else:
        mode = "Baseline (no hint)"

    print(f"\n[Eval] Mode: {mode}")

    gen_config = GenerationConfig()
    print(f"[Eval] Config: beam={gen_config.num_beams}, "
          f"max_tokens={gen_config.max_new_tokens}")

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------
    sample_ids, refs, hyps, is_error, gen_stats = generate_reports(
        model, processor, test_data, data_config.images_dir, gen_config,
        disease_classifier=disease_classifier, threshold=args.threshold,
        max_samples=args.max_samples,
        cat_hint_dict=cat_hint_dict,
        vff_feat_dict=vff_feat_dict,
    )

    # --------------------------------------------------------
    # SAVE PREDS FIRST — before any metric code that could fail
    # --------------------------------------------------------
    preds_file = preds_path_for(args.output_file)
    save_predictions(preds_file, sample_ids, refs, hyps, is_error)

    # --------------------------------------------------------
    # Diagnostics + metrics
    # --------------------------------------------------------
    print_generation_diagnostics(refs, hyps, is_error)

    metrics = compute_all_metrics(refs, hyps)

    ref_len = float(np.mean([len(r.split()) for r in refs])) if refs else 0.0
    hyp_len = float(np.mean([len(h.split()) for h in hyps])) if hyps else 0.0

    print(f"\n{'=' * 70}")
    print(f" EVALUATION RESULTS ({len(refs)} samples)")
    print(f" Data: R2Gen standard test split")
    print(f" Mode: {mode}")
    print(f"{'=' * 70}")
    for m, s in sorted(metrics.items()):
        print(f"  {m:<15} {s:>10.2f}")
    print(f"\n  Ref length: {ref_len:.1f} | Gen length: {hyp_len:.1f}")

    # First-5 preview
    print(f"\n{'-' * 60}")
    print(f" FIRST 5 EXAMPLES")
    print(f"{'-' * 60}")
    examples = []
    for i in range(min(5, len(refs))):
        examples.append({
            "id": sample_ids[i],
            "reference": refs[i],
            "generated": hyps[i],
        })
        print(f"\n--- {sample_ids[i]} ---")
        print(f"  Ref: {refs[i][:120]}")
        print(f"  Gen: {hyps[i][:120]}")

    # --------------------------------------------------------
    # Ablation (optional)
    # --------------------------------------------------------
    ablation = {}
    if args.ablation:
        ablation = run_ablation(
            model, processor, test_data, data_config.images_dir, device,
            disease_classifier=disease_classifier,
            preds_base=args.output_file,
        )
        print(f"\n{'=' * 70}")
        print(f" ABLATION RESULTS (200-sample subset)")
        print(f"{'=' * 70}")
        header = f"  {'Method':<30}"
        for m in ["BLEU-4", "ROUGE-L", "METEOR"]:
            header += f" {m:>10}"
        print(header)
        print(f"  {'-' * 60}")
        for method, scores in ablation.items():
            row = f"  {method:<30}"
            for m in ["BLEU-4", "ROUGE-L", "METEOR"]:
                row += f" {scores.get(m, 0):>10.2f}"
            print(row)

    # --------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------
    output = {
        "metrics": metrics,
        "mode": mode,
        "data_split": "R2Gen standard",
        "ablation": ablation,
        "examples": examples,
        "generation_stats": {
            **gen_stats,
            "unique_outputs": len(set(hyps)),
            "total_samples": len(hyps),
            "fallback_count": int(sum(is_error)),
        },
        "config": {
            "model_path": args.model_path,
            "model_name": train_config.model_name,
            "num_samples": len(refs),
            "generation_config": vars(gen_config),
            "disease_threshold": args.threshold if disease_classifier else None,
            "use_vff": args.use_vff,
            "use_cat_hints": args.use_cat_hints,
            "no_disease": args.no_disease,
            "ref_avg_length": round(ref_len, 1),
            "hyp_avg_length": round(hyp_len, 1),
        },
        "preds_file": preds_file,
    }
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Summary saved to {args.output_file}")
    print(f"[Eval] Per-sample preds at {preds_file}")
    print(f"[Eval] Next step (in transformers<5 env):")
    print(f"       python clinical_metrics.py --input {preds_file} \\")
    print(f"                                  --output {preds_file.replace('_preds.json', '_clin.json')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Eval] Interrupted by user. "
              "Any predictions already saved to disk remain available.")
    except Exception:
        print("[Eval] Fatal error — traceback follows:")
        traceback.print_exc()
        raise
