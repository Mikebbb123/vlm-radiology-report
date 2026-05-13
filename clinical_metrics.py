"""
Clinical metrics for radiology report generation.

- CheXbert-F1: 14-label CheXpert disease extraction F1 (5-label and 14-label).
- RadGraph-F1: entity + relation extraction F1 (simple and partial).

ENVIRONMENT SETUP (one-time per Colab runtime):

    # 1. Pinned versions — radgraph's bundled allennlp calls tokenizer.encode_plus()
    #    which was REMOVED in transformers 5.x. Must use 4.x.
    !pip install -q 'transformers>=4.30,<5.0' 'tokenizers<0.20' f1chexbert radgraph appdirs

    # 2. CheXbert checkpoint — f1chexbert does NOT auto-download.
    #    Place the .pth file at /root/.cache/chexbert/chexbert.pth
    #    Sources:
    #      - Kaggle: positivecoder/chexbert-model-pth-file
    #      - Box:    https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9

    # 3. RadGraph model auto-downloads from HuggingFace on first call (~400MB).

USAGE (in notebook):
    from clinical_metrics import compute_clinical_metrics
    scores = compute_clinical_metrics(refs, hyps, device="cuda")

USAGE (CLI, rescoring a saved predictions file):
    python clinical_metrics.py --input preds_nn_top1.json --output clin_nn.json
"""
import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional

CHEXBERT_CKPT = Path(os.path.expanduser("~/.cache/chexbert/chexbert.pth"))


# ---------------------------------------------------------------------------
# Setup checks
# ---------------------------------------------------------------------------

def check_chexbert_checkpoint() -> bool:
    """Return True if the CheXbert checkpoint is where f1chexbert expects it."""
    if not CHEXBERT_CKPT.exists():
        print(
            f"[Clinical] ERROR: CheXbert checkpoint not found at {CHEXBERT_CKPT}\n"
            f"           Download from Kaggle (positivecoder/chexbert-model-pth-file)\n"
            f"           or Stanford Box (c3stck6w6dol3h36grdc97xoydzxd7w9), then:\n"
            f"           mkdir -p {CHEXBERT_CKPT.parent} && mv chexbert.pth {CHEXBERT_CKPT}"
        )
        return False
    size_mb = CHEXBERT_CKPT.stat().st_size / 1024 / 1024
    if size_mb < 100:
        print(f"[Clinical] WARNING: {CHEXBERT_CKPT} is only {size_mb:.0f}MB "
              f"(expected ~500MB). File may be corrupted.")
    return True


def check_transformers_version() -> None:
    """Warn if transformers >=5.0 (breaks radgraph's bundled allennlp)."""
    import transformers
    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        print(
            f"[Clinical] WARNING: transformers=={transformers.__version__} detected.\n"
            f"           RadGraph's bundled allennlp requires transformers <5.0 "
            f"(needs the removed `encode_plus` API).\n"
            f"           Run: pip install 'transformers>=4.30,<5.0' 'tokenizers<0.20'"
        )


# ---------------------------------------------------------------------------
# CheXbert
# ---------------------------------------------------------------------------

def compute_chexbert_f1(
    refs: List[str],
    hyps: List[str],
    device: str = "cuda",
) -> Dict[str, float]:
    from f1chexbert import F1CheXbert

    if not check_chexbert_checkpoint():
        raise FileNotFoundError(f"Missing {CHEXBERT_CKPT}")

    scorer = F1CheXbert(device=device)
    # Returns: (accuracy, accuracy_per_sample, chexbert_all, chexbert_5)
    result = scorer(hyps=hyps, refs=refs)

    if len(result) == 4:
        _, _, report_all, report_5 = result
    else:
        raise ValueError(
            f"Unexpected F1CheXbert return length {len(result)}. Raw: {result}"
        )

    def pct(report_dict, split_key, score_key):
        if split_key not in report_dict:
            fallback = "weighted avg" if split_key == "micro avg" else "macro avg"
            print(f"[Clinical] Note: '{split_key}' missing, using '{fallback}'")
            return round(report_dict[fallback][score_key] * 100, 2)
        return round(report_dict[split_key][score_key] * 100, 2)

    return {
        "CheXbert-5 (micro-F1)":  pct(report_5,   "micro avg", "f1-score"),
        "CheXbert-5 (macro-F1)":  pct(report_5,   "macro avg", "f1-score"),
        "CheXbert-14 (micro-F1)": pct(report_all, "micro avg", "f1-score"),
        "CheXbert-14 (macro-F1)": pct(report_all, "macro avg", "f1-score"),
    }


# ---------------------------------------------------------------------------
# RadGraph
# ---------------------------------------------------------------------------

def compute_radgraph_f1(
    refs: List[str],
    hyps: List[str],
    cuda: Optional[int] = 0,
) -> Dict[str, float]:
    from radgraph import F1RadGraph

    # Partial = entity + relation-modified entity. Headline number.
    scorer = F1RadGraph(reward_level="partial", cuda=cuda)
    result_partial = scorer(hyps=hyps, refs=refs)
    mean_partial = result_partial[0] if isinstance(result_partial, tuple) else result_partial

    # Simple = entity match only
    scorer_simple = F1RadGraph(reward_level="simple", cuda=cuda)
    result_simple = scorer_simple(hyps=hyps, refs=refs)
    mean_simple = result_simple[0] if isinstance(result_simple, tuple) else result_simple

    return {
        "RadGraph-Simple":  round(float(mean_simple)  * 100, 2),
        "RadGraph-Partial": round(float(mean_partial) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Combined entry point — tracebacks are VISIBLE, not swallowed
# ---------------------------------------------------------------------------

def compute_clinical_metrics(
    refs: List[str],
    hyps: List[str],
    device: str = "cuda",
    cuda: Optional[int] = 0,
    strict: bool = False,
) -> Dict[str, float]:
    """
    Compute CheXbert-F1 + RadGraph-F1.

    Args:
        refs, hyps: same-length lists of reference / generated reports
        device:     "cuda" or "cpu" for CheXbert
        cuda:       int device id for RadGraph; None = CPU
        strict:     if True, re-raise on failure; else print traceback and continue
    """
    assert len(refs) == len(hyps), f"len mismatch: {len(refs)} vs {len(hyps)}"

    check_transformers_version()
    results: Dict[str, float] = {}

    # --- CheXbert ---
    print(f"[Clinical] Computing CheXbert-F1 on {len(refs)} samples...")
    try:
        chex = compute_chexbert_f1(refs, hyps, device=device)
        results.update(chex)
        for k, v in chex.items():
            print(f"           {k}: {v}")
    except Exception:
        print("[Clinical] CheXbert FAILED with traceback:")
        traceback.print_exc()
        if strict:
            raise

    # --- RadGraph ---
    print(f"\n[Clinical] Computing RadGraph-F1 on {len(refs)} samples...")
    try:
        rg = compute_radgraph_f1(refs, hyps, cuda=cuda)
        results.update(rg)
        for k, v in rg.items():
            print(f"           {k}: {v}")
    except Exception:
        print("[Clinical] RadGraph FAILED with traceback:")
        traceback.print_exc()
        if strict:
            raise

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,
                        help="JSON: list of {id, reference, generated}")
    parser.add_argument("--output", default=None,
                        help="Optional JSON to save scores")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda",   type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    refs = [d["reference"] for d in data]
    hyps = [d["generated"] for d in data]
    print(f"[Clinical] Loaded {len(refs)} pairs from {args.input}")

    scores = compute_clinical_metrics(
        refs, hyps, device=args.device, cuda=args.cuda, strict=args.strict,
    )

    print("\n" + "=" * 60)
    print(f" Clinical metrics on {len(refs)} samples")
    print(f" Source: {args.input}")
    print("=" * 60)
    if not scores:
        print("  (no metrics computed — see tracebacks above)")
        sys.exit(1)
    for k, v in scores.items():
        print(f"  {k:<28} {v:>8.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\nSaved to {args.output}")
