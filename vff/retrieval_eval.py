"""
Week 2 Day 1: Retrieval-augmented ICL validation (no training)

流程:
1. 加载预计算的 DenseNet features (train + test)
2. 对每个 test 样本, 用 cosine similarity 检索 train 集 top-3
3. 把 top-3 的 GT 报告作为 few-shot examples 组装 prompt
4. 用 zero-shot Qwen2-VL 生成
5. 算 BLEU

Usage:
  python retrieval_eval.py
  python retrieval_eval.py --top_k 3 --max_samples 100  # 快速测试
"""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from config import TrainingConfig, DataConfig
from data_utils import load_r2gen_data, clean_report_text


SYSTEM_PROMPT = """You are a radiologist writing the FINDINGS section of a chest X-ray report.
Output ONLY the findings text as plain prose (1-3 sentences).
Do NOT include any headers, markdown, patient information, or "impression" sections.
Use standard radiology language."""


def build_prompt_with_retrieval(retrieved_reports):
    examples_text = "Below are example findings from 3 similar chest X-rays. "
    examples_text += "Write your findings in EXACTLY the same style: "
    examples_text += "short declarative sentences, no 'in the frontal view', "
    examples_text += "no 'appears' or 'apparent', start directly with the observation.\n\n"
    for i, rep in enumerate(retrieved_reports, 1):
        examples_text += f"Example {i}: {rep}\n"
    examples_text += (
        "\nNow write findings for the current X-ray in the SAME short declarative style. "
        "Output ONLY the findings, no preamble. Maximum 3 sentences."
    )
    return examples_text


def cosine_topk(query_feat, db_feats, k=3):
    """
    query_feat: [1024]
    db_feats: [N, 1024]
    返回: top-k indices 和 similarities
    """
    q = query_feat / (np.linalg.norm(query_feat) + 1e-8)
    d = db_feats / (np.linalg.norm(db_feats, axis=1, keepdims=True) + 1e-8)
    sims = d @ q  # [N]
    topk_idx = np.argsort(-sims)[:k]
    return topk_idx, sims[topk_idx]


def compute_bleu(refs, hyps):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    r = [[ref.split()] for ref in refs]
    h = [hyp.split() for hyp in hyps]
    smooth = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        w = tuple([1.0/n]*n + [0.0]*(4-n))
        try:
            scores[f"BLEU-{n}"] = round(corpus_bleu(r, h, weights=w, smoothing_function=smooth)*100, 2)
        except:
            scores[f"BLEU-{n}"] = 0.0
    return scores


def compute_rouge(refs, hyps):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    s = {"ROUGE-1":[],"ROUGE-2":[],"ROUGE-L":[]}
    for ref, hyp in zip(refs, hyps):
        sc = scorer.score(ref, hyp)
        s["ROUGE-1"].append(sc["rouge1"].fmeasure)
        s["ROUGE-2"].append(sc["rouge2"].fmeasure)
        s["ROUGE-L"].append(sc["rougeL"].fmeasure)
    return {k: round(np.mean(v)*100, 2) for k,v in s.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--feat_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="eval_retrieval.json")
    args = parser.parse_args()

    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    tc, dc = TrainingConfig(), DataConfig()

    # ============================================================
    # 1. 加载 DenseNet features 并建索引
    # ============================================================
    feat_file = args.feat_file or os.path.join(tc.output_dir, "densenet_feats.npz")
    print(f"[Retrieval] Loading features from {feat_file}")
    data = np.load(feat_file, allow_pickle=True)
    train_ids = [str(x) for x in data["train_ids"]]
    train_feats = data["train_feats"]  # [N_train, 1024]
    test_ids = [str(x) for x in data["test_ids"]]
    test_feats = data["test_feats"]    # [N_test, 1024]
    print(f"  Train: {train_feats.shape}, Test: {test_feats.shape}")

    # ============================================================
    # 2. 加载 train 集报告 (id → report 映射)
    # ============================================================
    raw_train, _, raw_test = load_r2gen_data(dc)
    train_id_to_report = {
        item["id"]: clean_report_text(item["report"])
        for item in raw_train
        if len(clean_report_text(item.get("report", ""))) >= 15
    }
    test_id_to_item = {item["id"]: item for item in raw_test}
    print(f"  Train reports: {len(train_id_to_report)}")

    # ============================================================
    # 3. 加载 Qwen2-VL (no LoRA, pure zero-shot)
    # ============================================================
    print(f"[Retrieval] Loading Qwen2-VL (zero-shot, no LoRA)")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        tc.model_name, device_map="auto", torch_dtype=torch.bfloat16)
    proc = AutoProcessor.from_pretrained(
        tc.model_name,
        min_pixels=tc.image_min_pixels,
        max_pixels=tc.image_max_pixels)
    model.eval()

    # ============================================================
    # 4. 对每个 test 样本: 检索 + 生成
    # ============================================================
    gen_kwargs = {
        "num_beams": 4,
        "max_new_tokens": 100,
        "min_new_tokens": 10,
        "do_sample": False,
        "length_penalty": 1.0,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }

    refs, hyps, sample_ids, retrieved_info = [], [], [], []
    samples_to_run = test_ids[:args.max_samples] if args.max_samples else test_ids

    for i, tid in enumerate(tqdm(samples_to_run, desc="Retrieval+Gen")):
        item = test_id_to_item.get(tid)
        if item is None:
            continue
        gt_report = clean_report_text(item.get("report", ""))
        if len(gt_report) < 15:
            continue

        frontal = os.path.join(dc.images_dir, item["image_path"][0])
        lateral = os.path.join(dc.images_dir, item["image_path"][1])
        if not os.path.exists(frontal) or not os.path.exists(lateral):
            continue

        # 检索 top-k
        query_feat = test_feats[i]
        topk_idx, topk_sims = cosine_topk(query_feat, train_feats, k=args.top_k)
        retrieved_reports = []
        retrieved_ids = []
        for idx in topk_idx:
            train_sample_id = train_ids[idx]
            rep = train_id_to_report.get(train_sample_id)
            if rep:
                retrieved_reports.append(rep)
                retrieved_ids.append(train_sample_id)
        if not retrieved_reports:
            continue

        # 组装 prompt
        user_text = build_prompt_with_retrieval(retrieved_reports)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{frontal}"},
                {"type": "image", "image": f"file://{lateral}"},
                {"type": "text", "text": user_text},
            ]},
        ]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = proc(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            hyp = proc.decode(out[0][input_len:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hyp = ""
        except Exception as e:
            print(f"[Err] {tid}: {e}")
            hyp = ""

        hyp = clean_report_text(hyp)
        refs.append(gt_report)
        hyps.append(hyp)
        sample_ids.append(tid)
        retrieved_info.append({
            "test_id": tid,
            "retrieved_ids": retrieved_ids,
            "similarities": [float(s) for s in topk_sims[:len(retrieved_ids)]],
        })

    # ============================================================
    # 5. 算 metrics
    # ============================================================
    print("\n" + "="*70)
    print(" NN baseline: top-1 retrieval as prediction (no generation)")
    print("="*70)
    nn_hyps = [retrieved_info[i]["retrieved_ids"][0] for i in range(len(refs))]
    nn_hyps = [train_id_to_report.get(rid, "") for rid in nn_hyps]
    nn_bleu = compute_bleu(refs, nn_hyps)
    for m, s in sorted(nn_bleu.items()):
        print(f"  NN {m}: {s}")
    print(f"\n{'='*70}")
    print(f" Retrieval-Augmented ICL Results (n={len(refs)})")
    print(f" Model: Qwen2-VL-7B-Instruct (zero-shot, no LoRA)")
    print(f" Retrieval: DenseNet features, top-{args.top_k}, cosine")
    print(f"{'='*70}")

    bleu = compute_bleu(refs, hyps)
    rouge = compute_rouge(refs, hyps)
    metrics = {**bleu, **rouge}
    for m, s in sorted(metrics.items()):
        print(f"  {m}: {s}")

    ref_len = np.mean([len(r.split()) for r in refs])
    hyp_len = np.mean([len(h.split()) for h in hyps])
    print(f"\n  Ref length: {ref_len:.1f} | Gen length: {hyp_len:.1f}")

    # 样本展示
    print(f"\n--- Sample generations ---")
    for i in range(min(5, len(refs))):
        print(f"\n[{sample_ids[i]}]")
        print(f"  Retrieved: {retrieved_info[i]['retrieved_ids']}")
        print(f"  Sims: {[f'{s:.3f}' for s in retrieved_info[i]['similarities']]}")
        print(f"  Ref: {refs[i][:120]}")
        print(f"  Gen: {hyps[i][:120]}")

    # 保存
    import json
    out = {
        "metrics": metrics,
        "config": {
            "top_k": args.top_k,
            "num_samples": len(refs),
            "ref_avg_length": round(ref_len, 1),
            "hyp_avg_length": round(hyp_len, 1),
            "model": "zero-shot Qwen2-VL-7B (no LoRA)",
            "retrieval": "DenseNet penultimate + cosine",
        },
        "examples": [
            {"id": sample_ids[i], "ref": refs[i], "gen": hyps[i],
             "retrieved": retrieved_info[i]}
            for i in range(min(10, len(refs)))
        ],
    }
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {args.output_file}")



if __name__ == "__main__":
    main()
    