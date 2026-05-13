"""
Oracle 实验: 推理时用 GT 报告的疾病标签作为 hint

目的: 证明 disease-guided prompting 方法的上限
  如果 oracle >> baseline >> CheXNet, 说明:
  1. 方法本身有效 (oracle 证明)
  2. 瓶颈在分类器准确率 (CheXNet 域偏移)
  3. 需要在 IU-Xray 上微调分类器或换更好的分类器

Usage:
  python oracle_eval.py
  python oracle_eval.py --max_samples 50   # 快速测试
"""
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from config import TrainingConfig, DataConfig, GenerationConfig
from data_utils import (
    load_r2gen_data, clean_report_text,
    extract_disease_labels, disease_findings_to_prompt,
    SYSTEM_PROMPT, USER_PROMPT_BASIC,
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/content/drive/MyDrive/medk_lora_r2gen/lora_discourse")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="oracle_results.json")
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
    gen_config = GenerationConfig()

    _, _, test_data = load_r2gen_data(data_config)
    print(f"[Oracle] Test: {len(test_data)}")

    # 加载模型
    print("[Oracle] Loading model...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        train_config.model_name, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        train_config.model_name, trust_remote_code=True,
        min_pixels=train_config.image_min_pixels,
        max_pixels=train_config.image_max_pixels,
    )
    model = PeftModel.from_pretrained(base, args.model_path, is_trainable=False)
    model.eval()

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

    # ============================================================
    # 跑 3 组: Oracle / Baseline / Oracle-normal-only
    # ============================================================

    results_all = {}

    for mode_name, use_oracle in [("Oracle (GT labels)", True), ("Baseline (no hint)", False)]:
        print(f"\n[{mode_name}] Generating...")

        ids, refs, hyps = [], [], []
        samples = test_data[:args.max_samples] if args.max_samples else test_data
        skipped = 0

        for item in tqdm(samples, desc=mode_name):
            report = clean_report_text(item.get("report", ""))
            img_paths = item.get("image_path", [])
            sample_id = item.get("id", "")

            if len(report) < 15 or len(img_paths) < 2:
                skipped += 1
                continue

            frontal = os.path.join(data_config.images_dir, img_paths[0])
            lateral = os.path.join(data_config.images_dir, img_paths[1])
            if not os.path.exists(frontal) or not os.path.exists(lateral):
                skipped += 1
                continue

            # 构造 prompt
            if use_oracle:
                # Oracle: 从 GT 报告提取疾病标签
                findings = extract_disease_labels(report)
                hint = disease_findings_to_prompt(findings)
                user_text = USER_PROMPT_BASIC + " " + hint
            else:
                user_text = USER_PROMPT_BASIC

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{frontal}"},
                    {"type": "image", "image": f"file://{lateral}"},
                    {"type": "text", "text": user_text},
                ]},
            ]

            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                hyp = processor.decode(out[0][input_len:], skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hyp = "normal findings."
            except:
                hyp = "normal findings."

            ids.append(sample_id)
            refs.append(report)
            hyps.append(clean_report_text(hyp))

        metrics = compute_all_metrics(refs, hyps)
        ref_len = np.mean([len(r.split()) for r in refs])
        hyp_len = np.mean([len(h.split()) for h in hyps])
        results_all[mode_name] = {
            "metrics": metrics,
            "ref_len": round(ref_len, 1),
            "hyp_len": round(hyp_len, 1),
            "num_samples": len(refs),
        }

        print(f"\n  [{mode_name}] {len(refs)} samples")
        for m, s in sorted(metrics.items()):
            print(f"    {m:<15} {s:>10.2f}")
        print(f"    Ref: {ref_len:.1f} | Gen: {hyp_len:.1f}")

    # 对比输出
    print(f"\n{'='*70}")
    print(f" ORACLE vs BASELINE COMPARISON")
    print(f"{'='*70}")
    header = f"  {'Method':<30}"
    for m in ["BLEU-4", "ROUGE-L", "METEOR"]:
        header += f" {m:>10}"
    header += f" {'Gen Len':>10}"
    print(header)
    print(f"  {'-'*70}")
    for method, data in results_all.items():
        row = f"  {method:<30}"
        for m in ["BLEU-4", "ROUGE-L", "METEOR"]:
            row += f" {data['metrics'].get(m,0):>10.2f}"
        row += f" {data['hyp_len']:>10.1f}"
        print(row)

    # 保存
    with open(args.output_file, "w") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"\n[Oracle] Saved to {args.output_file}")


if __name__ == "__main__":
    main()
