import numpy as np
import json
import sys
import os
sys.path.insert(0, '/content/drive/MyDrive/medvlm')
from config import TrainingConfig, DataConfig
from data_utils import load_r2gen_data, clean_report_text
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

tc, dc = TrainingConfig(), DataConfig()

# 加载 features
feat_file = os.path.join(tc.output_dir, "densenet_feats.npz")
data = np.load(feat_file, allow_pickle=True)
train_ids = [str(x) for x in data["train_ids"]]
train_feats = data["train_feats"]
test_ids = [str(x) for x in data["test_ids"]]
test_feats = data["test_feats"]

# 加载 GT reports
raw_train, _, raw_test = load_r2gen_data(dc)
train_id_to_report = {
    item["id"]: clean_report_text(item["report"])
    for item in raw_train
    if len(clean_report_text(item.get("report",""))) >= 15
}
test_id_to_item = {item["id"]: item for item in raw_test}

# Normalize features once
train_norm = train_feats / (np.linalg.norm(train_feats, axis=1, keepdims=True) + 1e-8)

refs, hyps_nn1 = [], []
for i, tid in enumerate(test_ids):
    item = test_id_to_item.get(tid)
    if item is None:
        continue
    gt = clean_report_text(item.get("report",""))
    if len(gt) < 15:
        continue
    # 检查图像存在 (和之前 pipeline 一致)
    frontal = os.path.join(dc.images_dir, item["image_path"][0])
    lateral = os.path.join(dc.images_dir, item["image_path"][1])
    if not os.path.exists(frontal) or not os.path.exists(lateral):
        continue

    q = test_feats[i] / (np.linalg.norm(test_feats[i]) + 1e-8)
    sims = train_norm @ q
    top1_idx = int(np.argmax(sims))
    top1_id = train_ids[top1_idx]
    pred = train_id_to_report.get(top1_id, "")
    if not pred:
        continue
    refs.append(gt)
    hyps_nn1.append(pred)

# 算 BLEU
r_bleu = [[x.split()] for x in refs]
h_bleu = [x.split() for x in hyps_nn1]
smooth = SmoothingFunction().method1
print(f"\n{'='*70}")
print(f" NN Baseline (top-1 retrieval) on FULL test set (n={len(refs)})")
print(f"{'='*70}")
for n in range(1, 5):
    w = tuple([1.0/n]*n + [0.0]*(4-n))
    score = corpus_bleu(r_bleu, h_bleu, weights=w, smoothing_function=smooth) * 100
    print(f"  BLEU-{n}: {score:.2f}")

# ROUGE
rs = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
rouge1, rouge2, rougeL = [], [], []
for ref, hyp in zip(refs, hyps_nn1):
    sc = rs.score(ref, hyp)
    rouge1.append(sc["rouge1"].fmeasure)
    rouge2.append(sc["rouge2"].fmeasure)
    rougeL.append(sc["rougeL"].fmeasure)
print(f"  ROUGE-1: {np.mean(rouge1)*100:.2f}")
print(f"  ROUGE-2: {np.mean(rouge2)*100:.2f}")
print(f"  ROUGE-L: {np.mean(rougeL)*100:.2f}")

ref_len = np.mean([len(r.split()) for r in refs])
hyp_len = np.mean([len(h.split()) for h in hyps_nn1])
print(f"\n  Ref length: {ref_len:.1f} | Gen length: {hyp_len:.1f}")