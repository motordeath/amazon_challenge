import json
from pathlib import Path

with open('kaggle_test/abb-aar-ya-paar.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

exp1_code = """import os
import sys
import gc
import json
import time
import logging
import psutil
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', force=True)
logger = logging.getLogger(__name__)

def get_rss_mb():
    return psutil.Process().memory_info().rss / 1e6

WORK = Path('/kaggle/working/ml_amazon')
cfg_path = WORK / 'code/business_entity_resolution/src/business_entity_resolution/config.py'
if cfg_path.exists():
    text = cfg_path.read_text(encoding='utf-8')
    if 'token_max_df_abs_other: int = 80' in text:
        text = text.replace('token_max_df_abs_other: int = 80', 'token_max_df_abs_other: int = 400')
        cfg_path.write_text(text, encoding='utf-8')
        print('Successfully patched config.py to token_max_df_abs_other = 400')

SRC_ROOT = WORK / 'code/business_entity_resolution/src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from business_entity_resolution import config
config.DEFAULT_CONFIG.blocking.token_max_df_abs_other = 400
assert config.DEFAULT_CONFIG.blocking.token_max_df_abs_other == 400, 'Must be 400!'
print('Active token_max_df_abs_other:', config.DEFAULT_CONFIG.blocking.token_max_df_abs_other)

from business_entity_resolution.data import loader
from business_entity_resolution.pipeline.sampling import stratified_sample_s1, sample_report
from business_entity_resolution.normalization.run_normalize import normalize_source
from business_entity_resolution.pipeline.common import NORMALIZED_KEEP_COLS, load_normalize_block
from business_entity_resolution.blocking.candidate_generator import union_and_rank
from business_entity_resolution.blocking.recall_eval import blocking_recall
from business_entity_resolution.pipeline.labeling import label_candidates
from business_entity_resolution.features.build_feature_matrix import build_feature_matrix
from business_entity_resolution.models.train_lgbm import split_all_s1_ids, train_lgbm
from business_entity_resolution.models.threshold import tune_global_threshold, predict_matches
from business_entity_resolution.evaluation.report import full_validation_report

cfg = config.DEFAULT_CONFIG

print('=' * 75)
print('STARTING EXPERIMENT 1 EXECUTION')
print('Parameter Change: token_max_df_abs_other: 80 -> 400')
print('Sample Size: 500,001 (Same validation regime as baseline)')
print('=' * 75)

t_start = time.time()
report = {'experiment_id': 'EXP-001', 'config_changes': {'token_max_df_abs_other': 400}}

# Step 1: Loading
t0 = time.time()
print('\\n[1/7] Loading S1 and Ground Truth...')
s1 = loader.load_source(config.TRAIN_FILES['S1'])
gt = loader.load_ground_truth(config.TRAIN_GROUND_TRUTH)

sampled_ids = stratified_sample_s1(s1, gt, 500001, seed=42)
report['sample_distribution'] = sample_report(s1, gt, sampled_ids)
s1 = s1[s1['entity_id'].isin(sampled_ids)].reset_index(drop=True)
gt = gt[gt['source1_entity_id'].isin(sampled_ids)].reset_index(drop=True)
report['n_s1_sampled'] = len(s1)
print(f'Sampled {len(s1)} S1 entities in {time.time()-t0:.1f}s | RSS: {get_rss_mb():.0f}MB')

gt_long = loader.explode_ground_truth(gt)
del gt
gc.collect()

# Step 2: Normalization
t0 = time.time()
print('\\n[2/7] Normalizing S1...')
s1n = normalize_source(s1)
del s1
s1n = s1n[NORMALIZED_KEEP_COLS].copy()
gc.collect()
print(f'S1 normalized in {time.time()-t0:.1f}s | RSS: {get_rss_mb():.0f}MB')

# Step 3: Blocking
t0_block = time.time()
print('\\n[3/7] Blocking S2 and S3...')
cand2, s2_subset = load_normalize_block(s1n, config.TRAIN_FILES['S2'], 'S2', cfg.blocking)
cand3, s3_subset = load_normalize_block(s1n, config.TRAIN_FILES['S3'], 'S3', cfg.blocking)

all_candidates_uncapped = pd.concat([cand2, cand3], ignore_index=True)
del cand2, cand3
gc.collect()

t_block_elapsed = time.time() - t0_block
rss_block = get_rss_mb()
print(f'Blocking complete in {t_block_elapsed:.1f}s | RSS: {rss_block:.0f}MB')

print('\\nEvaluating Uncapped Blocking Recall...')
uncapped_recall_report = blocking_recall(all_candidates_uncapped, gt_long)
print(f"UNCAPPED Blocking Recall: {uncapped_recall_report['blocking_recall']:.4f} ({uncapped_recall_report['retrieved_true_pairs']}/{uncapped_recall_report['total_true_pairs']})")
print(f"UNCAPPED Candidate Count: {len(all_candidates_uncapped)}")

print(f'\\nApplying Top-K Cap (max_candidates_per_s1={cfg.blocking.max_candidates_per_s1})...')
capped_candidates = union_and_rank(all_candidates_uncapped, cfg.blocking.max_candidates_per_s1)
del all_candidates_uncapped
gc.collect()

capped_recall_report = blocking_recall(capped_candidates, gt_long)
print(f"CAPPED Blocking Recall: {capped_recall_report['blocking_recall']:.4f} ({capped_recall_report['retrieved_true_pairs']}/{capped_recall_report['total_true_pairs']})")
print(f"CAPPED Candidate Count: {len(capped_candidates)}")

report['uncapped_blocking_recall'] = uncapped_recall_report
report['capped_blocking_recall'] = capped_recall_report
report['n_candidate_pairs'] = len(capped_candidates)
report['blocking_runtime_sec'] = round(t_block_elapsed, 1)
report['blocking_peak_rss_mb'] = round(rss_block, 1)

# Step 4: Labeling
print('\\n[4/7] Labeling Candidate Pairs...')
labeled = label_candidates(capped_candidates, gt_long)
del capped_candidates
gc.collect()

n_pos = int((labeled['label'] == 1).sum())
n_neg = int((labeled['label'] == 0).sum())
report['n_positive_pairs'] = n_pos
report['n_negative_pairs'] = n_neg
print(f'Labeled pairs: Positive={n_pos} Negative={n_neg} (Ratio 1:{n_neg/max(n_pos,1):.2f})')

y = labeled['label'].to_numpy()
small_labeled = labeled[['source1_entity_id', 'candidate_entity_id', 'score']]
del labeled
gc.collect()

# Step 5: Features
t0_feat = time.time()
print('\\n[5/7] Building Feature Matrix...')
others_n = pd.concat([s2_subset, s3_subset], ignore_index=True)
del s2_subset, s3_subset
gc.collect()

rss_before_feat = get_rss_mb()
X, meta = build_feature_matrix(small_labeled, s1n, others_n, cfg.features, chunk_size=4_000_000)
del others_n, small_labeled
gc.collect()
rss_after_feat = get_rss_mb()

t_feat_elapsed = time.time() - t0_feat
report['feature_engineering_rss_mb'] = {
    'before': round(rss_before_feat, 1),
    'after': round(rss_after_feat, 1),
    'delta': round(rss_after_feat - rss_before_feat, 1)
}
report['feature_engineering_time_sec'] = round(t_feat_elapsed, 1)
print(f'Features computed in {t_feat_elapsed:.1f}s | Delta RSS: {rss_after_feat-rss_before_feat:.0f}MB')

# Step 6: LightGBM Training
print('\\n[6/7] Training LightGBM...')
groups = meta['source1_entity_id']
train_ids, val_ids = split_all_s1_ids(s1n['entity_id'], n_splits=5, fold=0, seed=cfg.random_seed)
train_mask = groups.isin(train_ids).to_numpy()
val_mask = groups.isin(val_ids).to_numpy()

X_train, X_val = X[train_mask], X[val_mask]
y_train, y_val = y[train_mask], y[val_mask]
meta_val = meta[val_mask].reset_index(drop=True)
del X, y
gc.collect()

t0_train = time.time()
model = train_lgbm(X_train, y_train, X_val, y_val, cfg.model)
t_train_elapsed = time.time() - t0_train
report['training_time_sec'] = round(t_train_elapsed, 1)
print(f'LightGBM trained in {t_train_elapsed:.1f}s')

# Step 7: Threshold Tuning & Validation
print('\\n[7/7] Threshold Tuning & Final Validation...')
val_s1_ids = sorted(val_ids)
gt_val = gt_long[gt_long['source1_entity_id'].isin(val_ids)]
gt_val_dict = gt_val.groupby('source1_entity_id')['matched_entity_id'].apply(set).to_dict()
gt_val_dict = {sid: gt_val_dict.get(sid, set()) for sid in val_s1_ids}

proba_val = model.predict_proba(X_val)[:, 1]
scored_val = meta_val.copy()
scored_val['proba'] = proba_val

thresh_report = tune_global_threshold(scored_val, gt_val_dict, val_s1_ids, cfg.threshold)
report['threshold'] = thresh_report['best']
tau = thresh_report['best']['tau']

val_predictions = predict_matches(scored_val, tau, apply_reverse_constraint=True)
full_report = full_validation_report(val_predictions, gt_val_dict, beta=cfg.threshold.beta)
report['validation'] = full_report

total_elapsed = time.time() - t_start
report['total_runtime_sec'] = round(total_elapsed, 1)

print('\\n' + '=' * 75)
print('EXPERIMENT 1 FINAL RESULTS (500,001 S1 Sample)')
print('=' * 75)
print(f"Uncapped Blocking Recall: {uncapped_recall_report['blocking_recall']:.4f}")
print(f"Capped Blocking Recall:   {capped_recall_report['blocking_recall']:.4f}")
print(f"Candidate Count:          {len(meta)}")
print(f"Positive Candidates:      {n_pos}")
print(f"Optimal Threshold (tau):  {tau:.4f}")
print(f"Macro F0.5:               {full_report['macro_f_beta']:.4f}")
print(f"Macro Precision:          {full_report['macro_precision']:.4f}")
print(f"Macro Recall:             {full_report['macro_recall']:.4f}")
print(f"Singleton Accuracy:       {full_report['singleton_accuracy']:.2%}")
print(f"Total Runtime:            {total_elapsed:.1f}s")
print('=' * 75)

out_path = Path('/kaggle/working/experiment_1_results.json')
with open(out_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f'\\nSaved report to {out_path}')
"""

nb['cells'][11]['source'] = exp1_code
with open('kaggle_test/abb-aar-ya-paar.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated kaggle_test/abb-aar-ya-paar.ipynb successfully.')
