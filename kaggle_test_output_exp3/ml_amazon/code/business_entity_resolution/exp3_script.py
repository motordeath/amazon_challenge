import json
import time
import pandas as pd
import gc
from pathlib import Path

import sys
sys.path.insert(0, '/kaggle/working/ml_amazon/code/business_entity_resolution/src')

from business_entity_resolution import config
from business_entity_resolution.data import loader
from business_entity_resolution.pipeline.sampling import stratified_sample_s1
from business_entity_resolution.normalization.run_normalize import normalize_source
from business_entity_resolution.pipeline.common import load_normalize_block
from business_entity_resolution.blocking.candidate_generator import union_and_rank, generate_candidates_for_pair
from business_entity_resolution.blocking.recall_eval import blocking_recall
from business_entity_resolution.blocking.vector import vector_candidate_pairs

cfg = config.DEFAULT_CONFIG

def get_rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024

print('=' * 75)
print('STARTING EXPERIMENT 3: VECTOR RETRIEVAL')
print('=' * 75)

t_start = time.time()
report = {'experiment_id': 'EXP-003'}

# Step 1: Loading
t0 = time.time()
print('\n[1] Loading S1 and Ground Truth...')
s1 = loader.load_source(config.TRAIN_FILES['S1'])
gt = loader.load_ground_truth(config.TRAIN_GROUND_TRUTH)

sampled_ids = stratified_sample_s1(s1, gt, 500001, seed=42)
s1 = s1[s1['entity_id'].isin(sampled_ids)].reset_index(drop=True)
gt = gt[gt['source1_entity_id'].isin(sampled_ids)].reset_index(drop=True)

gt_long = loader.explode_ground_truth(gt)
del gt
gc.collect()

# Step 2: Normalization
t0 = time.time()
print('\n[2] Normalizing S1...')
s1n = normalize_source(s1)
del s1
gc.collect()

# Step 3: Existing Lexical Blocking
print('\n[3] Generating Lexical Candidates...')
cand2_lex, s2_subset = load_normalize_block(s1n, config.TRAIN_FILES['S2'], 'S2', cfg.blocking)
cand3_lex, s3_subset = load_normalize_block(s1n, config.TRAIN_FILES['S3'], 'S3', cfg.blocking)

all_lex_uncapped = pd.concat([cand2_lex, cand3_lex], ignore_index=True)
del cand2_lex, cand3_lex
gc.collect()

lex_capped = union_and_rank(all_lex_uncapped, 25)

report['lexical_uncapped'] = blocking_recall(all_lex_uncapped, gt_long)
report['lexical_capped'] = blocking_recall(lex_capped, gt_long)
print(f"Lexical Uncapped Recall: {report['lexical_uncapped']['blocking_recall']:.4f}")
print(f"Lexical Capped Recall:   {report['lexical_capped']['blocking_recall']:.4f}")

# Step 4: Vector Blocking (Name)
print('\n[4] Generating Vector Name Candidates (K=10)...')
vname2 = vector_candidate_pairs(s1n, s2_subset, "name_normalized", "vec_name_S2", top_k=10)
vname3 = vector_candidate_pairs(s1n, s3_subset, "name_normalized", "vec_name_S3", top_k=10)
all_vname = pd.concat([vname2, vname3], ignore_index=True)
del vname2, vname3
gc.collect()

report['vector_name'] = blocking_recall(all_vname, gt_long)
print(f"Vector Name Recall: {report['vector_name']['blocking_recall']:.4f}")

# Step 5: Vector Blocking (Address)
print('\n[5] Generating Vector Address Candidates (K=10)...')
vaddr2 = vector_candidate_pairs(s1n, s2_subset, "address_normalized", "vec_addr_S2", top_k=10)
vaddr3 = vector_candidate_pairs(s1n, s3_subset, "address_normalized", "vec_addr_S3", top_k=10)
all_vaddr = pd.concat([vaddr2, vaddr3], ignore_index=True)
del vaddr2, vaddr3
gc.collect()

report['vector_addr'] = blocking_recall(all_vaddr, gt_long)
print(f"Vector Address Recall: {report['vector_addr']['blocking_recall']:.4f}")

# Union Vector
all_vec = pd.concat([all_vname, all_vaddr], ignore_index=True)
report['vector_only'] = blocking_recall(all_vec, gt_long)
print(f"Vector Only (Name+Addr) Recall: {report['vector_only']['blocking_recall']:.4f}")

# Union All Uncapped
all_union_uncapped = pd.concat([all_lex_uncapped, all_vec], ignore_index=True)
report['union_uncapped'] = blocking_recall(all_union_uncapped, gt_long)
print(f"Union Uncapped Recall: {report['union_uncapped']['blocking_recall']:.4f}")

# Union All Capped
union_capped = union_and_rank(all_union_uncapped, 25)
report['union_capped'] = blocking_recall(union_capped, gt_long)
print(f"Union Capped Recall: {report['union_capped']['blocking_recall']:.4f}")

# Compute exclusivities
def get_retrieved_set(df):
    pos = df.merge(gt_long, on=['source1_entity_id', 'candidate_entity_id'], how='inner')
    return set(zip(pos['source1_entity_id'], pos['candidate_entity_id']))

lex_capped_set = get_retrieved_set(lex_capped)
union_capped_set = get_retrieved_set(union_capped)
new_true_positives = union_capped_set - lex_capped_set

report['new_true_positives'] = len(new_true_positives)
print(f"New True Positives Rescued by Vector in Top-25: {len(new_true_positives)}")

out_path = Path('/kaggle/working/experiment_3_results.json')
with open(out_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f'\nSaved report to {out_path}')


# Downstream
if len(new_true_positives) > 1000:
    print('\n[6] Vector retrieval materially improved recall! Running downstream pipeline...')
    from business_entity_resolution.pipeline.labeling import label_candidates
    from business_entity_resolution.features.build_feature_matrix import build_feature_matrix
    from business_entity_resolution.models.train_lgbm import split_all_s1_ids, train_lgbm
    from business_entity_resolution.models.threshold import tune_global_threshold, predict_matches
    from business_entity_resolution.evaluation.report import full_validation_report

    # Step 4: Labeling
    print('\n[7] Labeling Candidate Pairs...')
    labeled = label_candidates(union_capped, gt_long)
    del union_capped
    gc.collect()

    y = labeled['label'].to_numpy()
    small_labeled = labeled[['source1_entity_id', 'candidate_entity_id', 'score']]
    del labeled
    gc.collect()

    # Step 5: Features
    print('\n[8] Building Feature Matrix...')
    others_n = pd.concat([s2_subset, s3_subset], ignore_index=True)
    del s2_subset, s3_subset
    gc.collect()

    X, meta = build_feature_matrix(small_labeled, s1n, others_n, cfg.features, chunk_size=4_000_000)
    del others_n, small_labeled
    gc.collect()

    # Step 6: LightGBM Training
    print('\n[9] Training LightGBM...')
    groups = meta['source1_entity_id']
    train_ids, val_ids = split_all_s1_ids(s1n['entity_id'], n_splits=5, fold=0, seed=cfg.random_seed)
    train_mask = groups.isin(train_ids).to_numpy()
    val_mask = groups.isin(val_ids).to_numpy()

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    meta_val = meta[val_mask].reset_index(drop=True)
    del X, y
    gc.collect()

    model = train_lgbm(X_train, y_train, X_val, y_val, cfg.model)

    # Step 7: Threshold Tuning & Validation
    print('\n[10] Threshold Tuning & Final Validation...')
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

    print(f"\nDownstream F0.5: {full_report['macro_f_beta']:.4f}")

    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\nSaved downstream report to {out_path}')
