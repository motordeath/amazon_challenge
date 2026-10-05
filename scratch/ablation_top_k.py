import gc
import json
import logging
import sys
import time
import psutil
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path("code/business_entity_resolution/src").resolve()))
from business_entity_resolution.data import loader
from business_entity_resolution.pipeline.sampling import stratified_sample_s1
from business_entity_resolution.normalization.run_normalize import normalize_source
from business_entity_resolution.pipeline.common import NORMALIZED_KEEP_COLS
from business_entity_resolution.blocking.exact import exact_key_blocks
from business_entity_resolution.blocking.token import token_candidate_pairs
from business_entity_resolution.blocking.candidate_generator import union_and_rank
from business_entity_resolution.config import DEFAULT_CONFIG, TRAIN_FILES, TRAIN_GROUND_TRUTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

def get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

cfg = DEFAULT_CONFIG

logger.info("Loading data...")
s1 = loader.load_source(TRAIN_FILES["S1"])
gt = loader.load_ground_truth(TRAIN_GROUND_TRUTH)

sampled_ids = stratified_sample_s1(s1, gt, 500001, seed=42)
s1 = s1[s1["entity_id"].isin(sampled_ids)].reset_index(drop=True)
gt = gt[gt["source1_entity_id"].isin(sampled_ids)].reset_index(drop=True)

gt_long = loader.explode_ground_truth(gt)
true_pairs = set(zip(gt_long["source1_entity_id"], gt_long["matched_entity_id"]))
s1n = normalize_source(s1)
del s1; gc.collect()

s2 = loader.load_source(TRAIN_FILES["S2"])
s3 = loader.load_source(TRAIN_FILES["S3"])
s2n = normalize_source(s2)
s3n = normalize_source(s3)
del s2, s3; gc.collect()

def eval_channel(cand_df):
    if cand_df.empty: return 0
    pairs = set(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_id"]))
    return len(pairs & true_pairs)

logger.info("Running exact name channels...")
exact_s2 = exact_key_blocks(s1n, s2n, "name_compact", "exact_name_S2")
exact_s3 = exact_key_blocks(s1n, s3n, "name_compact", "exact_name_S3")
exact_all = pd.concat([exact_s2, exact_s3])
exact_recall = eval_channel(exact_all)
logger.info(f"Exact name recall: {exact_recall} / {len(true_pairs)}")

results = []

for k in [20, 50, 100, 200]:
    logger.info(f"--- Processing k={k} ---")
    gc.collect()
    start_time = time.time()
    rss_start = get_rss_mb()
    
    # Name channels
    tok_name_s2 = token_candidate_pairs(
        s1n, s2n, "name_normalized", "tok_name_S2",
        min_token_length=cfg.blocking.min_token_length, max_df_fraction=cfg.blocking.max_token_doc_frequency,
        max_df_abs_other=cfg.blocking.token_max_df_abs_other, max_df_abs_s1=cfg.blocking.token_max_df_abs_s1,
        top_k_per_s1=k
    )
    tok_name_s3 = token_candidate_pairs(
        s1n, s3n, "name_normalized", "tok_name_S3",
        min_token_length=cfg.blocking.min_token_length, max_df_fraction=cfg.blocking.max_token_doc_frequency,
        max_df_abs_other=cfg.blocking.token_max_df_abs_other, max_df_abs_s1=cfg.blocking.token_max_df_abs_s1,
        top_k_per_s1=k
    )
    tok_name_all = pd.concat([tok_name_s2, tok_name_s3])
    tok_name_recall = eval_channel(tok_name_all)
    
    # Address channels
    tok_addr_s2 = token_candidate_pairs(
        s1n, s2n, "address_normalized", "tok_address_S2",
        min_token_length=cfg.blocking.min_token_length, max_df_fraction=cfg.blocking.max_token_doc_frequency,
        max_df_abs_other=cfg.blocking.token_max_df_abs_other, max_df_abs_s1=cfg.blocking.token_max_df_abs_s1,
        top_k_per_s1=k
    )
    tok_addr_s3 = token_candidate_pairs(
        s1n, s3n, "address_normalized", "tok_address_S3",
        min_token_length=cfg.blocking.min_token_length, max_df_fraction=cfg.blocking.max_token_doc_frequency,
        max_df_abs_other=cfg.blocking.token_max_df_abs_other, max_df_abs_s1=cfg.blocking.token_max_df_abs_s1,
        top_k_per_s1=k
    )
    tok_addr_all = pd.concat([tok_addr_s2, tok_addr_s3])
    tok_addr_recall = eval_channel(tok_addr_all)
    
    all_channels = pd.concat([exact_all, tok_name_all, tok_addr_all], ignore_index=True)
    uncapped_pairs = set(zip(all_channels["source1_entity_id"], all_channels["candidate_entity_id"]))
    uncapped_true = uncapped_pairs & true_pairs
    uncapped_recall = len(uncapped_true)
    uncapped_count = len(uncapped_pairs)
    
    # Peak memory likely reached here around concatenation
    rss_peak = get_rss_mb()
    
    capped_all = union_and_rank(all_channels, cfg.blocking.max_candidates_per_s1)
    capped_pairs = set(zip(capped_all["source1_entity_id"], capped_all["candidate_entity_id"]))
    capped_true = capped_pairs & true_pairs
    capped_recall = len(capped_true)
    capped_count = len(capped_pairs)
    
    runtime = time.time() - start_time
    
    results.append({
        "top_k": k,
        "tok_name_recall": tok_name_recall,
        "tok_addr_recall": tok_addr_recall,
        "uncapped_recall": uncapped_recall,
        "uncapped_count": uncapped_count,
        "capped_recall": capped_recall,
        "capped_count": capped_count,
        "runtime_sec": runtime,
        "peak_mem_mb": rss_peak
    })
    logger.info(f"k={k} -> Uncapped: {uncapped_recall}, Capped: {capped_recall}, Time: {runtime:.1f}s, Mem: {rss_peak:.0f}MB")

    del tok_name_s2, tok_name_s3, tok_name_all, tok_addr_s2, tok_addr_s3, tok_addr_all, all_channels, capped_all
    gc.collect()

with open("scratch/ablation_results.json", "w") as f:
    json.dump({"exact_recall": exact_recall, "total_true": len(true_pairs), "results": results}, f, indent=2)
logger.info("All done!")
