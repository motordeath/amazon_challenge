import gc
import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# add code to path
sys.path.append(str(Path("code/business_entity_resolution/src").resolve()))
from business_entity_resolution.data import loader
from business_entity_resolution.pipeline.sampling import stratified_sample_s1
from business_entity_resolution.normalization.run_normalize import normalize_source
from business_entity_resolution.pipeline.common import NORMALIZED_KEEP_COLS
from business_entity_resolution.blocking.exact import exact_key_blocks
from business_entity_resolution.blocking.token import token_candidate_pairs
from business_entity_resolution.config import DEFAULT_CONFIG, TRAIN_FILES, TRAIN_GROUND_TRUTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

cfg = DEFAULT_CONFIG

# 1. Load data
logger.info("Loading data...")
s1 = loader.load_source(TRAIN_FILES["S1"])
gt = loader.load_ground_truth(TRAIN_GROUND_TRUTH)

# Sample to match baseline
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

# We need the original country to check country exclusion
s1_country = s1n.set_index("entity_id")["country"].to_dict()
s2_country = s2n.set_index("entity_id")["country"].to_dict()
s3_country = s3n.set_index("entity_id")["country"].to_dict()
s2s3_country = {**s2_country, **s3_country}

# True match demographics
gt_long["s1_country"] = gt_long["source1_entity_id"].map(s1_country)
gt_long["other_country"] = gt_long["matched_entity_id"].map(s2s3_country)
gt_long["same_country"] = gt_long["s1_country"] == gt_long["other_country"]
gt_long["is_s2"] = gt_long["matched_entity_id"].str.startswith("S2-")

# Evaluate blocking channels
def eval_channel(cand_df):
    if cand_df.empty: return 0
    pairs = set(zip(cand_df["source1_entity_id"], cand_df["candidate_entity_id"]))
    return len(pairs & true_pairs)

logger.info("Running exact name channels...")
exact_s2 = exact_key_blocks(s1n, s2n, "name_compact", "exact_name_S2")
exact_s3 = exact_key_blocks(s1n, s3n, "name_compact", "exact_name_S3")
exact_all = pd.concat([exact_s2, exact_s3])
logger.info(f"Exact name recall: {eval_channel(exact_all)} / {len(true_pairs)}")

logger.info("Running token name channels...")
tok_name_s2 = token_candidate_pairs(
    s1n, s2n, "name_normalized", "tok_name_S2",
    min_token_length=cfg.blocking.min_token_length, max_df_fraction=cfg.blocking.max_token_doc_frequency,
    max_df_abs_other=cfg.blocking.token_max_df_abs_other, max_df_abs_s1=cfg.blocking.token_max_df_abs_s1,
    top_k_per_s1=cfg.blocking.token_top_k_per_s1
)
tok_name_s3 = token_candidate_pairs(
    s1n, s3n, "name_normalized", "tok_name_S3",
    min_token_length=cfg.blocking.min_token_length, max_df_fraction=cfg.blocking.max_token_doc_frequency,
    max_df_abs_other=cfg.blocking.token_max_df_abs_other, max_df_abs_s1=cfg.blocking.token_max_df_abs_s1,
    top_k_per_s1=cfg.blocking.token_top_k_per_s1
)
tok_name_all = pd.concat([tok_name_s2, tok_name_s3])
logger.info(f"Token name recall: {eval_channel(tok_name_all)} / {len(true_pairs)}")

logger.info("Running token address channels...")
tok_addr_s2 = token_candidate_pairs(
    s1n, s2n, "address_normalized", "tok_address_S2",
    min_token_length=cfg.blocking.min_token_length, max_df_fraction=cfg.blocking.max_token_doc_frequency,
    max_df_abs_other=cfg.blocking.token_max_df_abs_other, max_df_abs_s1=cfg.blocking.token_max_df_abs_s1,
    top_k_per_s1=cfg.blocking.token_top_k_per_s1
)
tok_addr_s3 = token_candidate_pairs(
    s1n, s3n, "address_normalized", "tok_address_S3",
    min_token_length=cfg.blocking.min_token_length, max_df_fraction=cfg.blocking.max_token_doc_frequency,
    max_df_abs_other=cfg.blocking.token_max_df_abs_other, max_df_abs_s1=cfg.blocking.token_max_df_abs_s1,
    top_k_per_s1=cfg.blocking.token_top_k_per_s1
)
tok_addr_all = pd.concat([tok_addr_s2, tok_addr_s3])
logger.info(f"Token address recall: {eval_channel(tok_addr_all)} / {len(true_pairs)}")

logger.info("Combining all channels (uncapped)...")
all_channels = pd.concat([exact_all, tok_name_all, tok_addr_all], ignore_index=True)
uncapped_pairs = set(zip(all_channels["source1_entity_id"], all_channels["candidate_entity_id"]))
uncapped_true = uncapped_pairs & true_pairs
logger.info(f"Union uncapped recall: {len(uncapped_true)} / {len(true_pairs)}")

logger.info("Applying top-K cap per S1...")
from business_entity_resolution.blocking.candidate_generator import union_and_rank
capped_all = union_and_rank(all_channels, cfg.blocking.max_candidates_per_s1)
capped_pairs = set(zip(capped_all["source1_entity_id"], capped_all["candidate_entity_id"]))
capped_true = capped_pairs & true_pairs
logger.info(f"Capped recall (baseline): {len(capped_true)} / {len(true_pairs)}")

logger.info("Analyzing missed matches...")
missed_uncapped = true_pairs - uncapped_pairs
missed_capped = uncapped_true - capped_pairs
missed_total = true_pairs - capped_pairs

logger.info(f"True matches missed entirely (no channel retrieved): {len(missed_uncapped)}")
logger.info(f"True matches retrieved but lost to candidate cap: {len(missed_capped)}")

# Country exclusion logic
gt_long["missed_uncapped"] = gt_long.apply(lambda r: (r.source1_entity_id, r.matched_entity_id) in missed_uncapped, axis=1)
gt_long["missed_capped"] = gt_long.apply(lambda r: (r.source1_entity_id, r.matched_entity_id) in missed_capped, axis=1)
gt_long["missed_total"] = gt_long.apply(lambda r: (r.source1_entity_id, r.matched_entity_id) in missed_total, axis=1)

missed_due_to_country = gt_long[gt_long["missed_uncapped"] & ~gt_long["same_country"]]
logger.info(f"Missed due to country partitioning (cross-country matches): {len(missed_due_to_country)}")

# S1 cardinality
gt["is_singleton"] = gt["matched_entity_ids"].isna() | (gt["matched_entity_ids"] == "")
s1_card = gt.set_index("source1_entity_id")["matched_entity_ids"].apply(lambda x: len(x.split(",")) if isinstance(x, str) and x else 0).to_dict()
gt_long["s1_matches"] = gt_long["source1_entity_id"].map(s1_card)
gt_long["is_singleton_s1"] = gt_long["s1_matches"] == 0

logger.info("Breakdown by Country:")
for c in gt_long["s1_country"].dropna().unique():
    subset = gt_long[gt_long["s1_country"] == c]
    retrieved = len(subset[~subset["missed_total"]])
    print(f"Country {c}: {retrieved}/{len(subset)} ({retrieved/len(subset):.2%})")

logger.info("Breakdown by S2/S3:")
for is_s2 in [True, False]:
    subset = gt_long[gt_long["is_s2"] == is_s2]
    retrieved = len(subset[~subset["missed_total"]])
    print(f"Source {'S2' if is_s2 else 'S3'}: {retrieved}/{len(subset)} ({retrieved/len(subset):.2%})")

logger.info("Breakdown by S1 match cardinality:")
for card in [1, 2, 3, 4, 5]:
    subset = gt_long[gt_long["s1_matches"] == card]
    retrieved = len(subset[~subset["missed_total"]])
    if len(subset): print(f"S1 Matches={card}: {retrieved}/{len(subset)} ({retrieved/len(subset):.2%})")
subset = gt_long[gt_long["s1_matches"] >= 6]
retrieved = len(subset[~subset["missed_total"]])
if len(subset): print(f"S1 Matches>=6: {retrieved}/{len(subset)} ({retrieved/len(subset):.2%})")

logger.info("Writing missed analysis df to investigate further...")
Path("scratch").mkdir(exist_ok=True)
gt_long.to_csv("scratch/blocking_analysis_gt.csv", index=False)
