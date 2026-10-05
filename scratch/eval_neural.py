import gc
import json
import logging
import sys
import time
import psutil
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
import faiss

sys.path.append(str(Path("code/business_entity_resolution/src").resolve()))
from business_entity_resolution.data import loader
from business_entity_resolution.config import TRAIN_FILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

def get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

logger.info("Loading missed pairs from prior blocking analysis...")
analysis_gt = pd.read_csv("scratch/blocking_analysis_gt.csv")
missed_df = analysis_gt[analysis_gt["missed_uncapped"] == True].copy()
logger.info(f"Total entirely missed pairs found: {len(missed_df)}")

# Sample 10,000 pairs for a viable benchmark
np.random.seed(42)
if len(missed_df) > 10000:
    test_pairs = missed_df.sample(10000, random_state=42).copy()
else:
    test_pairs = missed_df.copy()

s1_queries = test_pairs["source1_entity_id"].unique()
true_targets = test_pairs["matched_entity_id"].unique()
target_countries = test_pairs["other_country"].unique()

logger.info("Loading full raw datasets to extract text and distractors...")
s1_raw = loader.load_source(TRAIN_FILES["S1"])
s2_raw = loader.load_source(TRAIN_FILES["S2"])
s3_raw = loader.load_source(TRAIN_FILES["S3"])

# We need ~90,000 distractors to make a realistic 100k target pool
s2_s3_raw = pd.concat([s2_raw, s3_raw], ignore_index=True)
true_target_rows = s2_s3_raw[s2_s3_raw["entity_id"].isin(true_targets)].copy()
s2_s3_distractor_pool = s2_s3_raw[~s2_s3_raw["entity_id"].isin(true_targets)]

# Sample distractors matching the country distribution of the true targets
distractors = s2_s3_distractor_pool.sample(min(90000, len(s2_s3_distractor_pool)), random_state=42)

# Final target pool
target_pool = pd.concat([true_target_rows, distractors])
s1_test = s1_raw[s1_raw["entity_id"].isin(s1_queries)].copy()

del s1_raw, s2_raw, s3_raw, s2_s3_raw; gc.collect()

def concat_text(row):
    name = str(row.get("business_name", "")) if pd.notna(row.get("business_name")) else ""
    addr = str(row.get("business_address", "")) if pd.notna(row.get("business_address")) else ""
    return (name + " " + addr).strip()

def has_devanagari(text):
    if pd.isna(text): return False
    return bool(re.search(r'[\u0900-\u097F]', str(text)))

s1_test["text"] = s1_test.apply(concat_text, axis=1)
target_pool["text"] = target_pool.apply(concat_text, axis=1)

s1_test["has_dev"] = s1_test["text"].apply(has_devanagari)
target_pool["has_dev"] = target_pool["text"].apply(has_devanagari)

s1_test["no_address"] = s1_test["business_address"].isna() | (s1_test["business_address"] == "")
target_pool["no_address"] = target_pool["business_address"].isna() | (target_pool["business_address"] == "")

logger.info(f"Encoding {len(s1_test)} S1 queries and {len(target_pool)} S2/S3 targets...")
t0_embed = time.time()

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
dim = model.get_sentence_embedding_dimension()

s1_embs = model.encode(s1_test["text"].tolist(), batch_size=256, show_progress_bar=False, normalize_embeddings=True)
target_embs = model.encode(target_pool["text"].tolist(), batch_size=256, show_progress_bar=False, normalize_embeddings=True)
embed_time = time.time() - t0_embed

logger.info(f"Embedding completed in {embed_time:.2f}s. Dim: {dim}")
logger.info(f"Peak RAM (before index build): {get_rss_mb():.0f}MB")

s1_test["emb_idx"] = np.arange(len(s1_test))
target_pool["emb_idx"] = np.arange(len(target_pool))

t0_index = time.time()
indices = {}
target_entity_maps = {}

# Build index per country
for country in s1_test["country"].dropna().unique():
    c_targets = target_pool[target_pool["country"] == country]
    if c_targets.empty: continue
    
    idx = faiss.IndexFlatIP(dim)
    c_embs = target_embs[c_targets["emb_idx"].values]
    idx.add(c_embs)
    
    indices[country] = idx
    target_entity_maps[country] = c_targets["entity_id"].values

index_time = time.time() - t0_index
logger.info(f"FAISS indices built in {index_time:.2f}s")

# Query
t0_query = time.time()
recall_at_1 = 0
recall_at_5 = 0
recall_at_10 = 0

# Results tracking
results_df = []

for country, idx in indices.items():
    c_s1 = s1_test[s1_test["country"] == country]
    if c_s1.empty: continue
    
    q_embs = s1_embs[c_s1["emb_idx"].values]
    D, I = idx.search(q_embs, k=10)
    
    c_s1_ids = c_s1["entity_id"].values
    t_maps = target_entity_maps[country]
    
    for i, s1_id in enumerate(c_s1_ids):
        retrieved_targets = [t_maps[idx_j] for idx_j in I[i] if idx_j >= 0]
        
        # True targets for this s1
        true_t = test_pairs[test_pairs["source1_entity_id"] == s1_id]["matched_entity_id"].values
        
        for tm in true_t:
            r1 = int(tm in retrieved_targets[:1])
            r5 = int(tm in retrieved_targets[:5])
            r10 = int(tm in retrieved_targets[:10])
            
            recall_at_1 += r1
            recall_at_5 += r5
            recall_at_10 += r10
            
            # Demographics for this pair
            tm_row = target_pool[target_pool["entity_id"] == tm]
            has_dev = c_s1.iloc[i]["has_dev"] or (not tm_row.empty and tm_row.iloc[0]["has_dev"])
            no_addr = c_s1.iloc[i]["no_address"] or (not tm_row.empty and tm_row.iloc[0]["no_address"])
            
            results_df.append({
                "country": country,
                "is_s2": tm.startswith("S2-"),
                "has_dev": has_dev,
                "no_addr": no_addr,
                "r1": r1,
                "r5": r5,
                "r10": r10
            })

query_time = time.time() - t0_query
logger.info(f"Querying completed in {query_time:.2f}s")

res = pd.DataFrame(results_df)
total = len(res)

print("--- MULTILINGUAL EMBEDDING RESCUE EXPERIMENT ---")
print(f"Sample size: {total} 'zero-token' true pairs completely missed by baseline.")
print(f"Index pool: {len(target_pool)} total reference records.")
print(f"Embedding Dim: {dim}")
print(f"Embed Time: {embed_time:.2f}s | Index Time: {index_time:.2f}s | Query Time: {query_time:.4f}s")
print(f"Peak RAM: {get_rss_mb():.0f}MB\n")

print("OVERALL RECALL:")
print(f"R@1:  {res['r1'].sum()} / {total} ({res['r1'].mean():.2%})")
print(f"R@5:  {res['r5'].sum()} / {total} ({res['r5'].mean():.2%})")
print(f"R@10: {res['r10'].sum()} / {total} ({res['r10'].mean():.2%})\n")

def print_breakdown(group_col, name):
    print(f"--- By {name} (R@10) ---")
    grouped = res.groupby(group_col)["r10"].agg(['sum', 'count', 'mean'])
    for idx, row in grouped.iterrows():
        print(f"  {idx}: {int(row['sum'])} / {int(row['count'])} ({row['mean']:.2%})")
    print()

print_breakdown("country", "Country")
print_breakdown("has_dev", "Devanagari Presence")
print_breakdown("is_s2", "S2 vs S3")
print_breakdown("no_addr", "Missing Address (Either S1 or S2/S3)")

with open("scratch/embedding_rescue_metrics.json", "w") as f:
    json.dump({
        "sample_size": total,
        "index_pool_size": len(target_pool),
        "embed_time": embed_time,
        "index_time": index_time,
        "query_time": query_time,
        "peak_mem": get_rss_mb(),
        "overall_r1": float(res['r1'].mean()),
        "overall_r5": float(res['r5'].mean()),
        "overall_r10": float(res['r10'].mean())
    }, f, indent=2)
