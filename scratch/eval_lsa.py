import sys
import gc
import time
import psutil
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

sys.path.append(str(Path("code/business_entity_resolution/src").resolve()))
from business_entity_resolution.data import loader
from business_entity_resolution.pipeline.sampling import stratified_sample_s1
from business_entity_resolution.normalization.run_normalize import normalize_source
from business_entity_resolution.config import TRAIN_FILES, TRAIN_GROUND_TRUTH

def get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("Loading data...")
s1 = loader.load_source(TRAIN_FILES["S1"])
gt = loader.load_ground_truth(TRAIN_GROUND_TRUTH)

# Identify a small subset of the 500k sample to test
sampled_ids = stratified_sample_s1(s1, gt, 500001, seed=42)
s1 = s1[s1["entity_id"].isin(sampled_ids)].reset_index(drop=True)
gt = gt[gt["source1_entity_id"].isin(sampled_ids)].reset_index(drop=True)

gt_long = loader.explode_ground_truth(gt)
s1n = normalize_source(s1)
del s1; gc.collect()

s2 = loader.load_source(TRAIN_FILES["S2"])
s3 = loader.load_source(TRAIN_FILES["S3"])
s2n = normalize_source(s2)
s3n = normalize_source(s3)
del s2, s3; gc.collect()

# We only care about pairs that were entirely missed by token channels
# Load the previous analysis CSV
try:
    analysis_gt = pd.read_csv("scratch/blocking_analysis_gt.csv")
    missed_uncapped_pairs = set(zip(analysis_gt[analysis_gt["missed_uncapped"]]["source1_entity_id"], 
                                    analysis_gt[analysis_gt["missed_uncapped"]]["matched_entity_id"]))
except:
    print("Run without the missed pairs filter, just evaluating LSA capability.")
    sys.exit(1)

# Sample 5,000 missed S1s to test
missed_s1_ids = list(set([p[0] for p in missed_uncapped_pairs]))[:5000]
s1_test = s1n[s1n["entity_id"].isin(missed_s1_ids)].copy()

# Get the corresponding targets (we only keep target rows in the true missed pairs for benchmarking retrieval)
valid_targets = set([p[1] for p in missed_uncapped_pairs if p[0] in missed_s1_ids])
other_test = pd.concat([s2n[s2n["entity_id"].isin(valid_targets)], s3n[s3n["entity_id"].isin(valid_targets)]]).copy()
print(f"Testing LSA on {len(s1_test)} S1 records against {len(other_test)} S2/S3 records (representing the exact missed true pairs).")

# We will add 10,000 random targets as noise to make it a real retrieval test
np.random.seed(42)
noise_s2 = s2n.sample(5000)
noise_s3 = s3n.sample(5000)
other_test = pd.concat([other_test, noise_s2, noise_s3]).drop_duplicates(subset=["entity_id"])

corpus = pd.concat([s1_test["name_normalized"], other_test["name_normalized"]]).fillna("")

print("Vectorizing...")
t0 = time.time()
vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer="char_wb", min_df=1, sublinear_tf=True)
tfidf = vectorizer.fit_transform(corpus)
svd = TruncatedSVD(n_components=64, random_state=42)
svd.fit(tfidf)

emb_source = svd.transform(vectorizer.transform(s1_test["name_normalized"].fillna("")))
emb_target = svd.transform(vectorizer.transform(other_test["name_normalized"].fillna("")))

norm_src = np.linalg.norm(emb_source, axis=1, keepdims=True)
norm_tgt = np.linalg.norm(emb_target, axis=1, keepdims=True)
norm_src[norm_src == 0] = 1.0
norm_tgt[norm_tgt == 0] = 1.0
emb_source = emb_source / norm_src
emb_target = emb_target / norm_tgt

sim_matrix = np.dot(emb_source, emb_target.T)
print(f"Embedding generation and dot product took {time.time() - t0:.2f}s")

# Check recall
retrieved = 0
total = 0
for i, s1_id in enumerate(s1_test["entity_id"].values):
    sim_scores = sim_matrix[i]
    top_indices = np.argsort(-sim_scores)[:20] # top 20
    retrieved_targets = set(other_test.iloc[top_indices]["entity_id"].values)
    
    true_missed = [p[1] for p in missed_uncapped_pairs if p[0] == s1_id]
    total += len(true_missed)
    for tm in true_missed:
        if tm in retrieved_targets:
            retrieved += 1

print(f"LSA Character N-gram Recall on missed pairs (Top-20 with small noise pool): {retrieved} / {total} ({(retrieved/max(1, total)):.2%})")
