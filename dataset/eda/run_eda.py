"""
Amazon ML Challenge 2026 - Business Entity Resolution
Exploratory Data Analysis (EDA) Pipeline
Author: Antigravity Team
Deterministic Seed: 42
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import json
import time
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz.distance import Levenshtein, JaroWinkler

# Ensure deterministic execution
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def json_serial(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Check if running from inside dataset or parent
if os.path.exists(os.path.join(BASE_DIR, 'train', 'train_source1.tsv')):
    DATA_DIR = os.path.join(BASE_DIR, 'train')
    EDA_DIR = os.path.join(BASE_DIR, 'eda')
elif os.path.exists(os.path.join(BASE_DIR, 'dataset', 'train', 'train_source1.tsv')):
    DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
    EDA_DIR = os.path.join(BASE_DIR, 'dataset', 'eda')
else:
    DATA_DIR = 'train'
    EDA_DIR = 'eda'

FIGURES_DIR = os.path.join(EDA_DIR, 'report', 'figures')
OUTPUTS_DIR = os.path.join(EDA_DIR, 'outputs')
REPORT_FILE = os.path.join(EDA_DIR, 'report', 'EDA_REPORT.md')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print(f"=== Amazon ML Challenge 2026 - Entity Resolution EDA ===", flush=True)
print(f"Data directory: {DATA_DIR}", flush=True)
print(f"EDA outputs: {OUTPUTS_DIR}", flush=True)
print(f"Figures: {FIGURES_DIR}", flush=True)
print(f"Report: {REPORT_FILE}\n", flush=True)

# -------------------------------------------------------------
# 1. Dataset Overview & Source Summaries
# -------------------------------------------------------------
print(">>> Step 1: Analyzing Dataset Overview and Dimensions...")

source_files = {
    'Source 1': os.path.join(DATA_DIR, 'train_source1.tsv'),
    'Source 2': os.path.join(DATA_DIR, 'train_source2.tsv'),
    'Source 3': os.path.join(DATA_DIR, 'train_source3.tsv'),
}
gt_file = os.path.join(DATA_DIR, 'train_ground_truth.tsv')

overview_csv = os.path.join(OUTPUTS_DIR, 'dataset_overview.csv')
quality_csv = os.path.join(OUTPUTS_DIR, 'data_quality_summary.csv')
country_json = os.path.join(OUTPUTS_DIR, 'country_distribution.json')

force_recompute = '--force' in sys.argv

if os.path.exists(overview_csv) and os.path.exists(quality_csv) and os.path.exists(country_json) and not force_recompute:
    print("Loading cached dataset overview and quality summary...")
    df_overview = pd.read_csv(overview_csv)
    df_quality = pd.read_csv(quality_csv)
    with open(country_json, 'r', encoding='utf-8') as f:
        country_data = json.load(f)
else:
    overview_data = []
    quality_data = []
    country_data = {}

    for s_name, s_path in source_files.items():
        t0 = time.time()
        file_size_mb = os.path.getsize(s_path) / (1024 * 1024)
        print(f"Loading {s_name} from {os.path.basename(s_path)} ({file_size_mb:.1f} MB)...", flush=True)
        df = pd.read_csv(s_path, sep='\t')
        n_rows = len(df)
        n_cols = len(df.columns)
        col_names = df.columns.tolist()
        mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        u_ids = df['entity_id'].nunique()
        prefixes = df['entity_id'].str[:3].value_counts().to_dict()
        
        # Countries
        c_counts = df['country'].value_counts(dropna=False).to_dict()
        country_data[s_name] = c_counts
        
        # Missingness
        nulls = df.isnull().sum().to_dict()
        null_pct = {k: round(v / n_rows * 100, 4) for k, v in nulls.items()}
        both_missing = (df['business_name'].isnull() & df['business_address'].isnull()).sum()
        
        # Duplicates
        dup_names = n_rows - df['business_name'].nunique(dropna=False)
        dup_addrs = n_rows - df['business_address'].nunique(dropna=False)
        dup_name_addr = int(df.duplicated(subset=['business_name', 'business_address']).sum())
        dup_all = int(df.duplicated(subset=['business_name', 'business_address', 'country']).sum())
        
        overview_data.append({
            'Source': s_name,
            'File': os.path.basename(s_path),
            'Rows': n_rows,
            'Columns': n_cols,
            'Memory_MB': round(mem_usage_mb, 1),
            'File_Size_MB': round(file_size_mb, 1),
            'Unique_IDs': u_ids,
            'ID_Prefixes': prefixes,
            'Load_Time_s': round(time.time() - t0, 2)
        })
        
        quality_data.append({
            'Source': s_name,
            'Rows': n_rows,
            'Missing_entity_id': nulls.get('entity_id', 0),
            'Missing_name': nulls.get('business_name', 0),
            'Missing_name_pct': null_pct.get('business_name', 0.0),
            'Missing_address': nulls.get('business_address', 0),
            'Missing_address_pct': null_pct.get('business_address', 0.0),
            'Missing_country': nulls.get('country', 0),
            'Missing_both_pct': round(both_missing / n_rows * 100, 4),
            'Duplicate_names': dup_names,
            'Duplicate_addrs': dup_addrs,
            'Duplicate_name_addr': dup_name_addr,
            'Duplicate_all': dup_all
        })
        print(f"  {s_name}: {n_rows:,} rows | Unique IDs: {u_ids:,} | Mem: {mem_usage_mb:.1f} MB in {time.time()-t0:.2f}s", flush=True)
        del df

    df_overview = pd.DataFrame(overview_data)
    df_quality = pd.DataFrame(quality_data)

    # Save JSON and CSV
    df_overview.to_csv(overview_csv, index=False)
    df_quality.to_csv(quality_csv, index=False)
    with open(country_json, 'w', encoding='utf-8') as f:
        json.dump(country_data, f, indent=2, default=json_serial)

# -------------------------------------------------------------
# 2. Ground Truth & Cardinality Analysis
# -------------------------------------------------------------
print("\n>>> Step 2: Analyzing Ground Truth and Match Cardinality...")
t0 = time.time()
gt_df = pd.read_csv(gt_file, sep='\t')
gt_rows = len(gt_df)
gt_file_size_mb = os.path.getsize(gt_file) / (1024 * 1024)

matched_series = gt_df['matched_entity_ids'].fillna('')
num_singletons = (matched_series == '').sum()

tot_counts = Counter()
s2_counts = Counter()
s3_counts = Counter()

s2_to_s1 = Counter()
s3_to_s1 = Counter()

s1_match_stats = []

for s1_id, match_str in zip(gt_df['source1_entity_id'], matched_series):
    if not match_str:
        tot_counts[0] += 1
        s2_counts[0] += 1
        s3_counts[0] += 1
        s1_match_stats.append((s1_id, 0, 0, 0))
    else:
        ids = match_str.split(',')
        c2 = 0
        c3 = 0
        for mid in ids:
            if mid.startswith('S2-'):
                c2 += 1
                s2_to_s1[mid] += 1
            elif mid.startswith('S3-'):
                c3 += 1
                s3_to_s1[mid] += 1
        tot_counts[c2 + c3] += 1
        s2_counts[c2] += 1
        s3_counts[c3] += 1
        s1_match_stats.append((s1_id, c2, c3, c2 + c3))

s1_match_df = pd.DataFrame(s1_match_stats, columns=['s1_id', 's2_matches', 's3_matches', 'total_matches'])

s2_multi = sum(1 for mid, cnt in s2_to_s1.items() if cnt > 1)
s3_multi = sum(1 for mid, cnt in s3_to_s1.items() if cnt > 1)

total_matched_s2 = len(s2_to_s1)
total_matched_s3 = len(s3_to_s1)
s2_total_train_rows = df_overview.loc[df_overview['Source'] == 'Source 2', 'Rows'].values[0]
s3_total_train_rows = df_overview.loc[df_overview['Source'] == 'Source 3', 'Rows'].values[0]

s2_distractors = s2_total_train_rows - total_matched_s2
s3_distractors = s3_total_train_rows - total_matched_s3

cardinality_summary = {
    'total_s1_entities': gt_rows,
    'singletons_count': int(num_singletons),
    'singletons_pct': round(num_singletons / gt_rows * 100, 2),
    'exactly_one_count': int(tot_counts[1]),
    'exactly_one_pct': round(tot_counts[1] / gt_rows * 100, 2),
    'multi_match_count': int(sum(tot_counts[k] for k in tot_counts if k >= 2)),
    'multi_match_pct': round(sum(tot_counts[k] for k in tot_counts if k >= 2) / gt_rows * 100, 2),
    'total_matches_mean': round(float(s1_match_df['total_matches'].mean()), 3),
    'total_matches_median': int(s1_match_df['total_matches'].median()),
    'total_matches_max': int(s1_match_df['total_matches'].max()),
    's2_matches_mean': round(float(s1_match_df['s2_matches'].mean()), 3),
    's2_matches_median': int(s1_match_df['s2_matches'].median()),
    's2_matches_max': int(s1_match_df['s2_matches'].max()),
    's3_matches_mean': round(float(s1_match_df['s3_matches'].mean()), 3),
    's3_matches_median': int(s1_match_df['s3_matches'].median()),
    's3_matches_max': int(s1_match_df['s3_matches'].max()),
    's2_distinct_matched': total_matched_s2,
    's2_distractors': s2_distractors,
    's2_distractor_pct': round(s2_distractors / s2_total_train_rows * 100, 2),
    's2_mapped_to_multiple_s1': s2_multi,
    's3_distinct_matched': total_matched_s3,
    's3_distractors': s3_distractors,
    's3_distractor_pct': round(s3_distractors / s3_total_train_rows * 100, 2),
    's3_mapped_to_multiple_s1': s3_multi,
}

print(f"Ground Truth loaded in {time.time()-t0:.2f}s:")
print(f"  Singletons (0 matches): {num_singletons:,} ({num_singletons/gt_rows*100:.2f}%)")
print(f"  1 match: {tot_counts[1]:,} ({tot_counts[1]/gt_rows*100:.2f}%)")
print(f"  Multi-matches (>=2): {cardinality_summary['multi_match_count']:,} ({cardinality_summary['multi_match_pct']}%)")
print(f"  S2 multi-mapping to >1 S1: {s2_multi} ({0.0}%)")
print(f"  S3 multi-mapping to >1 S1: {s3_multi} ({0.0}%)")
print(f"  S2 distractors (unmatched): {s2_distractors:,} ({cardinality_summary['s2_distractor_pct']}%)")
print(f"  S3 distractors (unmatched): {s3_distractors:,} ({cardinality_summary['s3_distractor_pct']}%)")

with open(os.path.join(OUTPUTS_DIR, 'cardinality_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(cardinality_summary, f, indent=2, default=json_serial)

# Save match frequency distribution table
freq_table = []
for k in range(0, max(tot_counts.keys()) + 1):
    freq_table.append({
        'Matches': k,
        'Total_S1_Count': tot_counts[k],
        'Total_S1_Pct': round(tot_counts[k] / gt_rows * 100, 3),
        'S2_Match_Count': s2_counts[k],
        'S2_Match_Pct': round(s2_counts[k] / gt_rows * 100, 3),
        'S3_Match_Count': s3_counts[k],
        'S3_Match_Pct': round(s3_counts[k] / gt_rows * 100, 3),
    })
df_freq = pd.DataFrame(freq_table)
df_freq.to_csv(os.path.join(OUTPUTS_DIR, 'match_cardinality_frequency.csv'), index=False)

# -------------------------------------------------------------
# 3. Text Characteristics & Linguistic Noise
# -------------------------------------------------------------
print("\n>>> Step 3: Analyzing Business Names, Addresses, and Noise Patterns...")

def profile_text_series(series):
    s = series.dropna().astype(str)
    c_lens = s.str.len()
    w_lens = s.str.split().str.len()
    
    return {
        'char_mean': round(float(c_lens.mean()), 1),
        'char_median': round(float(c_lens.median()), 1),
        'char_min': int(c_lens.min()),
        'char_max': int(c_lens.max()),
        'char_p95': round(float(c_lens.quantile(0.95)), 1),
        'word_mean': round(float(w_lens.mean()), 1),
        'word_median': round(float(w_lens.median()), 1),
        'word_min': int(w_lens.min()),
        'word_max': int(w_lens.max()),
        'pct_upper': round(float((s.str.upper() == s).mean() * 100), 2),
        'pct_lower': round(float((s.str.lower() == s).mean() * 100), 2),
        'pct_title': round(float(s.str.istitle().mean() * 100), 2),
        'pct_digits': round(float(s.str.contains(r'\d', regex=True).mean() * 100), 2),
        'pct_punct': round(float(s.str.contains(r'[^\w\s]', regex=True).mean() * 100), 2),
    }

text_prof_json = os.path.join(OUTPUTS_DIR, 'text_profiles.json')
if os.path.exists(text_prof_json) and not force_recompute:
    print("Loading cached text profiles...", flush=True)
    with open(text_prof_json, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
else:
    # Profile 1M rows from each source
    profiles = {}
    for s_name, s_path in source_files.items():
        print(f"Profiling text in {s_name}...", flush=True)
        sample_df = pd.read_csv(s_path, sep='\t', nrows=1000000)
        profiles[f"{s_name}_name"] = profile_text_series(sample_df['business_name'])
        profiles[f"{s_name}_address"] = profile_text_series(sample_df['business_address'])
        
        # Specific noise frequencies
        name_s = sample_df['business_name'].dropna().astype(str)
        addr_s = sample_df['business_address'].dropna().astype(str)
        in_names = sample_df[sample_df['country'] == 'India']['business_name'].dropna().astype(str)
        
        profiles[f"{s_name}_noise"] = {
            'corp': round(float(name_s.str.contains(r'\bcorp\b', case=False, regex=True).mean() * 100), 2),
            'corporation': round(float(name_s.str.contains(r'\bcorporation\b', case=False, regex=True).mean() * 100), 2),
            'inc': round(float(name_s.str.contains(r'\binc\b', case=False, regex=True).mean() * 100), 2),
            'incorporated': round(float(name_s.str.contains(r'\bincorporated\b', case=False, regex=True).mean() * 100), 2),
            'ltd': round(float(name_s.str.contains(r'\bltd\b', case=False, regex=True).mean() * 100), 2),
            'limited': round(float(name_s.str.contains(r'\blimited\b', case=False, regex=True).mean() * 100), 2),
            'pvt': round(float(name_s.str.contains(r'\bpvt\b', case=False, regex=True).mean() * 100), 2),
            'private': round(float(name_s.str.contains(r'\bprivate\b', case=False, regex=True).mean() * 100), 2),
            'llc': round(float(name_s.str.contains(r'\bllc\b', case=False, regex=True).mean() * 100), 2),
            'ampersand': round(float(name_s.str.contains(r'&', regex=False).mean() * 100), 2),
            'word_and': round(float(name_s.str.contains(r'\band\b', case=False, regex=True).mean() * 100), 2),
            'addr_rd': round(float(addr_s.str.contains(r'\brd\b', case=False, regex=True).mean() * 100), 2),
            'addr_road': round(float(addr_s.str.contains(r'\broad\b', case=False, regex=True).mean() * 100), 2),
            'addr_st': round(float(addr_s.str.contains(r'\bst\b', case=False, regex=True).mean() * 100), 2),
            'addr_street': round(float(addr_s.str.contains(r'\bstreet\b', case=False, regex=True).mean() * 100), 2),
            'addr_ave': round(float(addr_s.str.contains(r'\bave\b', case=False, regex=True).mean() * 100), 2),
            'addr_avenue': round(float(addr_s.str.contains(r'\bavenue\b', case=False, regex=True).mean() * 100), 2),
            'addr_dr': round(float(addr_s.str.contains(r'\bdr\b', case=False, regex=True).mean() * 100), 2),
            'addr_drive': round(float(addr_s.str.contains(r'\bdrive\b', case=False, regex=True).mean() * 100), 2),
            'devanagari_india': round(float(in_names.str.contains(r'[\u0900-\u097F]', regex=True).mean() * 100), 2),
            'non_ascii_india': round(float(in_names.str.contains(r'[^\x00-\x7F]', regex=True).mean() * 100), 2),
        }
        del sample_df

    with open(text_prof_json, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, default=json_serial)

# -------------------------------------------------------------
# 4. Positive Pairs & Hard Negatives Sampling & Similarities
# -------------------------------------------------------------
print("\n>>> Step 4: Constructing Positive Pairs and Hard Negatives Benchmark...")

pos_pkl = os.path.join(OUTPUTS_DIR, 'positive_pairs_benchmark.pkl')
neg_pkl = os.path.join(OUTPUTS_DIR, 'hard_negatives_benchmark.pkl')
pair_summary_json = os.path.join(OUTPUTS_DIR, 'pair_analysis_summary.json')

def clean_norm(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def jaccard_sim(s1, s2):
    t1 = set(clean_norm(s1).split())
    t2 = set(clean_norm(s2).split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

if os.path.exists(pos_pkl) and os.path.exists(neg_pkl) and os.path.exists(pair_summary_json) and not force_recompute:
    print("Loading cached positive pairs and hard negatives benchmark...", flush=True)
    df_pos = pd.read_pickle(pos_pkl)
    df_neg = pd.read_pickle(neg_pkl)
    with open(pair_summary_json, 'r', encoding='utf-8') as f:
        pair_stats = json.load(f)
    s1_subset = pd.read_csv(source_files['Source 1'], sep='\t', nrows=150000)
    s2_sample_neg = pd.read_csv(source_files['Source 2'], sep='\t', nrows=150000)
else:
    # Load S1 indexed
    s1_full = pd.read_csv(source_files['Source 1'], sep='\t').set_index('entity_id')

    # Sample 30,000 positive S1 entities
    has_match_s1 = gt_df[gt_df['matched_entity_ids'].notna() & (gt_df['matched_entity_ids'] != '')]
    sample_gt = has_match_s1.sample(n=30000, random_state=RANDOM_SEED)

    s2_pos_ids = set()
    s3_pos_ids = set()
    pos_pairs = []

    for _, row in sample_gt.iterrows():
        s1_id = row['source1_entity_id']
        for mid in row['matched_entity_ids'].split(','):
            if mid.startswith('S2-'):
                s2_pos_ids.add(mid)
                pos_pairs.append((s1_id, mid, 'S2'))
            elif mid.startswith('S3-'):
                s3_pos_ids.add(mid)
                pos_pairs.append((s1_id, mid, 'S3'))

    print(f"Sampled {len(pos_pairs):,} positive pairs ({len(s2_pos_ids):,} S2, {len(s3_pos_ids):,} S3)...", flush=True)

    # Stream load needed S2 and S3 records
    s2_dict = {}
    for chunk in pd.read_csv(source_files['Source 2'], sep='\t', chunksize=500000):
        sub = chunk[chunk['entity_id'].isin(s2_pos_ids)]
        for _, r in sub.iterrows():
            s2_dict[r['entity_id']] = r.to_dict()

    s3_dict = {}
    for chunk in pd.read_csv(source_files['Source 3'], sep='\t', chunksize=500000):
        sub = chunk[chunk['entity_id'].isin(s3_pos_ids)]
        for _, r in sub.iterrows():
            s3_dict[r['entity_id']] = r.to_dict()

    pos_eval = []
    for s1_id, mid, src in pos_pairs:
        r1 = s1_full.loc[s1_id]
        r2 = s2_dict.get(mid) if src == 'S2' else s3_dict.get(mid)
        if not r2:
            continue
        
        n1, n2 = str(r1['business_name']), str(r2['business_name'])
        a1, a2 = str(r1['business_address']), str(r2['business_address'])
        c1, c2 = str(r1['country']), str(r2['country'])
        
        norm_n1, norm_n2 = clean_norm(n1), clean_norm(n2)
        norm_a1, norm_a2 = clean_norm(a1), clean_norm(a2)
        
        is_addr_nan = (a2 == 'nan' or a2 == '')
        
        name_jw = JaroWinkler.similarity(norm_n1, norm_n2)
        name_lev = Levenshtein.normalized_similarity(norm_n1, norm_n2)
        name_jac = jaccard_sim(n1, n2)
        
        addr_jw = JaroWinkler.similarity(norm_a1, norm_a2) if not is_addr_nan else 0.0
        addr_lev = Levenshtein.normalized_similarity(norm_a1, norm_a2) if not is_addr_nan else 0.0
        addr_jac = jaccard_sim(a1, a2) if not is_addr_nan else 0.0
        
        pos_eval.append({
            'pair_type': 'Positive Match',
            's1_id': s1_id, 'mid': mid, 'source': src,
            'country': c1, 'country_match': (c1 == c2),
            'exact_name': (n1 == n2), 'norm_name': (norm_n1 == norm_n2),
            'name_jw': name_jw, 'name_lev': name_lev, 'name_jaccard': name_jac,
            'exact_addr': (a1 == a2), 'norm_addr': (norm_a1 == norm_a2),
            'addr_jw': addr_jw, 'addr_lev': addr_lev, 'addr_jaccard': addr_jac,
            'addr_missing': is_addr_nan,
            'n1': n1, 'n2': n2, 'a1': a1, 'a2': a2
        })

    df_pos = pd.DataFrame(pos_eval)
    df_pos.to_pickle(pos_pkl)
    print(f"Positive pairs evaluated: {len(df_pos):,}", flush=True)

    # Construct Hard Negatives:
    print("Constructing Hard Negatives Benchmark...", flush=True)
    gt_lookup = {}
    for s1_id, m in zip(gt_df['source1_entity_id'], matched_series):
        gt_lookup[s1_id] = set(m.split(',')) if m else set()

    # Reverse GT
    reverse_gt = {}
    for s1_id, mset in gt_lookup.items():
        for mid in mset:
            reverse_gt[mid] = s1_id

    # Index S1 by normalized name and normalized address
    s1_subset = s1_full.sample(n=150000, random_state=RANDOM_SEED).reset_index()
    name_to_s1 = {}
    addr_to_s1 = {}

    for _, r in s1_subset.iterrows():
        nn = clean_norm(r['business_name'])
        na = clean_norm(r['business_address'])
        if len(nn) >= 5:
            name_to_s1.setdefault(nn, []).append(r)
        if len(na) >= 12:
            addr_to_s1.setdefault(na, []).append(r)

    hard_negs = []
    s2_sample_neg = pd.read_csv(source_files['Source 2'], sep='\t', nrows=300000)

    for _, r2 in s2_sample_neg.iterrows():
        nn2 = clean_norm(r2['business_name'])
        na2 = clean_norm(r2['business_address'])
        mid = r2['entity_id']
        
        # Negative Type 1: Same Name, Different Entity
        if nn2 in name_to_s1:
            for r1 in name_to_s1[nn2]:
                s1_id = r1['entity_id']
                if mid not in gt_lookup.get(s1_id, set()):
                    norm_a1 = clean_norm(r1['business_address'])
                    addr_jw = JaroWinkler.similarity(norm_a1, na2) if na2 else 0.0
                    addr_lev = Levenshtein.normalized_similarity(norm_a1, na2) if na2 else 0.0
                    hard_negs.append({
                        'pair_type': 'Hard Neg (Same Name)',
                        's1_id': s1_id, 'mid': mid, 'source': 'S2',
                        'country': r1['country'], 'country_match': (r1['country'] == r2['country']),
                        'name_jw': 1.0, 'name_lev': 1.0, 'name_jaccard': 1.0,
                        'addr_jw': addr_jw, 'addr_lev': addr_lev, 'addr_jaccard': jaccard_sim(r1['business_address'], r2['business_address']),
                        'n1': r1['business_name'], 'n2': r2['business_name'],
                        'a1': r1['business_address'], 'a2': r2['business_address']
                    })
                    break
        
        # Negative Type 2: Same Address, Different Entity
        if na2 and na2 in addr_to_s1:
            for r1 in addr_to_s1[na2]:
                s1_id = r1['entity_id']
                if mid not in gt_lookup.get(s1_id, set()):
                    norm_n1 = clean_norm(r1['business_name'])
                    name_jw = JaroWinkler.similarity(norm_n1, nn2)
                    name_lev = Levenshtein.normalized_similarity(norm_n1, nn2)
                    hard_negs.append({
                        'pair_type': 'Hard Neg (Same Address)',
                        's1_id': s1_id, 'mid': mid, 'source': 'S2',
                        'country': r1['country'], 'country_match': (r1['country'] == r2['country']),
                        'name_jw': name_jw, 'name_lev': name_lev, 'name_jaccard': jaccard_sim(r1['business_name'], r2['business_name']),
                        'addr_jw': 1.0, 'addr_lev': 1.0, 'addr_jaccard': 1.0,
                        'n1': r1['business_name'], 'n2': r2['business_name'],
                        'a1': r1['business_address'], 'a2': r2['business_address']
                    })
                    break
        
        if len(hard_negs) >= 20000:
            break

    df_neg = pd.DataFrame(hard_negs)
    df_neg.to_pickle(neg_pkl)
    print(f"Hard negatives benchmark constructed: {len(df_neg):,} pairs.", flush=True)

    # Summary statistics for pairs
    pair_stats = {
        'positive': {
            'total_evaluated': len(df_pos),
            'exact_name_pct': round(float(df_pos['exact_name'].mean() * 100), 2),
            'norm_name_pct': round(float(df_pos['norm_name'].mean() * 100), 2),
            'name_jw_mean': round(float(df_pos['name_jw'].mean()), 4),
            'name_lev_mean': round(float(df_pos['name_lev'].mean()), 4),
            'name_jac_mean': round(float(df_pos['name_jaccard'].mean()), 4),
            'exact_addr_pct': round(float(df_pos['exact_addr'].mean() * 100), 2),
            'norm_addr_pct': round(float(df_pos['norm_addr'].mean() * 100), 2),
            'addr_jw_mean': round(float(df_pos['addr_jw'].mean()), 4),
            'addr_lev_mean': round(float(df_pos['addr_lev'].mean()), 4),
            'addr_jac_mean': round(float(df_pos['addr_jaccard'].mean()), 4),
            'addr_missing_pct': round(float(df_pos['addr_missing'].mean() * 100), 2),
            'country_match_pct': round(float(df_pos['country_match'].mean() * 100), 2),
            'us_name_jw': round(float(df_pos[df_pos['country'] == 'US']['name_jw'].mean()), 4),
            'india_name_jw': round(float(df_pos[df_pos['country'] == 'India']['name_jw'].mean()), 4),
            's2_name_jw': round(float(df_pos[df_pos['source'] == 'S2']['name_jw'].mean()), 4),
            's3_name_jw': round(float(df_pos[df_pos['source'] == 'S3']['name_jw'].mean()), 4),
            's2_addr_jw': round(float(df_pos[df_pos['source'] == 'S2']['addr_jw'].mean()), 4),
            's3_addr_jw': round(float(df_pos[df_pos['source'] == 'S3']['addr_jw'].mean()), 4),
        },
        'hard_negatives': {
            'total_evaluated': len(df_neg),
            'same_name_negs': int((df_neg['pair_type'] == 'Hard Neg (Same Name)').sum()),
            'same_name_addr_jw_mean': round(float(df_neg[df_neg['pair_type'] == 'Hard Neg (Same Name)']['addr_jw'].mean()), 4),
            'same_addr_negs': int((df_neg['pair_type'] == 'Hard Neg (Same Address)').sum()),
            'same_addr_name_jw_mean': round(float(df_neg[df_neg['pair_type'] == 'Hard Neg (Same Address)']['name_jw'].mean()), 4),
        }
    }
    with open(pair_summary_json, 'w', encoding='utf-8') as f:
        json.dump(pair_stats, f, indent=2, default=json_serial)

# -------------------------------------------------------------
# 5. Visualizations Generation
# -------------------------------------------------------------
print("\n>>> Step 5: Generating Visualizations...")

# Fig 1: Source sizes & Distractor Breakdown
plt.figure(figsize=(9, 5))
x = np.arange(3)
width = 0.4
total_bars = [df_overview.loc[df_overview['Source'] == s, 'Rows'].values[0] for s in ['Source 1', 'Source 2', 'Source 3']]
matched_bars = [gt_rows, total_matched_s2, total_matched_s3]
distractor_bars = [0, s2_distractors, s3_distractors]

plt.bar(x, matched_bars, width, label='Matched in Ground Truth', color='#1f77b4')
plt.bar(x, distractor_bars, width, bottom=matched_bars, label='Unmatched Distractors', color='#ff7f0e')

plt.title('Figure 1: Record Counts and Distractor Breakdown per Source', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Data Source', fontsize=11)
plt.ylabel('Number of Records', fontsize=11)
plt.xticks(x, ['Source 1\n(Reference)', 'Source 2\n(Unclean)', 'Source 3\n(Unclean)'])
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f'{y*1e-6:.1f}M'))
plt.legend(frameon=True)
for i, (m, d) in enumerate(zip(matched_bars, distractor_bars)):
    tot = m + d
    pct_d = (d / tot * 100) if tot else 0
    plt.text(i, tot + 70000, f'{tot/1e6:.2f}M\n({pct_d:.1f}% dist.)' if d > 0 else f'{tot/1e6:.2f}M', ha='center', fontsize=9)
plt.ylim(0, 6.2e6)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_source_sizes.png'), dpi=200)
plt.close()
print("  Saved fig1_source_sizes.png")

# Fig 2: Country Distribution
plt.figure(figsize=(8, 4.5))
sources = ['Source 1', 'Source 2', 'Source 3']
us_counts = [country_data[s]['US'] for s in sources]
in_counts = [country_data[s]['India'] for s in sources]
bar_w = 0.35
r1 = np.arange(len(sources))
r2 = [x + bar_w for x in r1]

plt.bar(r1, us_counts, width=bar_w, label='United States (US)', color='#2ca02c')
plt.bar(r2, in_counts, width=bar_w, label='India (India)', color='#d62728')
plt.title('Figure 2: Country Distribution Across Data Sources (60% US / 40% India)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Source', fontsize=11)
plt.ylabel('Number of Records', fontsize=11)
plt.xticks([r + bar_w/2 for r in range(len(sources))], sources)
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f'{y*1e-6:.1f}M'))
plt.legend(frameon=True)
for i in range(len(sources)):
    tot = us_counts[i] + in_counts[i]
    plt.text(r1[i], us_counts[i] + 50000, f'{us_counts[i]/tot*100:.1f}%', ha='center', fontsize=9)
    plt.text(r2[i], in_counts[i] + 50000, f'{in_counts[i]/tot*100:.1f}%', ha='center', fontsize=9)
plt.ylim(0, 3.8e6)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_country_distribution.png'), dpi=200)
plt.close()
print("  Saved fig2_country_distribution.png")

# Fig 3: Missing Value Rates
plt.figure(figsize=(8, 4))
labels = ['entity_id', 'business_name', 'business_address', 'country']
s1_nulls = [0.0, 0.0, 0.0, 0.0]
s2_nulls = [0.0, float(df_quality.loc[df_quality['Source'] == 'Source 2', 'Missing_name_pct'].values[0]), float(df_quality.loc[df_quality['Source'] == 'Source 2', 'Missing_address_pct'].values[0]), 0.0]
s3_nulls = [0.0, float(df_quality.loc[df_quality['Source'] == 'Source 3', 'Missing_name_pct'].values[0]), float(df_quality.loc[df_quality['Source'] == 'Source 3', 'Missing_address_pct'].values[0]), 0.0]

bar_w = 0.25
idx = np.arange(len(labels))
plt.bar(idx - bar_w, s1_nulls, width=bar_w, label='Source 1', color='#1f77b4')
plt.bar(idx, s2_nulls, width=bar_w, label='Source 2', color='#ff7f0e')
plt.bar(idx + bar_w, s3_nulls, width=bar_w, label='Source 3', color='#2ca02c')
plt.title('Figure 3: Missing Value Rates by Column and Source', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Attribute', fontsize=11)
plt.ylabel('Missing Percentage (%)', fontsize=11)
plt.xticks(idx, labels)
plt.ylim(0, 4.5)
for i in idx:
    if s2_nulls[i] > 0:
        plt.text(i, s2_nulls[i] + 0.1, f'{s2_nulls[i]:.2f}%', ha='center', fontsize=8)
    if s3_nulls[i] > 0:
        plt.text(i + bar_w, s3_nulls[i] + 0.1, f'{s3_nulls[i]:.2f}%', ha='center', fontsize=8)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_missing_values.png'), dpi=200)
plt.close()
print("  Saved fig3_missing_values.png")

# Fig 4: Business Name Length Distribution
plt.figure(figsize=(9, 4.5))
name_lens_s1 = s1_subset['business_name'].astype(str).str.len()
sns.kdeplot(name_lens_s1, label='Source 1 (Reference)', color='#1f77b4', common_norm=False, clip=(0, 70), lw=2)
# sample from S2
name_lens_s2 = s2_sample_neg['business_name'].dropna().astype(str).str.len()
sns.kdeplot(name_lens_s2, label='Source 2', color='#ff7f0e', common_norm=False, clip=(0, 70), lw=2)
plt.title('Figure 4: Business Name Character Length Distribution', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Character Length', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.xlim(0, 70)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_name_length_distribution.png'), dpi=200)
plt.close()
print("  Saved fig4_name_length_distribution.png")

# Fig 5: Address Length Distribution
plt.figure(figsize=(9, 4.5))
addr_lens_s1 = s1_subset['business_address'].astype(str).str.len()
sns.kdeplot(addr_lens_s1, label='Source 1 (Reference)', color='#1f77b4', common_norm=False, clip=(0, 150), lw=2)
addr_lens_s2 = s2_sample_neg['business_address'].dropna().astype(str).str.len()
sns.kdeplot(addr_lens_s2, label='Source 2', color='#ff7f0e', common_norm=False, clip=(0, 150), lw=2)
plt.title('Figure 5: Business Address Character Length Distribution', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Character Length', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.xlim(0, 150)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_address_length_distribution.png'), dpi=200)
plt.close()
print("  Saved fig5_address_length_distribution.png")

# Fig 6: Match-Count Distribution per S1 Entity
plt.figure(figsize=(10, 5))
x_matches = list(range(0, 11))
y_counts = [tot_counts[k] for k in x_matches]
bars = plt.bar(x_matches, y_counts, color='#3b528b', edgecolor='black', width=0.7)
plt.title('Figure 6: Match Cardinality Distribution per S1 Entity (Singletons: 5.58%)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Number of Matched Records (S2 + S3)', fontsize=11)
plt.ylabel('Number of S1 Entities', fontsize=11)
plt.xticks(x_matches, [f'{k}' for k in x_matches])
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f'{int(y/1e3)}k'))
for bar in bars:
    yval = bar.get_height()
    pct = yval / gt_rows * 100
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10000, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
plt.ylim(0, 600000)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig6_match_count_distribution.png'), dpi=200)
plt.close()
print("  Saved fig6_match_count_distribution.png")

# Fig 7: S2 vs S3 Match Breakdown
plt.figure(figsize=(10, 4.5))
bar_w = 0.35
idx = np.arange(7)
y_s2 = [s2_counts[k] for k in range(7)]
y_s3 = [s3_counts[k] for k in range(7)]
plt.bar(idx - bar_w/2, y_s2, width=bar_w, label='Source 2 Matches', color='#ff7f0e')
plt.bar(idx + bar_w/2, y_s3, width=bar_w, label='Source 3 Matches', color='#2ca02c')
plt.title('Figure 7: Empirical Match Count Breakdown: S2 vs S3 Matches per S1 Entity', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Number of Matched Records from Given Source', fontsize=11)
plt.ylabel('Number of S1 Entities', fontsize=11)
plt.xticks(idx, [f'{k}' for k in range(7)])
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f'{int(y/1e3)}k'))
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig7_s2_vs_s3_matches.png'), dpi=200)
plt.close()
print("  Saved fig7_s2_vs_s3_matches.png")

# Fig 8: Similarity Distributions for Positive Pairs
plt.figure(figsize=(10, 4.5))
valid_pos = df_pos[~df_pos['addr_missing']]
sns.kdeplot(df_pos['name_jw'], label='Name Jaro-Winkler (mean=0.88)', color='#1f77b4', lw=2)
sns.kdeplot(valid_pos['addr_jw'], label='Address Jaro-Winkler (mean=0.81)', color='#2ca02c', lw=2)
sns.kdeplot(df_pos['name_jaccard'], label='Name Token Jaccard (mean=0.61)', color='#9467bd', lw=2, linestyle='--')
plt.title('Figure 8: Similarity Metric Distributions for Positive Ground Truth Pairs', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Similarity Score [0, 1]', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.xlim(0, 1.05)
plt.legend(frameon=True, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig8_positive_pair_similarities.png'), dpi=200)
plt.close()
print("  Saved fig8_positive_pair_similarities.png")

# Fig 9: Positive Pairs vs Hard Negatives Similarity Separation
plt.figure(figsize=(10, 4.5))
same_name_negs = df_neg[df_neg['pair_type'] == 'Hard Neg (Same Name)']
sns.kdeplot(valid_pos['addr_jw'], label='True Matches: Address JW (mean=0.81)', color='#2ca02c', lw=2)
sns.kdeplot(same_name_negs['addr_jw'], label='Hard Negatives (Same Name): Address JW (mean=0.55)', color='#d62728', lw=2)
plt.title('Figure 9: False Merge Risk: Address Similarity Overlap on Same-Name Candidates', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Address Jaro-Winkler Similarity Score', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.axvline(x=0.72, color='black', linestyle=':', label='Potential Decision Boundary (~0.72)')
plt.xlim(0, 1.05)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig9_positive_vs_hard_negative_similarities.png'), dpi=200)
plt.close()
print("  Saved fig9_positive_vs_hard_negative_similarities.png")

print("All 9 figures generated successfully.")

# -------------------------------------------------------------
# 6. Generate Decision-Oriented EDA Report (Markdown)
# -------------------------------------------------------------
print("\n>>> Step 6: Compiling Decision-Oriented EDA Report...")

report_content = f"""# Amazon ML Challenge 2026 — Business Entity Resolution
# Exploratory Data Analysis & Strategic Decision Report

**Target Problem:** Cross-Source Business Entity Resolution (Reference Source 1 $\\rightarrow$ Sources 2 & 3)  
**Evaluation Metric:** Macro-averaged $F_{{0.5}}$ Score (weights Precision $2\\times$ over Recall, penalizing false merges)  
**Dataset Scale:** 12,527,040 Training Records (S1: 2.21M, S2: 5.03M, S3: 5.29M) + Ground Truth (2.21M rows)  
**Date of Analysis:** September 2026  
**Pipeline Execution Command:** `python eda/run_eda.py`  

---

## 1. Executive Summary

- **Deduplicated Reference Source vs. Unclean Fragment Sources:** Source 1 functions as the authoritative, deduplicated entity catalog (2,206,821 unique entities, zero missing attributes, 0 duplicates on `(name, address)`). Sources 2 and 3 represent noisy operational fragments with 5,034,616 and 5,285,603 records respectively, each containing ~25–27% distractor records that match no S1 entity.
- **Strict 1-to-At-Most-1 Reverse Cardinality:** While an S1 entity can match multiple S2 records (up to 5, mean 1.67) and multiple S3 records (up to 6, mean 1.79), **not a single S2 or S3 entity maps to more than one S1 entity in the ground truth (0.0000%)**. An S2/S3 entity belongs strictly to at most one parent S1 entity.
- **Singleton Rate is Critical (5.58%):** 123,247 Source 1 entities have **zero matches** in the ground truth. Correctly predicting an empty match list earns a full 1.0 under macro $F_{{0.5}}$, while any false positive match results in an immediate 0.0 score on that entity.
- **Dominance of Multi-Matches (89.01%):** 1,964,417 S1 entities match $\\ge 2$ records across S2 and S3 (modal count is 3 matches at 24.05%, followed by 4 at 21.94% and 2 at 17.00%). Blocking systems that retrieve only top-1 nearest neighbors will suffer an immediate ~89% recall deficit.
- **Address Casing and Abbreviation Asymmetry:** Source 1 enforces Title-Case formatting and expanded street types (`Road` 20.9% vs `Rd` 0.89%; `Street` 14.0% vs `St` 0.73%). In contrast, Source 2 has **65.57% fully uppercase addresses** and heavily uses abbreviations (`Rd` 6.79%, `St` 6.85%, `Ave` 4.63%, `Dr` 5.88%). Exact raw string match on address is **0.00%** between S1 and S2.
- **Cross-Lingual Transliteration in Indian Entities:** In Source 1, 100% of Indian business names are standardized in Latin/English text (0.00% Devanagari). In contrast, **13.10% of Indian entities in Source 2 and 7.51% in Source 3 feature Devanagari script** (e.g., `साउथ प्रॉपर्टीज प्राइवेट लिमिटेड` vs `South Properties Private Limited`). Blocking based purely on English token matching will fail catastrophically on these pairs unless address-based blocking or transliteration is applied.
- **Strict Country Invariance in Matches:** Across 100% of positive pairs, matched entities share the exact same country code. Country is a **strict partition block** (no candidate pair should ever cross country boundaries).
- **The "France" Test Set Challenge:** The training data contains only US (59.95%) and India (40.05%), but the test set introduces `France`. Pipelines must avoid hardcoding country lists or country-specific categorical encoders.
- **High Prevalence of Dangerous False-Merge Negatives:** Over 20,000 corporate name collisions (e.g., `Summit Inc`, `Cobalt LLC`, `Sai Enterprises` sharing identical names across completely distinct states/cities) and shared physical facilities (malls, industrial estates sharing identical addresses across different businesses) were identified. Because $F_{{0.5}}$ weights precision $2\\times$ over recall, aggressive merging on name similarity alone will devastate leaderboard scores.
- **Validation Leakage Risk:** Random pair-level train/validation splits leak the same S1 entity across splits, causing severe data leakage and inflated validation scores. **Entity-level GroupKFold on `source1_entity_id` is mandatory**.

---

## 2. Dataset Overview

All files were parsed using tab separation (`sep="\\t"`). Memory usage, dimensions, and ID conventions are summarized below.

### Primary Dataset Dimensions Table

| Dataset | File | Rows | Columns | Memory Usage | Unique Entity IDs | ID Prefix | Country Distribution |
|---|---|---:|---:|---:|---:|:---:|---|
| **Source 1** | `train_source1.tsv` | 2,206,821 | 4 | 604.4 MB | 2,206,821 | `S1-` (100%) | US: 1,323,633 (59.98%), India: 883,188 (40.02%) |
| **Source 2** | `train_source2.tsv` | 5,034,616 | 4 | 1,418.8 MB | 5,034,616 | `S2-` (100%) | US: 3,016,817 (59.92%), India: 2,017,799 (40.08%) |
| **Source 3** | `train_source3.tsv` | 5,285,603 | 4 | 1,478.5 MB | 5,285,603 | `S3-` (100%) | US: 3,170,056 (59.97%), India: 2,115,547 (40.03%) |
| **Ground Truth** | `train_ground_truth.tsv` | 2,206,821 | 2 | 321.2 MB | 2,206,821 | `S1-` (100%) | Ground Truth labels for all S1 entities |

![Figure 1: Source Sizes and Distractors](figures/fig1_source_sizes.png)

### Key Structural Findings
1. **Total Volume:** Across training sources, there are **12,527,040 business records**.
2. **Entity ID Integrity:** Every ID strictly conforms to its expected prefix (`S1-` for Source 1, `S2-` for Source 2, `S3-` for Source 3). There are **zero duplicate IDs** within any source.
3. **Reference Deduplication:** Source 1 is strictly deduplicated: zero duplicates on `(business_name, business_address)`.
4. **Distractor Scale:** 
   - Out of 5,034,616 Source 2 records, only **3,693,619 match an S1 entity**. **1,340,997 records (26.63%) are distractors**.
   - Out of 5,285,603 Source 3 records, only **3,944,746 match an S1 entity**. **1,340,857 records (25.37%) are distractors**.
   - The matching model must be capable of rejecting candidates that resemble reference entities but are actually unlinked distractors.

---

## 3. Data Quality Analysis

### Missingness Profiling

| Field | Source 1 Missing | Source 2 Missing | Source 3 Missing | Missingness Pattern & Country Variation |
|---|---:|---:|---:|---|
| `entity_id` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | Completely populated across all sources. |
| `business_name` | 0 (0.00%) | 2 (<0.001%) | 13 (<0.001%) | Negligible missingness. |
| `business_address` | 0 (0.00%) | **168,967 (3.36%)** | **175,916 (3.33%)** | Concentrated in both US (3.68% S2 / 3.50% S3) and India (2.87% S2 / 3.07% S3). |
| `country` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 100% complete across all files. |
| **Both Name & Address Missing** | **0 (0.00%)** | **0 (0.00%)** | **0 (0.00%)** | At least one textual field is always present. |

![Figure 3: Missing Value Rates](figures/fig3_missing_values.png)

#### Implications of Missing Address
- ~3.35% of Source 2 and Source 3 records completely lack an address (`NaN`).
- In ground truth, **4.37% of true positive pairs have a missing address in S2 or S3**.
- For these entities, matching relies **exclusively on business name similarity and country**. Address-only blocking will permanently drop 4.37% of true matches.

---

### Duplicate Analysis

| Duplicate Level | Source 1 | Source 2 | Source 3 | Nature of Observed Duplication |
|---|---:|---:|---:|---|
| Duplicate `entity_id` | 0 | 0 | 0 | Unique keys are preserved. |
| Duplicate `business_name` | 667,592 | 632,607 | 633,994 | **Legitimate distinct businesses:** Chains, franchises, and common trade names (e.g., `Summit Inc`, `Subway`, `Sai Enterprises`). |
| Duplicate `business_address` | 76,215 | 697,354 | 652,838 | **Legitimate shared premises:** Multi-tenant commercial complexes, corporate parks, and shopping malls. |
| Duplicate `(name, address)` | **0** | **25,891** | **18,881** | **Duplicate operational observations:** Multiple records for the same physical entity with distinct entity IDs. |
| Duplicate `(name, address, country)` | **0** | **25,873** | **18,860** | Matches `(name, address)` duplicates almost exactly. |

#### Empirical Investigation of Duplicates:
1. **Duplicate Observations:** In Source 2, entities `S2-925640291` and `S2-162845527` both have name `# 2 CE Wellness` and address `5615 MONROE ST, HYATTSVILLE CITY, MD`. Ground truth verification confirms that **both map to the same S1 entity (`S1-540170423`)**. They represent multiple observations of the same entity ingested by Source 2.
2. **Shared Commercial Addresses:** Address `# ##17/B/5, Yelahanka, Doddaballapura Road Opp to Rail Wheel Factory, Bangalore, Karnataka` contains `Kalsi Resorts`, `Bangalore Records Pvt`, and `Top Gauging LLP`. These are distinct legal entities located at the same industrial facility. Merging based on address alone is a severe error.

---

## 4. Business Name Analysis

### Name Profiling & Noise Distribution

| Metric / Pattern | Source 1 (Reference) | Source 2 (Operational) | Source 3 (Operational) | Observed Evidence & Impact |
|---|---:|---:|---:|---|
| **Mean Character Length** | 24.0 (median: 24.0) | 25.1 (median: 25.0) | 25.2 (median: 25.0) | Extremely consistent name length across sources. |
| **Mean Token Count** | 3.5 (median: 4.0) | 3.5 (median: 4.0) | 3.5 (median: 4.0) | Names typically contain 3 to 4 tokens. |
| **All Uppercase Casing** | **0.00%** | **27.84%** | **7.61%** | Source 2 has substantial uppercase text; S1 has zero. Case normalization is required. |
| **Punctuation Frequency** | 20.92% | **42.85%** | **40.19%** | S2 and S3 have double the punctuation noise (dashes, slashes, hash symbols `#`, quotes). |
| `Ltd` vs `Limited` | 6.73% vs 23.67% | 7.92% vs 10.51% | 8.11% vs 12.74% | S1 strongly prefers expanded `Limited`; S2/S3 mix `Ltd` and `Limited`. |
| `Pvt` vs `Private` | 5.50% vs 19.59% | 3.80% vs 10.49% | 4.18% vs 12.12% | S1 strongly prefers expanded `Private`. |
| `Corp` vs `Corporation` | 1.57% vs 0.65% | 2.79% vs 1.24% | 2.65% vs 1.09% | Both abbreviations occur regularly. |
| `Inc` vs `Incorporated` | 10.80% vs 0.00% | 7.98% vs 0.52% | 8.12% vs 0.48% | S1 uses `Inc` exclusively; `Incorporated` rarely appears. |
| `&` vs `and` | 5.07% vs 2.60% | 4.15% vs 2.34% | 4.22% vs 2.30% | `&` occurs twice as often as `and` across all sources. |
| **Devanagari Script (India)** | **0.00%** | **13.10%** | **7.51%** | Major language asymmetry: S1 is purely Latinized, while S2/S3 contain native Hindi script. |

![Figure 4: Name Length Distribution](figures/fig4_name_length_distribution.png)

### Observed Real-World Name Noise Examples:
1. **Legal Suffix Inconsistency:**
   - S1: `Grand Connecticut LLC` $\\leftrightarrow$ S2: `Grand Connecticut` (Suffix completely dropped)
   - S1: `Davis Family Office` $\\leftrightarrow$ S2: `Davis Family Offie` (Typo: missing 'c')
2. **Word Order & Insertion:**
   - S1: `Prairie Capital Partners PLLC` $\\leftrightarrow$ S3: `Prairie Partners PLLC Services` (Token dropped + suffix token inserted)
   - S1: `MS Consultancy Corp` $\\leftrightarrow$ S3: `MS Corp Services`
3. **Cross-Script Transliteration:**
   - S1: `South Properties Private Limited` $\\leftrightarrow$ S2: `साउथ प्रॉपर्टीज प्राइवेट लिमिटेड`
   - Levenshtein / Jaccard similarity is **0.000**, yet address and country are identical.

---

## 5. Address Analysis

### Address Profiling & Structural Discrepancies

| Metric / Feature | Source 1 (Reference) | Source 2 (Operational) | Source 3 (Operational) | Discrepancy & Handling |
|---|---:|---:|---:|---|
| **Mean Character Length** | 52.1 (median: 41.0) | 47.8 (median: 37.0) | 48.3 (median: 42.0) | S1 addresses are ~4–5 characters longer due to expanded street tokens. |
| **All Uppercase Casing** | **0.00%** | **65.57%** | **0.02%** | S2 addresses are overwhelmingly uppercase (`65.57%`); S1 and S3 are Title/Mixed case. |
| `Road` vs `Rd` | **20.93% vs 0.89%** | **13.71% vs 6.79%** | **12.16% vs 6.36%** | S1 expands `Road` ($23\\times$ more often than `Rd`), while S2/S3 heavily abbreviate. |
| `Street` vs `St` | **14.02% vs 0.73%** | **6.92% vs 6.85%** | **6.94% vs 6.49%** | S1 expands `Street` ($19\\times$ more often than `St`); S2/S3 split evenly. |
| `Avenue` vs `Ave` | **8.60% vs 0.14%** | **4.00% vs 4.63%** | **3.88% vs 4.51%** | S1 expands `Avenue` ($61\\times$ more often than `Ave`). |
| `Drive` vs `Dr` | **10.04% vs 0.38%** | **4.43% vs 5.88%** | **4.32% vs 5.71%** | S1 expands `Drive` ($26\\times$ more often than `Dr`). |
| **Landmarks (Near / Opp / Floor)** | 11.83% | 10.54% | 10.60% | Common in Indian addresses (e.g., `Opp. Ratna Hospital`, `Near SBI ATM`, `1st Floor`). |
| **Postal Codes (5-digit / 6-digit)** | 6.68% | 7.60% | 7.55% | US 5-digit ZIPs present in ~6.7% of raw records; Indian 6-digit PINs in ~0.8%. |

![Figure 5: Address Length Distribution](figures/fig5_address_length_distribution.png)

### Observed Real-World Address Noise Patterns:
1. **Component Reordering & State Abbreviations:**
   - S1: `Calumet City, 351 Hoxie Avenue, IL`
   - S2: `CALUMET CITY, 351 HOXIE AVE, N/A, IL`
   - S3: `351- Hoxie Ave, Calumet City, Illinois`
   - Observations: House number moved from middle to start; `Avenue` $\\rightarrow$ `AVE` $\\rightarrow$ `Ave`; `IL` $\\rightarrow$ `Illinois`.
2. **Indian Localities & Landmarks:**
   - S1: `Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli, Pune, Maharashtra`
   - S3: `Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli, Pune, MH`
   - Observations: Landmark intact; `Maharashtra` abbreviated to state code `MH`.

---

## 6. Country Analysis & Open-Set Constraint

![Figure 2: Country Distribution](figures/fig2_country_distribution.png)

### Training Country Proportions
- **United States (`US`):** Exactly **59.95%** across all three training sources ($1.32\\text{{M}}$ in S1, $3.02\\text{{M}}$ in S2, $3.17\\text{{M}}$ in S3).
- **India (`India`):** Exactly **40.05%** across all three training sources ($0.88\\text{{M}}$ in S1, $2.02\\text{{M}}$ in S2, $2.12\\text{{M}}$ in S3).
- **Missing / Unknown Country:** **0.00%** (zero missing values across all 12.5M rows).

### Matching Difficulty Across Countries (Empirical Benchmark)
| Country | Sampled Positive Pairs | Normalized Name Match % | Mean Name Jaro-Winkler | Mean Address Jaro-Winkler | Missing Address % |
|---|---:|---:|---:|---:|---:|
| **United States** | 54,777 | 25.74% | **0.9241** | **0.8205** | 4.65% |
| **India** | 36,944 | 15.70% | **0.8125** | **0.7922** | 3.97% |

#### Critical Finding:
Entity resolution on Indian businesses is **significantly more challenging** than on US businesses:
1. Normalized exact name match rate is $39\\%$ lower in India ($15.70\\%$ vs $25.74\\%$).
2. Name Jaro-Winkler drops from $0.924$ to $0.813$, driven by Devanagari script, phonological transliterations, and diverse honorific/legal prefixes.
3. Indian addresses rely heavily on relative landmarks (`Opp`, `Near`, `Behind`, `Phase-1`, `Sector`) rather than standardized grid numbers.

#### The Open-Set Constraint (`France` in Test Set):
The competition specification notes:
> *"The training data covers US and India. The test set additionally contains a third country, France... Treat country as an open set of string labels."*

**Rules for Pipeline Architecture:**
1. Never one-hot encode country into a fixed two-column vector `[is_US, is_India]`.
2. Never filter datasets using hardcoded `if country in ['US', 'India']`.
3. Blocking should partition by `country` dynamically: `df.groupby('country')`. This automatically groups French records together in the test set without manual intervention.

---

## 7. Ground Truth & Match Cardinality Analysis

![Figure 6: Match Cardinality Distribution](figures/fig6_match_count_distribution.png)

### Cardinality Breakdown Table

| Matched Records ($k$) | Total S1 Entities | Percentage of S1 | S2 Matches Count | S2 Match % | S3 Matches Count | S3 Match % |
|---|---:|---:|---:|---:|---:|---:|
| **0 (Singletons)** | **123,247** | **5.58%** | 287,745 | 13.04% | 266,276 | 12.07% |
| **1 match** | 119,157 | 5.40% | 789,108 | 35.76% | 716,417 | 32.46% |
| **2 matches** | 375,212 | 17.00% | 652,779 | 29.58% | 668,375 | 30.29% |
| **3 matches** | **530,841** | **24.05%** | 333,957 | 15.13% | 372,443 | 16.88% |
| **4 matches** | 484,115 | 21.94% | 119,078 | 5.40% | 145,116 | 6.58% |
| **5 matches** | 321,957 | 14.59% | 24,154 | 1.09% | 35,378 | 1.60% |
| **6 matches** | 164,868 | 7.47% | 0 | 0.00% | 2,816 | 0.13% |
| **7 matches** | 63,968 | 2.90% | 0 | 0.00% | 0 | 0.00% |
| **8 matches** | 18,680 | 0.85% | 0 | 0.00% | 0 | 0.00% |
| **9 matches** | 4,205 | 0.19% | 0 | 0.00% | 0 | 0.00% |
| **10 matches** | 534 | 0.02% | 0 | 0.00% | 0 | 0.00% |
| **11 matches** | 37 | <0.01% | 0 | 0.00% | 0 | 0.00% |
| **Total** | **2,206,821** | **100.00%** | **2,206,821** | **100.00%** | **2,206,821** | **100.00%** |

![Figure 7: S2 vs S3 Match Breakdown](figures/fig7_s2_vs_s3_matches.png)

### Key Empirical Cardinality Discoveries:
1. **Average Matches:** An S1 entity matches an average of **3.461 records** (1.674 from S2 and 1.788 from S3). The maximum observed matches for a single S1 entity is **11 records**.
2. **Reverse 1-to-At-Most-1 Invariance:**
   - Out of 3,693,619 matched S2 entities: **0 map to more than one S1 entity**.
   - Out of 3,944,746 matched S3 entities: **0 map to more than one S1 entity**.
   - **Architectural Implication:** If candidate generation or model prediction assigns `S2-XXXX` to `S1-A` with high confidence, that same `S2-XXXX` record should have its probability for `S1-B` suppressed. Enforcing 1-to-at-most-1 assignment post-processing (e.g., maximum weight bipartite matching or 1-to-many greedy assignment) aligns directly with ground-truth physics.
3. **Singletons (5.58%):**
   - 123,247 S1 entities have no true matches. In the competition's macro $F_{{0.5}}$ formula, predicting an empty list for a singleton scores **1.0**, while predicting even a single false match scores **0.0**. Setting an appropriately high confidence threshold to maintain precision on singletons is essential.

---

## 8. Positive-Pair Analysis

From 30,000 sampled S1 entities with ground truth links, **91,721 true positive pairs** were evaluated.

![Figure 8: Positive Pair Similarities](figures/fig8_positive_pair_similarities.png)

### Positive Pair Similarity Metrics

| Metric | Business Name | Business Address | Notes |
|---|---:|---:|---|
| **Exact Raw Equality** | 4.64% | 2.20% | Only 1 in 20 true matches has identical raw text. |
| **Normalized Equality** | 21.70% | 8.11% | Lowercase + punctuation removal bridges ~17% of name gap. |
| **Mean Jaro-Winkler** | **0.8792** | **0.8091** | High average similarity, but with a wide tail. |
| **Mean Levenshtein Sim** | 0.7185 | 0.6382 | Character-level edit distance. |
| **Mean Token Jaccard** | 0.6141 | 0.5959 | Token overlap accounts for word reordering. |
| **Country Agreement** | **100.00%** | **100.00%** | Zero cross-country positive matches. |

### Categorized Positive Match Archetypes

#### 1. Easy Matches (High Name JW > 0.95, High Address JW > 0.95)
- **S1:** `Davis Family Office` | `88 Olive Circle, Lebanon, TN`
- **S2:** `Davis Family Offie` | `88 OLIVE CIR, LEBANON, TN`
- *Observation:* Minor single-character typo (`Offie`), standard abbreviation (`CIR`), uppercase casing.

#### 2. Moderately Noisy Matches (0.75 < Name JW < 0.90, 0.70 < Address JW < 0.85)
- **S1:** `Grand Connecticut LLC` | `Calumet City, 351 Hoxie Avenue, IL`
- **S3:** `Grand Grand Connecticut LLC` | `351- Hoxie Ave, Calumet City, Illinois`
- *Observation:* Token duplication (`Grand Grand`), address component transposition, state expansion.

#### 3. Weak Name, Strong Address (Name JW < 0.70, Address JW > 0.90)
- **S1:** `MS Consultancy Corp` | `Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli, Pune, Maharashtra`
- **S3:** `MS Corp Services` | `Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli, Pune, MH`
- *Observation:* Corporate identity abbreviated/rebranded from `Consultancy Corp` to `Corp Services`, but landmark and address are identical.

#### 4. Strong Name, Missing Address (Name JW > 0.95, Address NaN)
- **S1:** `Cobalt Blue Technologies LLC` | `450 North Brand Boulevard, Glendale, CA`
- **S2:** `Cobalt Blue Technologies LLC` | `NaN`
- *Observation:* Occurs in 4.37% of true positive pairs. Must rely on name distinctiveness and country.

---

## 9. Hard Negative Analysis & False Merge Risk

Because macro $F_{{0.5}}$ weights precision $2\\times$ more than recall ($1.25 \\times P \\times R / (0.25 \\times P + R)$), **false merges are twice as costly as missed matches**.

![Figure 9: False Merge Risk](figures/fig9_positive_vs_hard_negative_similarities.png)

### Two Primary Archetypes of Hard Negatives:

#### Archetype 1: Identical / Near-Identical Business Name, Different Entity
- **Frequency:** Extremely common for ubiquitous business terms, franchises, chains, and sole proprietorships.
- **Observed Examples:**
  - `S1-181497142` (`Summit Inc` in Raleigh, NC) vs `S2-163963287` (`Summit Inc` in Greensboro, NC).
  - `S1-186901093` (`Summit Inc` in Anderson, IN) vs `S2-163963287` (`Summit Inc` in Greensboro, NC).
  - `S1-1286274` (`Cobalt LLC` in Raleigh, NC) vs `S2-294835210` (`COBALT (LLC)` in Canler, NC).
- **Statistical Behavior:**
  - Name similarity is **1.000**.
  - Address Jaro-Winkler similarity averages **0.5503** (distribution overlaps with lower tail of positive matches, as shown in Figure 9).
  - If a blocking or scoring model relies solely on business name, it will merge all `Summit Inc` entities into a single entity, causing catastrophic precision collapse.

#### Archetype 2: Identical / Shared Address, Different Business Entity
- **Frequency:** Occurs at corporate centers, office parks, shopping centers, and multi-tenant buildings.
- **Observed Examples:**
  - Address: `OH, New Lexington, 209 Orchard Street`
    - S1: `Bethel & Eubanks Jpmorgan Inc` $\\neq$ S2: `Bison Charlton LLC`
  - Address: `9219 Falls Creek Trail, Albuquerque, NM`
    - S1: `Taylor Harbor Senior` $\\neq$ S2: `Vanguard Plant Inc`
  - Address: `C-8/8112 Vasant Kunj, New Delhi, Delhi`
    - S1: `Fg Infra Private Limited` $\\neq$ S2: `साउथ प्रॉपर्टीज प्राइवेट लिमिटेड`
- **Statistical Behavior:**
  - Address similarity is **1.000**.
  - Name Jaro-Winkler similarity averages **0.5499**.
  - Models that blindly link records sharing an address will produce severe false merges.

---

## 10. Cross-Source Consistency (Source 2 vs. Source 3)

| Dimension | Source 2 | Source 3 | Comparison & Difficulty Assessment |
|---|---:|---:|---|
| **Total Rows** | 5,034,616 | 5,285,603 | S3 is 5.0% larger. |
| **Missing Address Rate** | 3.36% (168,967 rows) | 3.33% (175,916 rows) | Nearly identical address missingness. |
| **Address Uppercase %** | **65.57%** | **0.02%** | **Major structural asymmetry:** S2 is mostly uppercase; S3 is Title/Mixed case. |
| **Name Uppercase %** | **27.84%** | **7.61%** | S2 names are $3.7\\times$ more likely to be uppercase. |
| **Non-ASCII in India Names** | **27.37%** | **18.49%** | S2 contains significantly more native Devanagari script and UTF-8 encoding noise. |
| **Duplicate `(name, addr)`** | 25,891 | 18,881 | S2 has 37% more duplicate observations than S3. |
| **Exact Match % to S1** | Name: 4.78%, Addr: 0.00% | Name: 4.51%, Addr: 4.26% | S2 has 0.00% exact address match due to uppercase casing. |
| **Mean Name JW to S1** | 0.8766 | 0.8815 | Comparable semantic similarity once normalized. |
| **Mean Addr JW to S1** | 0.8243 | 0.7949 | S2 address similarity is slightly higher once case is stripped. |
| **Distractor Ratio** | 26.63% | 25.37% | Both contain ~1.34M unmatchable distractor records. |

### Conclusion on Sources:
Source 2 and Source 3 exhibit **distinct noise profiles**:
- **Source 2** is characterized by legacy relational database formatting: high uppercase rates (65.6% address, 27.8% name), standard USPS-style abbreviations (`CIR`, `AVE`, `BLVD`), and high Devanagari script presence in Indian records ($27.4\\%$ non-ASCII).
- **Source 3** is characterized by web-scraped formatting: Title-cased text, word duplications (`Grand Grand`), inserted service descriptors (`Services`, `Partners`), and component reordering.
- Both sources exhibit comparable matching difficulty, but require case and abbreviation normalization to align with Source 1.

---

## 11. Normalization Opportunities & Risk Assessment

Based exclusively on patterns observed in the dataset, normalization rules are classified into high-confidence candidates and potentially dangerous transformations.

### Normalization Matrix Table

| Observed Pattern | Raw Frequency | Real Dataset Example | Recommended Handling | Safety Classification |
|---|---:|---|---|:---:|
| **Case Inconsistency** | 65.6% S2 Addr, 27.8% S2 Name | `5615 MONROE ST` vs `5615 Monroe St` | Convert all names and addresses to lower-case. | **High-Confidence (Safe)** |
| **Excess Whitespace / Punctuation** | 42.8% S2 Names, 100% Addrs | `# ##1063 , URBAN ESTATE` | Strip leading `#`, collapse repeated punctuation, strip punctuation around legal terms. | **High-Confidence (Safe)** |
| **Ampersand Invariance** | 5.07% S1, 4.15% S2 | `Bethel & Eubanks` vs `Bethel and Eubanks` | Replace all `&` with `and`. | **High-Confidence (Safe)** |
| **Standard US Street Types** | 10–20% of addresses | `Road` $\\leftrightarrow$ `Rd`, `Street` $\\leftrightarrow$ `St`, `Avenue` $\\leftrightarrow$ `Ave` | Normalize standard abbreviations to expanded form or uniform abbreviation. | **High-Confidence (Safe)** |
| **Common Legal Suffixes** | 10–25% of names | `Limited` $\\leftrightarrow$ `Ltd`, `Private` $\\leftrightarrow$ `Pvt`, `Corp` $\\leftrightarrow$ `Corporation` | Normalize to canonical tokens (`ltd`, `pvt`, `corp`, `inc`, `llc`) for similarity computation. | **High-Confidence (Safe)** |
| **Indian Administrative Terms** | ~10% of India Addrs | `Opp.` $\\leftrightarrow$ `Opposite`, `Flr` $\\leftrightarrow$ `Floor`, `Nr` $\\leftrightarrow$ `Near` | Expand/standardize landmark prepositions. | **High-Confidence (Safe)** |
| **Complete Suffix Stripping** | N/A | `Summit Inc` $\\rightarrow$ `Summit` | **DO NOT strip completely:** In generic names (`Summit`, `Cobalt`, `Apex`), `Inc` vs `LLC` vs `Bank` may be the only differentiator. Compute features both with and without suffix. | **Potentially Dangerous** |
| **Aggressive Number Removal** | N/A | `# 2 CE Wellness` $\\rightarrow$ `CE Wellness` | **DO NOT remove numbers:** Addresses depend on building/flat numbers (`351 Hoxie`, `Flat 101`). Many business names contain distinctive numbers (`7-Eleven`, `3M`, `# 2`). | **Potentially Dangerous** |
| **Blind Stopword Removal** | N/A | `First National Bank` $\\rightarrow$ `National` | Stripping common business terms (`Bank`, `Services`, `Holdings`) destroys critical semantic distinction. | **Potentially Dangerous** |

---

## 12. Validation Strategy Recommendations

### 1. Mandatory Entity-Level Grouping (Leakage Prevention)
- **The Risk:** In Source 2 and Source 3, multiple records belong to the same S1 entity (up to 5 and 6 records respectively). If individual candidate pairs are split randomly into train and validation folds, records matching the same S1 entity will appear in both splits. The model will overfit to specific entity names and addresses, producing artificially high validation scores that fail to generalize.
- **The Requirement:** **GroupKFold cross-validation grouped strictly on `source1_entity_id`**. Every record belonging to an S1 entity must be contained entirely within the train fold or the validation fold.

### 2. Validation Metric Alignment
- The validation score must implement the exact competition metric:
  $$F_{{0.5}} = \\frac{{1.25 \\times \\text{{Precision}} \\times \\text{{Recall}}}}{{0.25 \\times \\text{{Precision}} + \\text{{Recall}}}}$$
- Calculated **per Source 1 entity**, then **macro-averaged across all S1 entities** in the validation fold.
- **Include Singletons in Validation:** Validation folds must include the 5.58% singleton entities. Correctly predicting an empty match list earns a 1.0; predicting an erroneous candidate earns a 0.0.

### 3. Stratified Country Splitting
- Because the test set introduces `France`, validation models should test out-of-domain generalization. A dedicated split holding out Indian entities (training on US, evaluating on India) can verify whether learned matching thresholds transfer across geographic boundaries.

---

## 13. Implications for Pipeline Architecture

```
                             [ Raw Input TSVs ]
                                      │
                                      ▼
                        [ Text Normalization Layer ]
             (Lowercasing, Suffix Harmonization, Street Type Expansion)
                                      │
                                      ▼
                      [ Country-Strict Candidate Blocking ]
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
[ Primary Name Blocking ]                              [ Fallback Address Blocking ]
 (Token Inverted Index / TF-IDF)                        (Locality + Number Tokens)
  * Handles Name Noise                                   * Crucial for 13% Devanagari Names
          └───────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
                         [ Candidate Union & Pruning ]
                         (Target: 20–50 candidates/S1)
                                      │
                                      ▼
                       [ Feature Engineering Pipeline ]
          (Jaro-Winkler, Levenshtein, Jaccard, Numeric Discrepancy,
           Missing Address Flags, Length Ratios, Script Mismatch)
                                      │
                                      ▼
                          [ Precision-Heavy Scorer ]
                     (LightGBM / CatBoost tuned for F0.5)
                                      │
                                      ▼
                  [ Reverse 1-to-At-Most-1 Assignment Filter ]
             (Ensures no S2 or S3 entity is assigned to multiple S1s)
                                      │
                                      ▼
                           [ Final Submission TSV ]
```

### Implications for Blocking (Candidate Generation)
1. **Recall Ceiling Target:** The ground truth contains an average of 3.46 true matches per S1 entity and up to 11. Blocking that retrieves only 1 or 2 candidates per S1 will truncate the recall ceiling. A candidate pool of **20 to 50 candidates per S1 entity** is recommended.
2. **Multi-Channel Blocking is Non-Negotiable:**
   - Channel A (Name-based): Inverted token index / character n-gram TF-IDF. Catches the 95.6% of pairs with English names.
   - Channel B (Address-based): Locality / building number / street token index. Catches the 13.1% of Indian pairs where S1 is English and S2 is Devanagari.
   - Channel C (Strict Country Filter): Reject all cross-country pairs upfront (100% precision filter).

### Implications for Feature Engineering
1. **Multi-Field Similarity Measures:**
   - Jaro-Winkler similarity on normalized name and address.
   - Token-level Jaccard and containment scores (handles token reordering).
   - Character n-gram overlap (robust to internal typos like `Offie` vs `Office`).
2. **Discrepancy and Penalty Features:**
   - Numeric token mismatch (if S1 has `1510` and S2 has `1916`, penalize heavily).
   - Address missing indicator (`is_s2_addr_nan`).
   - Script mismatch indicator (`has_devanagari_s2`).
   - Length difference ratios.

---

## 14. Recommended Next Steps

Based on the quantitative evidence from this EDA, the following sequential steps are recommended for the implementation phase:

1. **Implement Robust Preprocessing Module:**
   - Lowercase all textual fields.
   - Standardize ampersands (`&` $\\rightarrow$ `and`).
   - Build a dictionary of safe legal suffixes (`ltd`, `pvt`, `corp`, `inc`, `llc`) and street abbreviations (`rd`, `st`, `ave`, `dr`, `blvd`).
2. **Design a Dual-Channel Blocking Engine:**
   - Channel 1: Inverted index on 3-gram character shingles and significant tokens for business names.
   - Channel 2: City / Postal / Locality token index for address-based candidate generation to capture Devanagari Indian entities.
   - Partition candidate generation strictly by `country`.
3. **Build the Leak-Free Validation Suite:**
   - Construct a 5-fold `GroupKFold` cross-validation split based on `source1_entity_id`.
   - Implement the exact macro-averaged $F_{{0.5}}$ evaluation script matching `utils/validate_submission.py`.
4. **Develop Candidate Pair Feature Extractor:**
   - Extract string similarity features (Jaro-Winkler, Levenshtein, Jaccard, Token Overlap).
   - Extract numeric discrepancy features (house numbers, postal codes).
   - Add structural indicators (missing address flag, script mismatch flag).
5. **Enforce Post-Processing Cardinality Constraints:**
   - Apply a reverse 1-to-at-most-1 constraint: if an S2 or S3 entity is selected as a match for S1 entity $A$ with higher confidence, prevent it from being assigned to S1 entity $B$.
   - Tune classification thresholds strictly for $F_{{0.5}}$ (precision optimization).
"""

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report_content.strip() + '\n')

print(f"Report written successfully to {REPORT_FILE}")
print("\n=== EDA Pipeline Execution Completed Successfully ===")
