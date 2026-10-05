import pandas as pd
import numpy as np
import random
import re
from rapidfuzz.distance import Levenshtein, JaroWinkler

# Deterministic seed
random.seed(42)
np.random.seed(42)

print("Loading data for pair analysis...")
gt = pd.read_csv('train/train_ground_truth.tsv', sep='\t')
s1 = pd.read_csv('train/train_source1.tsv', sep='\t').set_index('entity_id')

# Sample 50,000 S1 entities that have matches
has_match = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')]
sample_gt = has_match.sample(n=25000, random_state=42)

# Collect matched S2 and S3 IDs
s2_ids_needed = set()
s3_ids_needed = set()
pairs = []

for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    m_list = row['matched_entity_ids'].split(',')
    for mid in m_list:
        if mid.startswith('S2-'):
            s2_ids_needed.add(mid)
            pairs.append((s1_id, mid, 'S2'))
        elif mid.startswith('S3-'):
            s3_ids_needed.add(mid)
            pairs.append((s1_id, mid, 'S3'))

print(f"Total positive pairs sampled: {len(pairs)} (S2: {len(s2_ids_needed)}, S3: {len(s3_ids_needed)})")

# Load S2 and S3 for needed IDs
print("Loading S2 records...")
s2_records = {}
for chunk in pd.read_csv('train/train_source2.tsv', sep='\t', chunksize=500000):
    match_chunk = chunk[chunk['entity_id'].isin(s2_ids_needed)]
    for _, r in match_chunk.iterrows():
        s2_records[r['entity_id']] = r.to_dict()

print("Loading S3 records...")
s3_records = {}
for chunk in pd.read_csv('train/train_source3.tsv', sep='\t', chunksize=500000):
    match_chunk = chunk[chunk['entity_id'].isin(s3_ids_needed)]
    for _, r in match_chunk.iterrows():
        s3_records[r['entity_id']] = r.to_dict()

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def token_jaccard(s1, s2):
    t1 = set(normalize_text(s1).split())
    t2 = set(normalize_text(s2).split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

# Analyze positive pairs
results = []
for s1_id, mid, src in pairs:
    rec1 = s1.loc[s1_id]
    rec2 = s2_records.get(mid) if src == 'S2' else s3_records.get(mid)
    if not rec2:
        continue
    
    n1, n2 = str(rec1['business_name']), str(rec2['business_name'])
    a1, a2 = str(rec1['business_address']), str(rec2['business_address'])
    c1, c2 = str(rec1['country']), str(rec2['country'])
    
    # Exact matches
    exact_name = (n1 == n2)
    norm_name = (normalize_text(n1) == normalize_text(n2))
    name_jw = JaroWinkler.similarity(normalize_text(n1), normalize_text(n2))
    name_lev = Levenshtein.normalized_similarity(normalize_text(n1), normalize_text(n2))
    name_jaccard = token_jaccard(n1, n2)
    
    exact_addr = (a1 == a2)
    norm_addr = (normalize_text(a1) == normalize_text(a2))
    addr_jw = JaroWinkler.similarity(normalize_text(a1), normalize_text(a2)) if a2 != 'nan' else 0.0
    addr_lev = Levenshtein.normalized_similarity(normalize_text(a1), normalize_text(a2)) if a2 != 'nan' else 0.0
    addr_jaccard = token_jaccard(a1, a2) if a2 != 'nan' else 0.0
    
    country_match = (c1 == c2)
    
    results.append({
        's1_id': s1_id, 'mid': mid, 'src': src,
        'country': c1, 'country_match': country_match,
        'exact_name': exact_name, 'norm_name': norm_name,
        'name_jw': name_jw, 'name_lev': name_lev, 'name_jaccard': name_jaccard,
        'exact_addr': exact_addr, 'norm_addr': norm_addr,
        'addr_jw': addr_jw, 'addr_lev': addr_lev, 'addr_jaccard': addr_jaccard,
        'n1': n1, 'n2': n2, 'a1': a1, 'a2': a2, 'a2_is_nan': (a2 == 'nan')
    })

df_res = pd.DataFrame(results)
print(f"\nEvaluated {len(df_res)} matched pairs.")
print("=== Positive Pairs Metrics ===")
print("Exact name match: %.2f%%" % (df_res['exact_name'].mean() * 100))
print("Normalized name match: %.2f%%" % (df_res['norm_name'].mean() * 100))
print("Mean Name Jaro-Winkler: %.4f" % (df_res['name_jw'].mean()))
print("Mean Name Levenshtein Sim: %.4f" % (df_res['name_lev'].mean()))
print("Mean Name Token Jaccard: %.4f" % (df_res['name_jaccard'].mean()))

print("\nExact address match: %.2f%%" % (df_res['exact_addr'].mean() * 100))
print("Normalized address match: %.2f%%" % (df_res['norm_addr'].mean() * 100))
print("Mean Address Jaro-Winkler: %.4f" % (df_res['addr_jw'].mean()))
print("Mean Address Levenshtein Sim: %.4f" % (df_res['addr_lev'].mean()))
print("Mean Address Token Jaccard: %.4f" % (df_res['addr_jaccard'].mean()))
print("Address missing in match: %.2f%%" % (df_res['a2_is_nan'].mean() * 100))

print("\nExact Country Match: %.2f%%" % (df_res['country_match'].mean() * 100))

# Break down by source
for src in ['S2', 'S3']:
    sub = df_res[df_res['src'] == src]
    print(f"\n--- Source {src} Matches ({len(sub)} pairs) ---")
    print("Exact name: %.2f%% | Norm name: %.2f%% | Name JW: %.4f" % (sub['exact_name'].mean()*100, sub['norm_name'].mean()*100, sub['name_jw'].mean()))
    print("Exact addr: %.2f%% | Norm addr: %.2f%% | Addr JW: %.4f | Addr NaN: %.2f%%" % (sub['exact_addr'].mean()*100, sub['norm_addr'].mean()*100, sub['addr_jw'].mean(), sub['a2_is_nan'].mean()*100))

# Break down by country
for ctry in ['US', 'India']:
    sub = df_res[df_res['country'] == ctry]
    print(f"\n--- Country {ctry} Matches ({len(sub)} pairs) ---")
    print("Norm name: %.2f%% | Name JW: %.4f | Addr JW: %.4f | Addr NaN: %.2f%%" % (sub['norm_name'].mean()*100, sub['name_jw'].mean(), sub['addr_jw'].mean(), sub['a2_is_nan'].mean()*100))

# Find categories of matches
print("\n=== Sample: Easy Matches (Name JW > 0.95, Addr JW > 0.95) ===")
easy = df_res[(df_res['name_jw'] > 0.95) & (df_res['addr_jw'] > 0.95)].head(2)
for _, r in easy.iterrows():
    print(f"S1: {r['n1']} | {r['a1']}")
    print(f"{r['src']}: {r['n2']} | {r['a2']}\n")

print("=== Sample: Moderate Noise (0.75 < Name JW < 0.90, 0.70 < Addr JW < 0.85) ===")
mod = df_res[(df_res['name_jw'] >= 0.75) & (df_res['name_jw'] <= 0.90) & (df_res['addr_jw'] >= 0.70) & (df_res['addr_jw'] <= 0.85)].head(2)
for _, r in mod.iterrows():
    print(f"S1: {r['n1']} | {r['a1']}")
    print(f"{r['src']}: {r['n2']} | {r['a2']}\n")

print("=== Sample: Weak Name, Strong Address (Name JW < 0.70, Addr JW > 0.90) ===")
weak_name = df_res[(df_res['name_jw'] < 0.70) & (df_res['addr_jw'] > 0.90)].head(3)
for _, r in weak_name.iterrows():
    print(f"S1: {r['n1']} | {r['a1']}")
    print(f"{r['src']}: {r['n2']} | {r['a2']}\n")

print("=== Sample: Strong Name, Weak/Missing Address (Name JW > 0.95, Addr JW < 0.50) ===")
weak_addr = df_res[(df_res['name_jw'] > 0.95) & (df_res['addr_jw'] < 0.50)].head(3)
for _, r in weak_addr.iterrows():
    print(f"S1: {r['n1']} | {r['a1']}")
    print(f"{r['src']}: {r['n2']} | {r['a2']}\n")

print("=== Sample: Very Noisy Matches (Name JW < 0.75, Addr JW < 0.70) ===")
very_noisy = df_res[(df_res['name_jw'] < 0.75) & (df_res['addr_jw'] < 0.70) & (~df_res['a2_is_nan'])].head(3)
for _, r in very_noisy.iterrows():
    print(f"S1: {r['n1']} | {r['a1']}")
    print(f"{r['src']}: {r['n2']} | {r['a2']}\n")

# Save summary dataframe for reporting
df_res.to_pickle('eda/outputs/sampled_positive_pairs.pkl')
print("Saved sampled positive pairs to eda/outputs/sampled_positive_pairs.pkl")
