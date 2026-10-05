import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import random
import re
from rapidfuzz.distance import Levenshtein, JaroWinkler

# Deterministic seed
random.seed(42)
np.random.seed(42)

print("Loading data for hard negative analysis...")
s1 = pd.read_csv('train/train_source1.tsv', sep='\t')
gt = pd.read_csv('train/train_ground_truth.tsv', sep='\t')

# Build mapping of S1 -> true matched IDs
gt_map = {}
for s1_id, m in zip(gt['source1_entity_id'], gt['matched_entity_ids'].fillna('')):
    gt_map[s1_id] = set(m.split(',')) if m else set()

# Reverse mapping: mid -> true s1_id
reverse_gt = {}
for s1_id, mset in gt_map.items():
    for mid in mset:
        reverse_gt[mid] = s1_id

# Sample 500,000 S2 rows
s2 = pd.read_csv('train/train_source2.tsv', sep='\t', nrows=500000)

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Find Hard Negatives:
# Type 1: Same/similar normalized name, DIFFERENT entity (check if mid is NOT in gt_map[s1])
s1_sample = s1.sample(n=100000, random_state=42)
s1_name_dict = {}
for _, r in s1_sample.iterrows():
    norm_n = normalize_text(r['business_name'])
    if len(norm_n) > 5:
        s1_name_dict.setdefault(norm_n, []).append(r)

hard_neg_type1 = [] # Same name, different entity
for _, r2 in s2.iterrows():
    norm_n2 = normalize_text(r2['business_name'])
    if norm_n2 in s1_name_dict:
        for r1 in s1_name_dict[norm_n2]:
            s1_id = r1['entity_id']
            mid = r2['entity_id']
            # Is this a true match?
            if mid not in gt_map.get(s1_id, set()):
                # This is a HARD NEGATIVE (same name, different entity!)
                addr_jw = JaroWinkler.similarity(normalize_text(r1['business_address']), normalize_text(r2['business_address']))
                hard_neg_type1.append({
                    's1_id': s1_id, 'mid': mid,
                    'name': r1['business_name'], 'name2': r2['business_name'],
                    'a1': r1['business_address'], 'a2': r2['business_address'],
                    'c1': r1['country'], 'c2': r2['country'],
                    'addr_jw': addr_jw,
                    'true_s1': reverse_gt.get(mid, 'Unmatched/Distractor')
                })
                if len(hard_neg_type1) >= 20000:
                    break
    if len(hard_neg_type1) >= 20000:
        break

df_hn1 = pd.DataFrame(hard_neg_type1)
print(f"Found {len(df_hn1)} hard negatives of Type 1 (identical/near-identical name, different entity).")
print(f"Average Address JW on same-name negatives: {df_hn1['addr_jw'].mean():.4f}")
print(f"Percentage in same country: {(df_hn1['c1'] == df_hn1['c2']).mean() * 100:.2f}%")

print("\n--- Examples of Hard Negatives Type 1 (Same Name, Different Entity) ---")
for idx, r in df_hn1.head(5).iterrows():
    print(f"S1 ({r['s1_id']}): {r['name']} | Addr: {r['a1']} | Country: {r['c1']}")
    print(f"S2 ({r['mid']}): {r['name2']} | Addr: {r['a2']} | Country: {r['c2']}")
    print(f"True S1 for S2: {r['true_s1']} | Addr JW: {r['addr_jw']:.3f}\n")

# Type 2: Same address, different business name
s1_addr_dict = {}
for _, r in s1_sample.iterrows():
    norm_a = normalize_text(r['business_address'])
    if len(norm_a) > 10:
        s1_addr_dict.setdefault(norm_a, []).append(r)

hard_neg_type2 = []
for _, r2 in s2.iterrows():
    norm_a2 = normalize_text(r2['business_address'])
    if norm_a2 in s1_addr_dict:
        for r1 in s1_addr_dict[norm_a2]:
            s1_id = r1['entity_id']
            mid = r2['entity_id']
            if mid not in gt_map.get(s1_id, set()):
                name_jw = JaroWinkler.similarity(normalize_text(r1['business_name']), normalize_text(r2['business_name']))
                hard_neg_type2.append({
                    's1_id': s1_id, 'mid': mid,
                    'name1': r1['business_name'], 'name2': r2['business_name'],
                    'addr': r1['business_address'],
                    'c1': r1['country'], 'c2': r2['country'],
                    'name_jw': name_jw,
                    'true_s1': reverse_gt.get(mid, 'Unmatched/Distractor')
                })
                if len(hard_neg_type2) >= 5000:
                    break
    if len(hard_neg_type2) >= 5000:
        break

df_hn2 = pd.DataFrame(hard_neg_type2)
print(f"Found {len(df_hn2)} hard negatives of Type 2 (same normalized address, different entity).")
print(f"Average Name JW on same-address negatives: {df_hn2['name_jw'].mean():.4f}")

print("\n--- Examples of Hard Negatives Type 2 (Same Address, Different Entity) ---")
for idx, r in df_hn2.head(5).iterrows():
    print(f"S1 ({r['s1_id']}): Name: {r['name1']} | Country: {r['c1']}")
    print(f"S2 ({r['mid']}): Name: {r['name2']} | Country: {r['c2']}")
    print(f"Address: {r['addr']}")
    print(f"True S1 for S2: {r['true_s1']} | Name JW: {r['name_jw']:.3f}\n")
