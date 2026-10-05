import pandas as pd
import numpy as np

print("Loading S2...")
df2 = pd.read_csv('train/train_source2.tsv', sep='\t')

# Duplicate (name, address)
dup_both = df2[df2.duplicated(subset=['business_name', 'business_address'], keep=False)].dropna(subset=['business_name', 'business_address'])
print("=== Example of duplicate (business_name, business_address) in S2 ===")
sample_dups = dup_both.sort_values(by=['business_name', 'business_address']).head(10)
for idx, r in sample_dups.iterrows():
    print(r['entity_id'], "|", r['business_name'], "|", r['business_address'], "|", r['country'])

# Duplicate address, different name
dup_addr = df2[df2.duplicated(subset=['business_address'], keep=False)].dropna(subset=['business_address'])
grouped = dup_addr.groupby('business_address')['business_name'].nunique()
multi_name_addrs = grouped[grouped > 1].index
print("\n=== Example of same address, different business_name in S2 ===")
for addr in multi_name_addrs[:3]:
    subset = df2[df2['business_address'] == addr]
    print(f"Address: {addr}")
    for _, r in subset.head(3).iterrows():
        print("  ", r['entity_id'], "|", r['business_name'])

# Duplicate name, different address
dup_name = df2[df2.duplicated(subset=['business_name'], keep=False)].dropna(subset=['business_name'])
grouped_names = dup_name.groupby('business_name')['business_address'].nunique()
multi_addr_names = grouped_names[grouped_names > 2].index
print("\n=== Example of same business_name, different address in S2 (chains/branches) ===")
for nm in multi_addr_names[:3]:
    subset = df2[df2['business_name'] == nm]
    print(f"Business Name: {nm}")
    for _, r in subset.head(3).iterrows():
        print("  ", r['entity_id'], "|", r['business_address'], "|", r['country'])
