import pandas as pd
import re
import sys
from pathlib import Path

# wait for the file to be ready
file_path = Path("scratch/blocking_analysis_gt.csv")
if not file_path.exists():
    print("CSV not found yet.")
    sys.exit(0)

gt = pd.read_csv(file_path)
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
other_df = pd.concat([s2, s3])

gt = gt.merge(s1.rename(columns={"entity_id": "source1_entity_id", "business_name": "s1_name", "business_address": "s1_address"}), on="source1_entity_id", how="left")
gt = gt.merge(other_df.rename(columns={"entity_id": "matched_entity_id", "business_name": "other_name", "business_address": "other_address"}), on="matched_entity_id", how="left")

def has_devanagari(text):
    if pd.isna(text): return False
    return bool(re.search(r'[\u0900-\u097F]', str(text)))

gt["s1_has_devanagari"] = gt["s1_name"].apply(has_devanagari)
gt["other_has_devanagari"] = gt["other_name"].apply(has_devanagari)
gt["pair_has_devanagari"] = gt["s1_has_devanagari"] | gt["other_has_devanagari"]

print("Breakdown by Devanagari script presence:")
for has_dev in [True, False]:
    subset = gt[gt["pair_has_devanagari"] == has_dev]
    if len(subset):
        retrieved = len(subset[~subset["missed_total"]])
        print(f"Has Devanagari={has_dev}: {retrieved}/{len(subset)} ({retrieved/len(subset):.2%})")

print("Breakdown by Address Availability:")
gt["both_addresses_present"] = ~gt["s1_address"].isna() & ~gt["other_address"].isna()
for both_avail in [True, False]:
    subset = gt[gt["both_addresses_present"] == both_avail]
    if len(subset):
        retrieved = len(subset[~subset["missed_total"]])
        print(f"Both Addresses Present={both_avail}: {retrieved}/{len(subset)} ({retrieved/len(subset):.2%})")

print("Overall Missing Analysis:")
print(f"Missed purely by cap: {gt['missed_capped'].sum()}")
print(f"Missed purely by country partitioning: {gt[gt['missed_uncapped'] & ~gt['same_country']].shape[0]}")
print(f"Missed by not retrieved in any channel: {gt['missed_uncapped'].sum()}")
