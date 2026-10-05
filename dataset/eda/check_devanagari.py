import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re

df_res = pd.read_pickle('eda/outputs/sampled_positive_pairs.pkl')
devan_matches = df_res[df_res['n2'].str.contains(r'[\u0900-\u097F]', regex=True)]
print(f"Sampled positive pairs with Devanagari in S2/S3 name: {len(devan_matches)} ({len(devan_matches)/len(df_res)*100:.2f}%)")
print("Among India pairs: %.2f%%" % (len(devan_matches) / len(df_res[df_res['country'] == 'India']) * 100))

print("\nExamples of True Matches with Devanagari in S2/S3:")
for _, r in devan_matches.head(5).iterrows():
    print(f"S1: {r['n1']} | {r['a1']}")
    print(f"{r['src']}: {r['n2']} | {r['a2']}")
    print(f"Name JW: {r['name_jw']:.3f} | Addr JW: {r['addr_jw']:.3f}\n")
