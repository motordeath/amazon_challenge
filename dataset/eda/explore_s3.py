import pandas as pd
import numpy as np

print("Loading S3 sample...")
s3 = pd.read_csv('train/train_source3.tsv', sep='\t', nrows=1000000)

s_name = s3['business_name'].dropna().astype(str)
s_addr = s3['business_address'].dropna().astype(str)

print("=== S3 Business Name Stats ===")
print("Char len mean=%.1f, median=%.1f, min=%d, max=%d" % (s_name.str.len().mean(), s_name.str.len().median(), s_name.str.len().min(), s_name.str.len().max()))
print("Token len mean=%.1f, median=%.1f, min=%d, max=%d" % (s_name.str.split().str.len().mean(), s_name.str.split().str.len().median(), s_name.str.split().str.len().min(), s_name.str.split().str.len().max()))
print("Upper: %.2f%%, Lower: %.2f%%, Title: %.2f%%, Punct: %.2f%%" % (
    (s_name.str.upper() == s_name).mean()*100,
    (s_name.str.lower() == s_name).mean()*100,
    s_name.str.istitle().mean()*100,
    s_name.str.contains(r'[^\w\s]', regex=True).mean()*100
))
print("Ltd: %.2f%% vs Limited: %.2f%%" % (
    s_name.str.contains(r'\bltd\b', case=False, regex=True).mean()*100,
    s_name.str.contains(r'\blimited\b', case=False, regex=True).mean()*100
))
print("Pvt: %.2f%% vs Private: %.2f%%" % (
    s_name.str.contains(r'\bpvt\b', case=False, regex=True).mean()*100,
    s_name.str.contains(r'\bprivate\b', case=False, regex=True).mean()*100
))
print("Corp: %.2f%% vs Corporation: %.2f%%" % (
    s_name.str.contains(r'\bcorp\b', case=False, regex=True).mean()*100,
    s_name.str.contains(r'\bcorporation\b', case=False, regex=True).mean()*100
))

print("\n=== S3 Business Address Stats ===")
print("Char len mean=%.1f, median=%.1f, min=%d, max=%d" % (s_addr.str.len().mean(), s_addr.str.len().median(), s_addr.str.len().min(), s_addr.str.len().max()))
print("Token len mean=%.1f, median=%.1f, min=%d, max=%d" % (s_addr.str.split().str.len().mean(), s_addr.str.split().str.len().median(), s_addr.str.split().str.len().min(), s_addr.str.split().str.len().max()))
print("Upper: %.2f%%, Lower: %.2f%%, Punct: %.2f%%" % (
    (s_addr.str.upper() == s_addr).mean()*100,
    (s_addr.str.lower() == s_addr).mean()*100,
    s_addr.str.contains(r'[^\w\s]', regex=True).mean()*100
))
print("Rd: %.2f%% vs Road: %.2f%%" % (
    s_addr.str.contains(r'\brd\b', case=False, regex=True).mean()*100,
    s_addr.str.contains(r'\broad\b', case=False, regex=True).mean()*100
))
print("St: %.2f%% vs Street: %.2f%%" % (
    s_addr.str.contains(r'\bst\b', case=False, regex=True).mean()*100,
    s_addr.str.contains(r'\bstreet\b', case=False, regex=True).mean()*100
))
