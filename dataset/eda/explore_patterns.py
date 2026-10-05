import pandas as pd
import numpy as np
import re
from collections import Counter

print("Loading S1 and S2...")
s1 = pd.read_csv('train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('train/train_source2.tsv', sep='\t', nrows=1000000)

def analyze_names(df, name_col='business_name'):
    s = df[name_col].dropna().astype(str)
    
    char_lens = s.str.len()
    word_lens = s.str.split().str.len()
    
    has_digit = s.str.contains(r'\d', regex=True).mean() * 100
    has_punct = s.str.contains(r'[^\w\s]', regex=True).mean() * 100
    is_upper = (s.str.upper() == s).mean() * 100
    is_lower = (s.str.lower() == s).mean() * 100
    is_title = s.str.istitle().mean() * 100
    has_amp = s.str.contains(r'&', regex=False).mean() * 100
    has_and = s.str.contains(r'\band\b', case=False, regex=True).mean() * 100
    
    # Specific noise keywords
    corp = s.str.contains(r'\bcorp\b', case=False, regex=True).mean() * 100
    corporation = s.str.contains(r'\bcorporation\b', case=False, regex=True).mean() * 100
    inc = s.str.contains(r'\binc\b', case=False, regex=True).mean() * 100
    incorporated = s.str.contains(r'\bincorporated\b', case=False, regex=True).mean() * 100
    ltd = s.str.contains(r'\bltd\b', case=False, regex=True).mean() * 100
    limited = s.str.contains(r'\blimited\b', case=False, regex=True).mean() * 100
    pvt = s.str.contains(r'\bpvt\b', case=False, regex=True).mean() * 100
    private = s.str.contains(r'\bprivate\b', case=False, regex=True).mean() * 100
    llc = s.str.contains(r'\bllc\b', case=False, regex=True).mean() * 100
    co = s.str.contains(r'\bco\b', case=False, regex=True).mean() * 100
    company = s.str.contains(r'\bcompany\b', case=False, regex=True).mean() * 100

    print("Char length: mean=%.1f, median=%.1f, min=%d, max=%d" % (char_lens.mean(), char_lens.median(), char_lens.min(), char_lens.max()))
    print("Token length: mean=%.1f, median=%.1f, min=%d, max=%d" % (word_lens.mean(), word_lens.median(), word_lens.min(), word_lens.max()))
    print("Digits: %.2f%%, Punct: %.2f%%, Upper: %.2f%%, Lower: %.2f%%, Title: %.2f%%" % (has_digit, has_punct, is_upper, is_lower, is_title))
    print("&: %.2f%%, and: %.2f%%" % (has_amp, has_and))
    print("Corp: %.2f%% vs Corporation: %.2f%%" % (corp, corporation))
    print("Inc: %.2f%% vs Incorporated: %.2f%%" % (inc, incorporated))
    print("Ltd: %.2f%% vs Limited: %.2f%%" % (ltd, limited))
    print("Pvt: %.2f%% vs Private: %.2f%%" % (pvt, private))
    print("LLC: %.2f%%, Co: %.2f%%, Company: %.2f%%" % (llc, co, company))

print("=== S1 Business Name Stats ===")
analyze_names(s1)

print("\n=== S2 Business Name Stats ===")
analyze_names(s2)

def analyze_addresses(df, addr_col='business_address'):
    s = df[addr_col].dropna().astype(str)
    
    char_lens = s.str.len()
    word_lens = s.str.split().str.len()
    
    has_digit = s.str.contains(r'\d', regex=True).mean() * 100
    has_punct = s.str.contains(r'[^\w\s]', regex=True).mean() * 100
    is_upper = (s.str.upper() == s).mean() * 100
    is_lower = (s.str.lower() == s).mean() * 100
    
    # Specific address tokens
    rd = s.str.contains(r'\brd\b', case=False, regex=True).mean() * 100
    road = s.str.contains(r'\broad\b', case=False, regex=True).mean() * 100
    st = s.str.contains(r'\bst\b', case=False, regex=True).mean() * 100
    street = s.str.contains(r'\bstreet\b', case=False, regex=True).mean() * 100
    ave = s.str.contains(r'\bave\b', case=False, regex=True).mean() * 100
    avenue = s.str.contains(r'\bavenue\b', case=False, regex=True).mean() * 100
    dr = s.str.contains(r'\bdr\b', case=False, regex=True).mean() * 100
    drive = s.str.contains(r'\bdrive\b', case=False, regex=True).mean() * 100
    near = s.str.contains(r'\bnear\b', case=False, regex=True).mean() * 100
    opp = s.str.contains(r'\bopp\b|\bopposite\b', case=False, regex=True).mean() * 100
    flr = s.str.contains(r'\bflr\b|\bfloor\b', case=False, regex=True).mean() * 100
    
    # Zip / Pin code patterns
    # US 5 digit: \b\d{5}\b
    # India 6 digit: \b\d{6}\b
    us_zip = s.str.contains(r'\b\d{5}\b', regex=True).mean() * 100
    in_pin = s.str.contains(r'\b\d{6}\b', regex=True).mean() * 100

    print("Char length: mean=%.1f, median=%.1f, min=%d, max=%d" % (char_lens.mean(), char_lens.median(), char_lens.min(), char_lens.max()))
    print("Token length: mean=%.1f, median=%.1f, min=%d, max=%d" % (word_lens.mean(), word_lens.median(), word_lens.min(), word_lens.max()))
    print("Digits: %.2f%%, Punct: %.2f%%, Upper: %.2f%%, Lower: %.2f%%" % (has_digit, has_punct, is_upper, is_lower))
    print("Rd: %.2f%% vs Road: %.2f%%" % (rd, road))
    print("St: %.2f%% vs Street: %.2f%%" % (st, street))
    print("Ave: %.2f%% vs Avenue: %.2f%%" % (ave, avenue))
    print("Dr: %.2f%% vs Drive: %.2f%%" % (dr, drive))
    print("Landmarks (Near: %.2f%%, Opp: %.2f%%, Flr: %.2f%%)" % (near, opp, flr))
    print("Postal codes (5-digit US ZIP: %.2f%%, 6-digit India PIN: %.2f%%)" % (us_zip, in_pin))

print("\n=== S1 Business Address Stats ===")
analyze_addresses(s1)

print("\n=== S2 Business Address Stats ===")
analyze_addresses(s2)
