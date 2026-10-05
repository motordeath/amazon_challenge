import sys

path = r'D:\ml_amazon\EXPERIMENT_LOG.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

stub = """
---

## Experiment 3: Vector-Assisted Candidate Retrieval

- **Experiment ID**: EXP-003
- **Date/Time**: 2026-09-27 14:40:00 UTC
- **Hypothesis**:
  While IDF lexical ranking improves candidate survival, vector/embedding-based retrieval (e.g., character n-gram cosine similarity) can recover additional true matches that lexical blocking misses completely (e.g., severe transliteration or missing tokens). By unioning vector candidates with lexical candidates before the final cap, we can boost candidate recall.
- **Files Changed**:
  - `code/business_entity_resolution/src/business_entity_resolution/blocking/vector.py`
  - `code/business_entity_resolution/exp3_script.py`
- **Vector Representation**:
  - `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=0.5, min_df=2)`
  - Exact sparse cosine similarity using `NearestNeighbors(metric="cosine", algorithm="brute")`
- **Configuration**:
  - `top_k` (Vector): 10 candidates per entity for Name, 10 for Address.
  - Total `max_candidates_per_s1` (Global Cap): 25 (kept from EXP-002)
- **Execution Environment**:
  - Kaggle Notebook (`fuckinggpu/abb-aar-ya-paar`) with Tesla T4 GPU / 30GB RAM, Python 3.12.
- **Results**:
  - [PENDING]
"""

content += stub

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
