with open(r'D:\ml_amazon\EXPERIMENT_LOG.md', 'a') as f:
    f.write('''
---

## Experiment 2: IDF-Weighted Candidate Ranking

- **Experiment ID**: EXP-002
- **Date/Time**: 2026-09-27 13:30:00 UTC
- **Hypothesis**:
  Experiment 1 showed that relaxing `token_max_df_abs_other` to 400 increased uncapped blocking recall to 66.75%, but the hard top-25 cap reduced it to 53.08%. The current candidate ranking uses basic token-count/Jaccard similarity, which gives too much influence to common tokens. Implementing IDF-weighted token ranking (where rare/discriminative tokens contribute more to the score and common tokens contribute less) will ensure that true entity matches sharing distinctive tokens are ranked higher and are more likely to survive the top-25 cutoff.
- **Files Changed**:
  - `code/business_entity_resolution/src/business_entity_resolution/blocking/token.py`
- **Ranking Formula**:
  - `IDF = log(N_other / DF_other) + 1.0` (Smoothed IDF calculated on the target partition, i.e. `other`).
  - `Score = sum(IDF of shared tokens) / max_possible_score_in_group`
- **Configuration**:
  - `token_max_df_abs_other`: 400 (kept from EXP-001)
  - `max_candidates_per_s1`: 25 (kept from EXP-001)
  - All other parameters held constant.
- **Execution Environment**:
  - Kaggle Notebook (`fuckinggpu/abb-aar-ya-paar`) with Tesla T4 GPU / 30GB RAM, Python 3.12.
- **Results**:
  - [PENDING]
''')
