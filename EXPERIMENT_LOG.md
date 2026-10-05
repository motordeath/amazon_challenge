# Business Entity Resolution — Experiment Log

This document tracks all iterative experiments conducted on the Business Entity Resolution pipeline.
Every experiment records hypothesis, minimal code/config changes, evaluation metrics, failure analysis, and next steps.

---

## Baseline Reference

- **Experiment ID**: BASELINE-GENUINE
- **Date/Time**: 2026-09-26 05:08:00 UTC
- **Configuration**:
  - `token_max_df_abs_other`: 80
  - `token_max_df_abs_s1`: 1500
  - `max_token_doc_frequency`: 0.01
  - `token_top_k_per_s1`: 20
  - `max_candidates_per_s1`: 25
  - `num_leaves`: 63
  - `learning_rate`: 0.05
  - `n_estimators`: 500
  - `scale_pos_weight`: ~9.56 (unconstrained negative/positive ratio)
  - Reverse 1-to-at-most-1 constraint enabled
- **Dataset / Sampling Regime**:
  - Full population: 2,206,821 S1 entities, 5,034,616 S2 entities, 5,285,603 S3 entities.
  - Training regime: 500,001 stratified S1 sample (5-fold GroupKFold on `source1_entity_id`, 100,001 S1 entities in validation fold 0).
  - Also evaluated on 50,000 stratified S1 sample.
- **Blocking Metrics (500k sample)**:
  - Total true pairs: 1,730,632
  - Retrieved true pairs (capped): 832,216
  - Capped blocking recall: **0.4809** (48.09%)
  - Uncapped blocking recall: **0.5974** (59.74%, 1,033,811 / 1,730,636 at top_k=20)
  - Candidate count: 8,787,553
  - Positive candidate pairs: 832,216
  - Negative candidate pairs: 7,955,337
  - Average candidates per S1: 18.70
- **Validation Metrics (500k sample, 100k val)**:
  - Decision threshold $\tau$: **0.9000**
  - **Macro $F_{0.5}$**: **0.6484**
  - Macro Precision: **0.7583**
  - Macro Recall: **0.4906**
  - Singleton Count: 5,738
  - Singleton Accuracy: **94.16%**
  - Singleton False Positive Rate: 5.84%
  - Multi-match entities: 94,263
- **Validation Metrics (50k sample)**:
  - Decision threshold $\tau$: 0.9000
  - **Macro $F_{0.5}$**: **0.6430**
  - Macro Precision: **0.7548**
  - Macro Recall: **0.4845**
  - Singleton Accuracy: **92.51%**
- **Runtime & Memory**:
  - LightGBM training: 64.0s
  - Feature engineering peak RSS delta: ~10.4 GB
- **Teammate Benchmark Reference**: ~0.667

---

## Experiment 1: Relaxing Inverted-Index Document Frequency Cap (`token_max_df_abs_other: 80 -> 400`)

- **Experiment ID**: EXP-001
- **Date/Time**: 2026-09-27 12:15:00 UTC
- **Hypothesis**:
  The baseline `token_max_df_abs_other = 80` is too aggressive. Across 5M records, discriminative industry words (e.g. `logistics`, `dental`, `consulting`, `energy`, `solutions`) and common street name tokens appear $>80$ times and are eliminated from the inverted index, causing $>50\%$ of true matches to never be retrieved. Increasing `token_max_df_abs_other` from 80 to 400 will allow these informative tokens to be indexed, raising blocking recall without causing memory overflow or excessive candidate bloat.
- **Files Changed**:
  - `code/business_entity_resolution/src/business_entity_resolution/config.py`
- **Configuration Changes**:
  - `token_max_df_abs_other`: `80` → `400`
  - All other parameters held constant:
    - `max_candidates_per_s1`: 25 (unchanged)
    - `token_top_k_per_s1`: 20 (unchanged)
    - Ranking logic: unchanged
    - Feature engineering: unchanged
    - LightGBM hyperparameters: unchanged
    - Threshold search grid & beta: unchanged
- **Execution Environment**:
  - Kaggle Notebook (`fuckinggpu/abb-aar-ya-paar`) with Tesla T4 GPU / 30GB RAM, Python 3.12.
- **Results**:
  - **Blocking Metrics**:
    - Uncapped blocking recall: **0.6675** (up from 0.5974)
    - Capped blocking recall: **0.5308** (up from 0.4809)
    - Uncapped candidate pairs: 29,629,557 (avg 60.2 per S1)
    - Capped candidate pairs: 11,424,200 (avg 23.2 per S1)
    - Positive candidate pairs: 918,614
    - Negative candidate pairs: 10,505,586
  - **Validation Metrics**:
    - Decision threshold $\tau$: **0.9100**
    - **Macro $F_{0.5}$**: **0.6833** (up from 0.6484)
    - Macro Precision: **0.7841** (up from 0.7583)
    - Macro Recall: **0.5344** (up from 0.4906)
    - Singleton Accuracy: **93.25%** (down slightly from 94.16%)
  - **Runtime & Memory**:
    - Blocking Runtime: 1115s (~18.6m)
    - Total Runtime: 2494s (~41.5m)
    - Blocking Peak RSS: ~15.8 GB
- **Conclusion**:
  - **Hypothesis Validated**: Increasing the frequency cap from 80 to 400 successfully preserved crucial tokens, leading to an immediate +13% relative improvement in capped blocking recall and pushing the downstream F0.5 score well past the teammate benchmark (0.648 -> 0.683).
  - **New Bottleneck Identified**: While *uncapped* recall increased to 66.75%, the capping step (`max_candidates_per_s1 = 25`) aggressively dropped it down to 53.08%. This massive ~13.7% gap indicates our current candidate ranking (based purely on token Jaccard similarity) is highly suboptimal and fails to place true matches in the top 25, wasting the expanded candidate pool.
- **Next Action**:
  - Do NOT blindly increase the cap yet, as runtime and memory will explode.
  - Instead, we must improve candidate *ranking* before capping. Wait for user authorization for the next experiment.

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
  - **Blocking Metrics**:
    - Uncapped blocking recall: **0.6782** (up from 0.6675 in Exp1)
    - Capped blocking recall: **0.5452** (up from 0.5308 in Exp1)
    - Uncapped candidate pairs: 29,629,557 (avg 60.2 per S1)
    - Capped candidate pairs: 11,424,166 (avg 23.2 per S1)
    - Positive candidate pairs: 943,578 (up from 918,614)
    - Negative candidate pairs: 10,480,588
  - **Validation Metrics**:
    - Decision threshold $\tau$: **0.9100**
    - **Macro $F_{0.5}$**: **0.6904** (up from 0.6833 in Exp1, and 0.6484 in Baseline)
    - Macro Precision: **0.7876** (up from 0.7841)
    - Macro Recall: **0.5470** (up from 0.5344)
    - Singleton Accuracy: **92.95%** (down slightly from 93.25%)
  - **Runtime & Memory**:
    - Blocking Runtime: 1175s (~19.6m)
    - Total Runtime: 2604s (~43.4m)
    - Blocking Peak RSS: ~16.3 GB
- **Conclusion**:
  - **Hypothesis Validated**: Implementing IDF weighting successfully penalized hyper-common tokens and rewarded discriminative tokens. This improved candidate ranking significantly, rescuing an additional 1.5% (absolute) of true positive pairs from being dropped during the top-25 truncation (and even improved the pre-cap channel truncation).
  - F0.5 climbed from 0.683 to 0.690 with virtually no increase in candidate volume or training runtime.
- **Next Action**:
  - Wait for user authorization for the next experiment. The pipeline is now operating at a solid 0.690 F0.5.

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
  - Exact sparse cosine similarity using `TruncatedSVD(n_components=64) + faiss.IndexFlatIP(64)`
- **Configuration**:
  - `top_k` (Vector): 10 candidates per entity for Name, 10 for Address.
  - Total `max_candidates_per_s1` (Global Cap): 25 (kept from EXP-002)
- **Execution Environment**:
  - Kaggle Notebook (`fuckinggpu/abb-aar-ya-paar`) with Tesla T4 GPU / 30GB RAM, Python 3.12.
- **Results**:
  - [PENDING]
