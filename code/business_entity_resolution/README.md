# Business Entity Resolution Pipeline

Solution for the ML Challenge 2026 Business Entity Resolution task: given a
deduplicated reference source (S1) and two noisy operational sources (S2,
S3), predict every S2/S3 record that refers to the same real-world business
as each S1 entity. An S1 entity may have zero, one, or many matches. Scored
by macro-averaged F₀.₅ (precision-weighted).

See `../../eda_findings.md` for the full EDA this design is based on, and
`../../problem_statement.md` for the original task spec.

## 1. Architecture

```
Raw TSV (S1/S2/S3, ground truth)
    │
    ▼
data/loader.py            memory-conscious load, compact dtypes (pyarrow strings, categorical country)
    │
    ▼
normalization/            names.py, addresses.py — multiple preserved representations
    │                     (raw / normalized / compact / core / tokens / char-ngrams),
    │                     never a single destructive canonical string
    ▼
blocking/                 country-strict, multi-channel union:
    │                       - exact.py:        exact name_compact match
    │                       - tfidf_blocker.py: char-ngram TF-IDF kNN on name AND on address
    │                     candidate_generator.py unions channels, caps top-K per S1
    ▼
features/                 name/address similarity (Jaro-Winkler, Levenshtein, Jaccard on
    │                     tokens/ngrams/numbers), country agreement, blocking rank/score
    ▼
models/train_lgbm.py      LightGBM binary classifier, GroupKFold on source1_entity_id
    │                     (leakage guard — EDA section 12)
    ▼
models/threshold.py       grid-search decision threshold maximizing macro F₀.₅ on a
    │                     held-out validation split
    ▼
models/one_to_one_constraint.py
    │                     reverse-uniqueness constraint: an S2/S3 record can belong to
    │                     at most one S1 (100% true in ground truth — EDA section 7);
    │                     keeps only the highest-scoring S1 claim per candidate
    ▼
postprocess/              assemble_outputs.py writes matching_results.tsv / candidate_pairs.tsv;
                          sanity_checks.py verifies the submission invariants
```

## 2. Why this design

- **Country is never hard-coded.** `country` is compared as an open string
  label everywhere; blocking partitions `groupby("country")` dynamically so
  test-only `France` is handled automatically (task spec section on
  open-set country; EDA section 6).
- **Blocking is a union of channels, never an intersection** (task spec
  section 11F): exact match, name TF-IDF, and address TF-IDF are combined so
  a Devanagari-script S2 name with a still-Latin-script address (EDA section
  4) is still recoverable via the address channel even when the name channel
  fails outright.
- **Normalization preserves multiple representations** instead of one
  aggressive canonical string, because the EDA flags full suffix-stripping,
  number removal, and stopword removal as "potentially dangerous" — e.g.
  `Summit Inc` vs `Summit LLC` may only differ by the legal suffix.
- **The decision layer is not `argmax`.** 89% of S1 entities have 2+ true
  matches (EDA section 7), so the final step keeps every candidate above a
  tuned threshold, not just the single best one — while singleton entities
  (5.58%) are handled correctly by predicting an empty set when nothing
  clears the bar.
- **Reverse-uniqueness post-processing** exploits a hard invariant found in
  the ground truth (0% of S2/S3 records match more than one S1 entity) to
  remove conflicting low-confidence claims without ever removing a
  legitimate multi-match for the correct S1 entity.

## 3. Environment setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ recommended. All models/libraries used (LightGBM, scikit-learn,
rapidfuzz) are MIT/BSD-licensed and far under the 8B-parameter cap (LightGBM
trees have no "parameters" in the LLM sense; the largest artifact is a few
hundred trees at `num_leaves=63`).

## 4. Reproduction

Run from this directory (`code/business_entity_resolution/`). Paths to the
dataset are resolved from `config.py` relative to the repo root
(`<repo>/dataset/train`, `<repo>/dataset/test`) — no other configuration is
required if the repo layout is preserved.

```bash
# 1. Train: normalizes, blocks, engineers features, trains LightGBM,
#    tunes the decision threshold, and writes artifacts/*.joblib + *.json
python -m business_entity_resolution train

# 2. Predict: loads the saved artifacts, runs the same blocking + feature
#    pipeline on dataset/test, and writes both submission files
python -m business_entity_resolution predict

# Or both in one call:
python -m business_entity_resolution full-pipeline
```

Outputs land in `<repo>/output/matching_results.tsv` and
`<repo>/output/candidate_pairs.tsv` by default (override with
`--matching-out` / `--candidate-out` on `predict`).

`--nrows N` on any subcommand truncates each input file to its first N rows
for a fast dev/debug run (rows are **not** ID-aligned across files at small
N, so blocking recall on a truncated run is not meaningful — use it only to
check the code path runs, not to judge quality).

### Validate before submitting

```bash
cd ../..   # repo root == student_resource-equivalent layout
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## 5. Reproducibility

- `RANDOM_SEED = 42` in `config.py` is threaded through the GroupKFold split
  and LightGBM's `random_state`.
- All hyperparameters (blocking caps, feature toggles, model params,
  threshold grid) live in `config.py` — nothing is hard-coded deeper in the
  pipeline.
- Dependencies are pinned to major/minor floors in `requirements.txt`.

## 6. Fair-play / license notes

- No external data, APIs, geocoders, or business registries are used
  anywhere in the pipeline — every signal is derived from
  `business_name` / `business_address` / `country` in the provided TSVs.
- Model: LightGBM (MIT license), a gradient-boosted tree ensemble — not an
  LLM, well under the 8B-parameter constraint.

## 7. Known limitations / next steps

See `notebooks/08_error_analysis.ipynb` for the empirical breakdown. In
summary: address-missing candidates (~3.3% of S2/S3) lean entirely on name
similarity; Devanagari-script names rely on the address TF-IDF channel for
blocking recall since char n-grams across scripts never match on the name
channel itself; both are documented failure modes rather than silently
dropped.
