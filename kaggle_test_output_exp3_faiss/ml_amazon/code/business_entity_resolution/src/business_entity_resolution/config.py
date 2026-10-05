"""Central configuration for the Business Entity Resolution pipeline.

All paths, seeds and tunable hyperparameters live here so nothing is
hard-coded deeper in the pipeline. Override any of these from the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RANDOM_SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = REPO_ROOT / "output"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

TRAIN_FILES = {
    "S1": TRAIN_DIR / "train_source1.tsv",
    "S2": TRAIN_DIR / "train_source2.tsv",
    "S3": TRAIN_DIR / "train_source3.tsv",
}
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_FILES = {
    "S1": TEST_DIR / "test_source1.tsv",
    "S2": TEST_DIR / "test_source2.tsv",
    "S3": TEST_DIR / "test_source3.tsv",
}


@dataclass
class BlockingConfig:
    # Token document-frequency cap (per country partition): tokens that appear
    # in more than this fraction of records in a source are considered
    # non-discriminative and excluded from token blocking (e.g. "inc", "llc").
    max_token_doc_frequency: float = 0.01
    min_token_length: int = 2
    # Inverted-index token blocking (blocking/token.py): tokens with a
    # document frequency above these caps are dropped from the index
    # entirely (not down-weighted) so every posting list stays short and the
    # hash-join candidate generation stays near-linear in corpus size.
    token_max_df_abs_other: int = 400
    token_max_df_abs_s1: int = 1500
    # per-channel cap applied before the multi-channel union (see token.py).
    # Tuned down from an initial 40 after full-scale runs on a 25GB machine
    # showed the downstream feature matrix (float64, ~15 columns) at the
    # larger cap needed more memory than was reliably available alongside
    # the rest of the pipeline; 20 keeps candidates/s1 well above the
    # EDA's observed mean of 3.46 true matches while bounding memory.
    token_top_k_per_s1: int = 20
    # Hard cap on candidates kept per S1 entity after union, ranked by
    # blocking score, to bound feature engineering / model inference cost.
    max_candidates_per_s1: int = 25
    # Minimum number of shared informative tokens/ngrams to accept a
    # candidate at all (avoids weak single-common-token noise pairs).
    min_shared_tokens: int = 1


@dataclass
class FeatureConfig:
    use_name_features: bool = True
    use_address_features: bool = True
    use_cross_features: bool = True
    use_rank_features: bool = True


@dataclass
class ModelConfig:
    objective: str = "binary"
    metric: str = "binary_logloss"
    boosting_type: str = "gbdt"
    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 30
    random_state: int = RANDOM_SEED
    n_jobs: int = -1


@dataclass
class ThresholdConfig:
    grid_min: float = 0.05
    grid_max: float = 0.95
    grid_step: float = 0.01
    beta: float = 0.5


@dataclass
class PipelineConfig:
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    random_seed: int = RANDOM_SEED


DEFAULT_CONFIG = PipelineConfig()
