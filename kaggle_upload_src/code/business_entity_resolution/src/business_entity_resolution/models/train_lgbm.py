"""Pairwise LightGBM classifier training with entity-level grouping.

Leakage guard (EDA section 12): candidate pairs must be split by
``source1_entity_id`` GroupKFold, never by individual pair, since many
candidate pairs share the same S1 entity and a naive row-level split would
let the model memorize entity-specific quirks.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..config import ModelConfig

logger = logging.getLogger(__name__)


def make_group_split(groups: pd.Series, n_splits: int = 5, fold: int = 0, seed: int = 42):
    """Deterministic single train/val split via GroupKFold (leakage-free on group)."""
    gkf = GroupKFold(n_splits=n_splits)
    dummy_X = np.zeros((len(groups), 1))
    splits = list(gkf.split(dummy_X, groups=groups))
    train_idx, val_idx = splits[fold]
    return train_idx, val_idx


def split_all_s1_ids(all_s1_ids, n_splits: int = 5, fold: int = 0, seed: int = 42) -> tuple[set, set]:
    """Assign EVERY S1 entity (not just ones with >=1 candidate) to a fold.

    ``make_group_split`` above only ever sees S1 ids that survived blocking
    with at least one candidate — an S1 entity blocking retrieved zero
    candidates for never appears in ``groups`` there, so it can't be
    assigned to any fold, and silently drops out of validation entirely.
    That's a real (if usually small) bias in the *reported* validation
    macro F_0.5: it means the "all_s1_ids" the threshold grid search
    evaluates over is missing exactly the zero-candidate entities, most of
    which are correctly-predicted-empty true singletons that would score a
    full 1.0. This function instead splits on the complete S1 id space, so
    the caller can build train/val row masks by id membership, and val
    always covers every S1 entity assigned to that fold regardless of
    whether it has any candidates.
    """
    ids = np.array(sorted(set(all_s1_ids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    folds = np.array_split(ids, n_splits)
    val_ids = set(folds[fold].tolist())
    train_ids = set(ids.tolist()) - val_ids
    return train_ids, val_ids


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: ModelConfig,
) -> lgb.LGBMClassifier:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        objective=cfg.objective,
        boosting_type=cfg.boosting_type,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        scale_pos_weight=scale_pos_weight,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=cfg.metric,
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def feature_importance(model: lgb.LGBMClassifier, feature_names: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    })
    return imp.sort_values("importance_gain", ascending=False).reset_index(drop=True)
