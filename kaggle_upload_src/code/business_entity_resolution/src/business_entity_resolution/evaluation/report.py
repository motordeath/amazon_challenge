"""Full validation report: macro F0.5 plus precision/recall/singleton breakdown.

Precision and recall here are macro-averaged the same way F_0.5 is (per-S1,
then averaged) rather than micro/aggregate, so all three numbers are
comparable on the same per-entity basis the competition metric uses.
"""
from __future__ import annotations

from .f_beta_macro import f_beta_score


def _precision_recall(predicted: set, actual: set) -> tuple[float, float]:
    """Per-entity precision/recall with the same edge-case convention as f_beta_score:
    correct empty prediction on a true singleton -> (1.0, 1.0); any other
    empty/actual-empty mismatch -> (0.0, 0.0)."""
    if not actual and not predicted:
        return 1.0, 1.0
    if not predicted or not actual:
        return 0.0, 0.0
    tp = len(predicted & actual)
    precision = tp / len(predicted)
    recall = tp / len(actual)
    return precision, recall


def full_validation_report(
    predictions: dict[str, set],
    ground_truth: dict[str, set],
    beta: float = 0.5,
) -> dict:
    f_betas, precisions, recalls = [], [], []
    singleton_total = 0
    singleton_correct = 0
    multi_total = 0

    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        f_betas.append(f_beta_score(predicted, actual, beta=beta))
        p, r = _precision_recall(predicted, actual)
        precisions.append(p)
        recalls.append(r)

        if len(actual) == 0:
            singleton_total += 1
            if len(predicted) == 0:
                singleton_correct += 1
        else:
            multi_total += 1

    n = len(f_betas)
    return {
        "n_entities": n,
        "macro_f_beta": sum(f_betas) / n if n else float("nan"),
        "macro_precision": sum(precisions) / n if n else float("nan"),
        "macro_recall": sum(recalls) / n if n else float("nan"),
        "singleton_count": singleton_total,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else float("nan"),
        "singleton_false_positive_rate": 1 - (singleton_correct / singleton_total) if singleton_total else float("nan"),
        "multi_match_count": multi_total,
    }
