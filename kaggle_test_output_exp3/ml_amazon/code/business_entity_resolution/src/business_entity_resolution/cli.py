"""Command-line entry point: ``python -m business_entity_resolution <command>``."""
from __future__ import annotations

import argparse
import json
import logging

from . import config
from .pipeline.inference_pipeline import run_inference_pipeline
from .pipeline.train_pipeline import run_train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="business_entity_resolution")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Run the training pipeline on train/ data")
    train_p.add_argument("--nrows", type=int, default=None, help="Limit rows per source file (dev/debug)")
    train_p.add_argument("--s1-sample-size", type=int, default=None,
                          help="Training/validation-mode only: stratified-subsample this many S1 entities "
                               "(preserving country + match-cardinality distribution). S2/S3 are never "
                               "subsampled. Omit to train on the full S1 population.")
    train_p.add_argument("--s1-sample-seed", type=int, default=42)

    infer_p = sub.add_parser("predict", help="Run inference on test/ data using saved artifacts")
    infer_p.add_argument("--nrows", type=int, default=None)
    infer_p.add_argument("--matching-out", type=str, default=None)
    infer_p.add_argument("--candidate-out", type=str, default=None)

    full_p = sub.add_parser("full-pipeline", help="Run train then predict")
    full_p.add_argument("--nrows", type=int, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "train":
        report = run_train_pipeline(nrows=args.nrows, s1_sample_size=args.s1_sample_size, s1_sample_seed=args.s1_sample_seed)
        print(json.dumps({k: v for k, v in report.items() if k != "feature_importance"}, default=str, indent=2))
    elif args.command == "predict":
        matching_out = config.OUTPUT_DIR / "matching_results.tsv" if args.matching_out is None else args.matching_out
        candidate_out = config.OUTPUT_DIR / "candidate_pairs.tsv" if args.candidate_out is None else args.candidate_out
        report = run_inference_pipeline(nrows=args.nrows, matching_out=matching_out, candidate_out=candidate_out)
        print(json.dumps(report, default=str, indent=2))
    elif args.command == "full-pipeline":
        run_train_pipeline(nrows=args.nrows)
        report = run_inference_pipeline(nrows=args.nrows)
        print(json.dumps(report, default=str, indent=2))


if __name__ == "__main__":
    main()
