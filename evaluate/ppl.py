"""Compute confidence-aware PPL summaries from a CSV column."""

from __future__ import annotations

import argparse
import json
import math
import os.path as osp
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from zgls_api import bootstrap_ci


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute PPL summary statistics with bootstrap confidence intervals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Path to the input .csv data file.")
    parser.add_argument("--ppl-col", type=str, default="ppl")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Drop missing/non-numeric PPL values instead of failing.",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output .json file.")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output file.")
    args = parser.parse_args()
    if not osp.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist.")
    if not args.force and osp.exists(args.output):
        raise FileExistsError(f"{args.output} already exists. Use --force to overwrite.")
    return args


def load_ppl_values(path: str, col: str, *, drop_invalid: bool) -> list[float]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"Data column '{col}' not in {path}.")
    parsed = pd.to_numeric(df[col], errors="coerce")
    invalid_mask = parsed.isna() | ~parsed.map(lambda x: math.isfinite(float(x)) if pd.notna(x) else False)
    invalid_count = int(invalid_mask.sum())
    if invalid_count and not drop_invalid:
        raise ValueError(
            f"Found {invalid_count} missing/non-numeric PPL values in '{col}'. "
            "Use --drop-invalid to ignore them."
        )
    values = [float(v) for v in parsed[~invalid_mask].tolist()]
    if not values:
        raise ValueError(f"No valid PPL values found in '{col}'.")
    return values


def main() -> None:
    args = parse_args()
    values = load_ppl_values(args.input, args.ppl_col, drop_invalid=args.drop_invalid)
    summary = bootstrap_ci(
        values,
        iters=args.bootstrap_iters,
        confidence=args.confidence,
        seed=args.seed,
    )
    print(f"Average PPL: {summary['mean']:.4f}")
    print(f"{int(summary['count'])} samples, {args.confidence:.1%} CI: "
          f"[{summary['ci_low']:.4f}, {summary['ci_high']:.4f}]")

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "ppl_mean": summary["mean"],
                "ppl_median": summary["median"],
                "ppl_stddev": summary["stddev"],
                "ppl_min": summary["min"],
                "ppl_max": summary["max"],
                "ppl_ci_low": summary["ci_low"],
                "ppl_ci_high": summary["ci_high"],
                "ppl_ci_half_width": summary["ci_half_width"],
                "ppl_relative_ci_half_width": summary["relative_ci_half_width"],
                "sample_count": summary["count"],
                "confidence": args.confidence,
                "bootstrap_iters": args.bootstrap_iters,
            },
            fp,
            indent=2,
        )


if __name__ == "__main__":
    main()
