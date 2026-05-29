"""Compute smoothed KLD between stego text and cover/plain text distributions."""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from llama_server import LlamaServerConfig, LlamaServerModel, LlamaServerTokenizer
from zgls_api import bootstrap_kld_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute smoothed KLD between stego and cover/plain text columns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stego_input", type=str, help="CSV containing stego text.")
    parser.add_argument("cover_input", type=str, help="CSV containing cover/plain baseline text.")
    parser.add_argument("--stego-col", type=str, default="stegotext")
    parser.add_argument("--cover-col", type=str, default="plaintext")
    parser.add_argument("--alpha", type=float, default=1e-6, help="Additive smoothing.")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--server-model", type=str, default=None)
    parser.add_argument(
        "--word-tokenizer",
        action="store_true",
        help="Use local word-token KLD even if server tokenizer args are provided.",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output .json file.")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output file.")
    args = parser.parse_args()
    for path in (args.stego_input, args.cover_input):
        if not osp.exists(path):
            raise FileNotFoundError(f"Input file {path} does not exist.")
    if not args.force and osp.exists(args.output):
        raise FileExistsError(f"{args.output} already exists. Use --force to overwrite.")
    return args


def read_texts(path: str, col: str) -> list[str]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"Data column '{col}' not in {path}.")
    return [str(v).strip() for v in df[col].dropna().tolist() if str(v).strip()]


def maybe_tokenizer(args: argparse.Namespace):
    if args.word_tokenizer or not (args.server_url and args.server_model):
        return None
    model = LlamaServerModel(
        LlamaServerConfig(base_url=args.server_url, model=args.server_model, seed=args.seed)
    )
    return LlamaServerTokenizer(model)


def main() -> None:
    args = parse_args()
    stego_texts = read_texts(args.stego_input, args.stego_col)
    cover_texts = read_texts(args.cover_input, args.cover_col)
    if not stego_texts:
        raise ValueError(f"No stego texts found in {args.stego_input}:{args.stego_col}")
    if not cover_texts:
        raise ValueError(f"No cover texts found in {args.cover_input}:{args.cover_col}")

    summary = bootstrap_kld_ci(
        stego_texts,
        cover_texts,
        tokenizer=maybe_tokenizer(args),
        alpha=args.alpha,
        iters=args.bootstrap_iters,
        confidence=args.confidence,
        seed=args.seed,
    )
    print(f"KLD symmetric: {summary['kld_symmetric']:.6f}")
    print(f"{int(summary['count'])} stego samples, {args.confidence:.1%} CI: "
          f"[{summary['ci_low']:.6f}, {summary['ci_high']:.6f}]")

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(
            {
                **summary,
                "confidence": args.confidence,
                "bootstrap_iters": args.bootstrap_iters,
                "alpha": args.alpha,
                "tokenizer": "word" if args.word_tokenizer or not args.server_url else "llama_server",
            },
            fp,
            indent=2,
        )


if __name__ == "__main__":
    main()
