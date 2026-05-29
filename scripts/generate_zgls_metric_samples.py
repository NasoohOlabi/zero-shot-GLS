"""Generate ZGLS samples until PPL/KLD confidence criteria are met."""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../src")

from zgls_api import EGSParams, SampleConfig, ZGLSClient, ZGLSConfig


DEFAULT_PROMPTS = [
    "Write one natural short sentence about a movie.",
    "Write a casual short reply about how the scene felt.",
    "Write a grounded Reddit-style comment about a mixed reaction.",
]
DEFAULT_SECRETS = ["a", "hello", "edge-case", "unicode: cafe"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--server-url", type=str, default="http://127.0.0.1:8081")
    p.add_argument("--server-model", type=str, default="Qwen3.5-9B-Q4_K_M.gguf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cover-path", type=str, default=None)
    p.add_argument("--cover-col", type=str, default="plaintext")
    p.add_argument("--prompt-cfg-path", type=str, default=None)
    p.add_argument("--corpus", type=str, default="Unknown")
    p.add_argument("--n-cover", type=int, default=2)
    p.add_argument("--prompt", action="append", dest="prompts", default=None)
    p.add_argument("--prompts-file", type=str, default=None, help="Text file with one prompt per line.")
    p.add_argument("--secret", action="append", dest="secrets", default=None)
    p.add_argument("--secrets-file", type=str, default=None, help="Text file with one secret per line.")
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--relative-ci-target", type=float, default=0.05)
    p.add_argument("--kld-cover-limit", type=int, default=1000)
    p.add_argument("--quality-max-retries", type=int, default=4)
    p.add_argument("--quality-max-repetition-ratio", type=float, default=0.65)
    p.add_argument("--quality-max-single-token-share", type=float, default=0.35)
    p.add_argument("--threshold", type=float, default=1e-2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--temperature-alpha", type=float, default=1.0)
    p.add_argument("--max-bpw", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--egs-mode", type=str, default="huffman", choices=["block", "huffman"])
    p.add_argument("--complete-sent", action="store_true")
    p.add_argument("--output-dir", type=str, default="tmp_saves/runs/zgls_metric_samples")
    p.add_argument("-f", "--force", action="store_true")
    return p.parse_args()


def read_lines(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def write_outputs(output_dir: Path, samples: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "samples.jsonl"
    csv_path = output_dir / "samples.csv"
    summary_path = output_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in samples:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = sorted({key for row in samples for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in samples:
            writer.writerow(flatten_for_csv(row))

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print(jsonl_path)
    print(csv_path)
    print(summary_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} already has files. Use --force to overwrite/add.")
    progress_path = output_dir / "progress.jsonl"
    if args.force and progress_path.exists():
        progress_path.unlink()

    prompts = (args.prompts or []) + read_lines(args.prompts_file)
    secrets = (args.secrets or []) + read_lines(args.secrets_file)
    if not prompts:
        prompts = DEFAULT_PROMPTS
    if not secrets:
        secrets = DEFAULT_SECRETS

    cfg = ZGLSConfig(
        server_url=args.server_url,
        server_model=args.server_model,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        cover_path=args.cover_path,
        cover_col=args.cover_col,
        prompt_cfg_path=args.prompt_cfg_path or ZGLSConfig.prompt_cfg_path,
        corpus=args.corpus,
        n_cover=args.n_cover,
        quality_max_retries=args.quality_max_retries,
        quality_max_repetition_ratio=args.quality_max_repetition_ratio,
        quality_max_single_token_share=args.quality_max_single_token_share,
        egs=EGSParams(
            mode=args.egs_mode,
            threshold=args.threshold,
            temperature=args.temperature,
            temperature_alpha=args.temperature_alpha,
            max_bpw=args.max_bpw,
        ),
    )
    client = ZGLSClient(cfg)
    run = client.generate_samples(
        SampleConfig(
            prompts=prompts,
            secrets=secrets,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
            bootstrap_iters=args.bootstrap_iters,
            confidence=args.confidence,
            relative_ci_target=args.relative_ci_target,
            seed=args.seed,
            complete_sent=args.complete_sent,
            corpus=args.corpus,
            kld_cover_limit=args.kld_cover_limit,
            progress_path=str(progress_path),
        )
    )
    summary = {
        "config": {
            "server_url": args.server_url,
            "server_model": args.server_model,
            "seed": args.seed,
            "min_samples": args.min_samples,
            "max_samples": args.max_samples,
            "bootstrap_iters": args.bootstrap_iters,
            "confidence": args.confidence,
            "relative_ci_target": args.relative_ci_target,
            "kld_cover_limit": args.kld_cover_limit,
            "egs_mode": args.egs_mode,
        },
        "metrics": asdict(run.metrics),
    }
    write_outputs(output_dir, run.samples, summary)


if __name__ == "__main__":
    main()
