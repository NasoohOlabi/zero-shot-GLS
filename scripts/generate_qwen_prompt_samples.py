"""Generate Qwen-only prompt sample artifacts and summary for the HTML report.

Outputs:
  - tmp_saves/runs/qwen_qwen3.5_9b_prompt_samples.json
  - tmp_saves/runs/prompt_sample_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os.path as osp
import random
import re
import statistics
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "tmp_saves" / "runs"
DATASET = ROOT / "datasets" / "imdb" / "imdb.csv"
PROMPTS_CFG = ROOT / "config" / "workflow_llm_prompts.json"
OUT_JSON = RUNS / "qwen_qwen3.5_9b_prompt_samples.json"
OUT_SUMMARY = RUNS / "prompt_sample_summary.csv"

WORD_RE = re.compile(r"[A-Za-z0-9']+")
TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--server-url", type=str, default="http://127.0.0.1:8081")
    p.add_argument("--server-model", type=str, default="Qwen3.5-9B-Q4_K_M.gguf")
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--n-cover", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.9)
    return p.parse_args()


def read_cover_rows(path: Path, col: str = "plaintext") -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if col not in (reader.fieldnames or []):
            raise ValueError(f"cover column '{col}' not found in {path}")
        for row in reader:
            text = (row.get(col) or "").strip()
            if text:
                rows.append(text)
    if not rows:
        raise ValueError(f"no usable rows in {path}")
    return rows


def load_prompt_config(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8-sig") as fp:
        obj = json.load(fp)
    if "stego_encode" not in obj:
        raise KeyError("Missing key: stego_encode")
    stego = obj["stego_encode"]
    if "system_template" not in stego:
        raise KeyError("Missing key: stego_encode.system_template")
    if "user_template" not in stego:
        raise KeyError("Missing key: stego_encode.user_template")
    return str(stego["system_template"]), str(stego["user_template"])


def old_qwen_prompt(context: str, corpus: str = "IMDB about movies") -> str:
    return (
        "/no_think\n"
        "Write exactly one new sentence in the same style as the examples below.\n"
        "Output only the sentence.\n"
        "Do not use tags, XML, markdown, labels, or explanations.\n"
        "Do not repeat the prompt.\n"
        "Use plain ASCII only.\n"
        f"Corpus: {corpus}\n"
        "Examples:\n"
        f"{context}\n"
        "New sentence:\n"
    )


def new_config_prompt(system_template: str, user_template: str, context: str) -> str:
    fmt = {
        "tangent": "mood",
        "category": "movies",
        "target_category": "movies",
        "target_tangent": "mood",
        "target_source_quote": "",
        "best_match": "",
        "title": "Movie discussion",
        "author": "reddit_user",
        "selftext": context,
        "chain_section": "",
    }
    return system_template.format(**fmt) + "\n\n" + user_template.format(**fmt)


def completion_text(
    server_url: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    body = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }
    r = requests.post(
        server_url.rstrip("/") + "/v1/completions",
        json=body,
        timeout=120,
    )
    r.raise_for_status()
    j = r.json()
    choice0 = (j.get("choices") or [{}])[0]
    return (choice0.get("text") or "").strip()


def metrics(text: str) -> dict[str, float | int | bool]:
    words = WORD_RE.findall(text.lower())
    if not words:
        return {
            "word_count": 0,
            "repetition_ratio": 1.0,
            "single_token_share": 1.0,
            "tag_hits": 0,
            "ascii_ok": text.isascii(),
        }
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    repetition_ratio = 1.0 - (len(counts) / len(words))
    single_token_share = max(counts.values()) / len(words)
    return {
        "word_count": len(words),
        "repetition_ratio": repetition_ratio,
        "single_token_share": single_token_share,
        "tag_hits": len(TAG_RE.findall(text)),
        "ascii_ok": text.isascii(),
    }


def mean_metric(samples: list[dict], key: str, new: bool) -> float:
    mk = "new_metrics" if new else "old_metrics"
    vals = []
    for s in samples:
        v = s[mk][key]
        vals.append(float(v) if isinstance(v, bool) else v)
    return statistics.fmean(vals) if vals else 0.0


def main() -> None:
    args = parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)

    cover_rows = read_cover_rows(DATASET)
    system_template, user_template = load_prompt_config(PROMPTS_CFG)

    rng = random.Random(args.seed)
    samples: list[dict] = []
    for trial in range(args.trials):
        context_rows = rng.sample(cover_rows, k=max(1, min(args.n_cover, len(cover_rows))))
        context = "\n\n".join(context_rows)
        old_prompt = old_qwen_prompt(context)
        new_prompt = new_config_prompt(system_template, user_template, context)
        old_text = completion_text(
            args.server_url, args.server_model, old_prompt, args.max_new_tokens, args.temperature
        )
        new_text = completion_text(
            args.server_url, args.server_model, new_prompt, args.max_new_tokens, args.temperature
        )
        samples.append(
            {
                "trial": trial,
                "context": context,
                "old_text": old_text,
                "new_text": new_text,
                "old_metrics": metrics(old_text),
                "new_metrics": metrics(new_text),
            }
        )

    payload = {
        "model": "qwen/qwen3.5-9b",
        "samples": samples,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "model": "qwen/qwen3.5-9b",
        "old_avg_tag_hits": f"{mean_metric(samples, 'tag_hits', new=False):.4f}",
        "new_avg_tag_hits": f"{mean_metric(samples, 'tag_hits', new=True):.4f}",
        "old_avg_word_count": f"{mean_metric(samples, 'word_count', new=False):.4f}",
        "new_avg_word_count": f"{mean_metric(samples, 'word_count', new=True):.4f}",
        "old_avg_repetition_ratio": f"{mean_metric(samples, 'repetition_ratio', new=False):.4f}",
        "new_avg_repetition_ratio": f"{mean_metric(samples, 'repetition_ratio', new=True):.4f}",
        "old_ascii_ok_rate": f"{mean_metric(samples, 'ascii_ok', new=False):.4f}",
        "new_ascii_ok_rate": f"{mean_metric(samples, 'ascii_ok', new=True):.4f}",
    }
    with OUT_SUMMARY.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(OUT_JSON)
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()
