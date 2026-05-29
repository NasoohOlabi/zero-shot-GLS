"""Generate a self-contained HTML trace of one ZGLS token-by-token hide run."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../src")

from llama_server import LlamaServerConfig, LlamaServerModel, LlamaServerTokenizer
from zgls_trace import generate_generation_trace, render_generation_trace_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:8081")
    parser.add_argument("--server-model", default="Qwen3.5-9B-Q4_K_M.gguf")
    parser.add_argument("--secret", default="A")
    parser.add_argument(
        "--prompt",
        default=(
            "/no_think\n"
            "Write exactly one short, natural movie-review sentence.\n"
            "Output only the sentence.\n"
            "Examples:\n"
            "The acting was uneven, but the ending still worked.\n"
            "I liked the premise more than the execution.\n"
            "New sentence:"
        ),
    )
    parser.add_argument("--output", default="tmp_runs/generation_trace_sample.html")
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-alpha", type=float, default=1.25)
    parser.add_argument("--repeat-penalty", type=float, default=4.0)
    parser.add_argument("--repeat-delta", type=float, default=0.5)
    parser.add_argument("--max-bpw", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--logprobs", type=int, default=256)
    parser.add_argument("--candidates-shown", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = LlamaServerModel(
        LlamaServerConfig(
            base_url=args.server_url,
            model=args.server_model,
            logprobs=args.logprobs,
            timeout_s=args.timeout,
            seed=args.seed,
        )
    ).eval()
    tokenizer = LlamaServerTokenizer(model)
    trace = generate_generation_trace(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        secret=args.secret,
        threshold=args.threshold,
        temperature=args.temperature,
        temperature_alpha=args.temperature_alpha,
        max_bpw=args.max_bpw,
        max_new_tokens=args.max_new_tokens,
        max_steps=args.max_steps,
        repeat_penalty=args.repeat_penalty,
        repeat_delta=args.repeat_delta,
        candidates_shown=None if args.candidates_shown <= 0 else args.candidates_shown,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_generation_trace_html(trace), encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
