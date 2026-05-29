"""Token-by-token HTML tracing for one ZGLS hide run."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import torch
from bitstring import BitStream

import codec
import search
from llama_server import LlamaServerTokenizer
from strategy import LogitsRepeatPenaltyStrategy, TemperatureAlphaStrategy
from zgls_api import payload_to_bits


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _token_text(tokenizer: LlamaServerTokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return f"<token {token_id}>"


def _bits_preview(bits: str, pos: int, width: int = 24) -> str:
    before = bits[max(0, pos - 8) : pos]
    after = bits[pos : pos + width]
    tail = "..." if pos + width < len(bits) else ""
    return f"{before}<span class='cursor'>|</span>{after}{tail}"


def generate_generation_trace(
    *,
    model: Any,
    tokenizer: LlamaServerTokenizer,
    prompt: str,
    secret: str,
    threshold: float,
    temperature: float,
    temperature_alpha: float,
    max_bpw: int,
    max_new_tokens: int,
    max_steps: int = 64,
    repeat_penalty: float = 4.0,
    repeat_delta: float = 0.5,
    candidates_shown: int | None = None,
) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = int(prompt_ids.size(1))
    bits = payload_to_bits(secret)
    bit_string = bits.bin
    bs = BitStream(bits)
    cur_ids = prompt_ids
    temp_strategy = TemperatureAlphaStrategy(temperature, temperature_alpha)
    logits_strategy = LogitsRepeatPenaltyStrategy(
        penalty=repeat_penalty,
        delta=repeat_delta,
        vocab_size=model.vocab_size,
        device=model.device,
    )

    steps: list[dict[str, Any]] = []
    nlls: list[float] = []
    stop_reason = "payload_exhausted"

    while bs.pos < len(bs):
        if len(steps) >= max_steps:
            stop_reason = "max_steps"
            break
        if cur_ids.size(1) - prompt_ids.size(1) >= max_new_tokens:
            stop_reason = "max_new_tokens"
            break

        pre_ids = cur_ids[0, prompt_len:].tolist()
        pretext = tokenizer.decode(pre_ids) if pre_ids else ""
        input_pos = int(bs.pos)
        temperature_before = float(temp_strategy.temperature)
        egs = search.enhanced_greedy_search(
            model,
            cur_ids,
            threshold=threshold,
            temperature=temperature_before,
            max_bits_len=max_bpw,
            logits_offset=logits_strategy.logits_offset,
        )
        new_ids = egs["comb_ids"]
        trunc_bits = int(egs["trunc_bits"].item())
        sorted_probs = egs["sorted_probs"]
        choice_count = int(new_ids.size(0))

        step: dict[str, Any] = {
            "index": len(steps) + 1,
            "pretext": pretext,
            "bit_pos_before": input_pos,
            "temperature": temperature_before,
            "choice_count": choice_count,
            "trunc_bits": trunc_bits,
            "candidates": [],
        }

        if choice_count == 1:
            chosen_id = int(new_ids[0, -1].item())
            cur_ids = new_ids
            temp_strategy.update(choice_count)
            logits_strategy.update(chosen_id)
            chosen_text = _token_text(tokenizer, chosen_id)
            step.update(
                {
                    "mode": "forced",
                    "why": "Only one candidate survived pruning, so no payload bit was consumed.",
                    "bits_consumed": "",
                    "bit_pos_after": int(bs.pos),
                    "chosen_idx": 0,
                    "chosen_id": chosen_id,
                    "chosen_text": chosen_text,
                    "temperature_after": float(temp_strategy.temperature),
                }
            )
            step["candidates"].append(
                {
                    "rank": 0,
                    "token_id": chosen_id,
                    "text": chosen_text,
                    "prob": float(sorted_probs[0].item()),
                    "code": "",
                    "selected": True,
                }
            )
            steps.append(step)
            continue

        probs_list = [float(p) for p in sorted_probs.tolist()]
        idx2probs = {idx: prob for idx, prob in enumerate(probs_list)}
        idx2code = codec.huffman.from_frequencies(idx2probs)
        code2idx = {code: idx for idx, code in idx2code.items()}
        max_code_len = max(len(code) for code in code2idx)

        tmp_bits = ""
        padded = False
        while bs.pos < len(bs):
            tmp_bits += bs.read("bin:1")
            if tmp_bits in code2idx:
                break
        else:
            while len(tmp_bits) < max_code_len and tmp_bits not in code2idx:
                tmp_bits += "0"
                padded = True
            if tmp_bits not in code2idx:
                raise RuntimeError(f"Could not match Huffman code for prefix {tmp_bits!r}")

        chosen_idx = int(code2idx[tmp_bits])
        chosen_id = int(new_ids[chosen_idx, -1].item())
        chosen_prob = float(sorted_probs[chosen_idx].item())
        cur_ids = new_ids[chosen_idx].unsqueeze(0)
        nlls.append(float(-torch.log2(sorted_probs[chosen_idx]).item()))
        temp_strategy.update()
        logits_strategy.update(chosen_id)

        shown_count = choice_count if candidates_shown is None else min(choice_count, candidates_shown)
        for idx, prob in enumerate(probs_list[:shown_count]):
            tid = int(new_ids[idx, -1].item())
            step["candidates"].append(
                {
                    "rank": idx,
                    "token_id": tid,
                    "text": _token_text(tokenizer, tid),
                    "prob": prob,
                    "code": idx2code[idx],
                    "selected": idx == chosen_idx,
                }
            )

        step.update(
            {
                "mode": "huffman",
                "why": (
                    f"The next payload prefix was {tmp_bits!r}, which maps to candidate "
                    f"rank {chosen_idx} in the Huffman table."
                ),
                "bits_consumed": tmp_bits,
                "padded": padded,
                "bit_pos_after": int(bs.pos),
                "chosen_idx": chosen_idx,
                "chosen_id": chosen_id,
                "chosen_text": _token_text(tokenizer, chosen_id),
                "chosen_prob": chosen_prob,
                "temperature_after": float(temp_strategy.temperature),
            }
        )
        steps.append(step)

    generated_ids = cur_ids[0, prompt_len:].tolist()
    return {
        "generated_text": tokenizer.decode(generated_ids) if generated_ids else "",
        "generated_token_ids": generated_ids,
        "payload_bits": bit_string,
        "payload_bit_len": len(bit_string),
        "secret": secret,
        "prompt": prompt,
        "params": {
            "threshold": threshold,
            "temperature": temperature,
            "temperature_alpha": temperature_alpha,
            "max_bpw": max_bpw,
            "max_new_tokens": max_new_tokens,
            "max_steps": max_steps,
            "repeat_penalty": repeat_penalty,
            "repeat_delta": repeat_delta,
            "candidates_shown": candidates_shown,
        },
        "steps": steps,
        "stop_reason": stop_reason,
        "used_bits": int(bs.pos),
        "ppl": 2 ** (sum(nlls) / len(nlls)) if nlls else 1.0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def render_generation_trace_html(trace: dict[str, Any]) -> str:
    rows = []
    for step in trace["steps"]:
        candidate_rows = []
        for cand in step["candidates"]:
            cls = " class='selected'" if cand["selected"] else ""
            candidate_rows.append(
                "<tr{cls}><td>{rank}</td><td>{tid}</td><td><code>{txt}</code></td>"
                "<td>{prob:.6f}</td><td><code>{code}</code></td><td>{sel}</td></tr>".format(
                    cls=cls,
                    rank=cand["rank"],
                    tid=cand["token_id"],
                    txt=_e(repr(cand["text"])),
                    prob=cand["prob"],
                    code=_e(cand["code"]),
                    sel="picked" if cand["selected"] else "",
                )
            )
        rows.append(
            f"""
            <section class="step">
              <h2>Step {step['index']}: picked <code>{_e(repr(step['chosen_text']))}</code></h2>
              <div class="grid">
                <div>
                  <h3>Pretext Before Selection</h3>
                  <pre>{_e(step['pretext']) or '<empty>'}</pre>
                </div>
                <div>
                  <h3>Payload Cursor</h3>
                  <p class="bits"><code>{_bits_preview(trace['payload_bits'], step['bit_pos_before'])}</code></p>
                  <p>Consumed: <code>{_e(step['bits_consumed']) or '(none)'}</code></p>
                  <p>Bit position: {step['bit_pos_before']} -> {step['bit_pos_after']}</p>
                </div>
              </div>
              <p><strong>Why:</strong> {_e(step['why'])}</p>
              <p class="meta">mode={_e(step['mode'])} · temp={step['temperature']:.3f} -> {step['temperature_after']:.3f} · candidates={step['choice_count']} · trunc_bits={step['trunc_bits']}</p>
              <table>
                <thead><tr><th>Rank</th><th>Token ID</th><th>Token Text</th><th>Probability</th><th>Huffman Code</th><th></th></tr></thead>
                <tbody>{''.join(candidate_rows)}</tbody>
              </table>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZGLS Generation Trace</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1d2329; background: #f6f7f9; }}
    header {{ padding: 28px 36px; background: #17202a; color: white; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; }}
    h3 {{ font-size: 14px; color: #4c5967; text-transform: uppercase; }}
    .summary, .step {{ background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 20px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, .7fr); gap: 18px; }}
    pre {{ white-space: pre-wrap; background: #f1f3f6; border-radius: 6px; padding: 12px; min-height: 42px; }}
    code {{ font-family: Consolas, monospace; }}
    .bits code {{ font-size: 15px; word-break: break-all; }}
    .cursor {{ color: #d12b2b; font-weight: 700; }}
    .meta {{ color: #5f6b78; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e9ef; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f3f6; color: #344050; }}
    tr.selected {{ background: #fff4ca; }}
    .final {{ font-size: 18px; }}
  </style>
</head>
<body>
  <header>
    <h1>ZGLS Token Selection Trace</h1>
    <p>Generated {_e(trace['created_at'])}. Every section shows the state before one token is selected.</p>
  </header>
  <main>
    <section class="summary">
      <h2>Run Summary</h2>
      <p><strong>Secret:</strong> <code>{_e(repr(trace['secret']))}</code></p>
      <p><strong>Payload bits:</strong> <code>{_e(trace['payload_bits'])}</code> ({trace['payload_bit_len']} bits, no header)</p>
      <p><strong>Used bits:</strong> {trace['used_bits']} · <strong>Steps:</strong> {len(trace['steps'])} · <strong>PPL estimate:</strong> {trace['ppl']:.4f} · <strong>Stop:</strong> {_e(trace['stop_reason'])}</p>
      <p><strong>Final stegotext:</strong></p>
      <pre class="final">{_e(trace['generated_text'])}</pre>
      <p><strong>Prompt:</strong></p>
      <pre>{_e(trace['prompt'])}</pre>
    </section>
    {''.join(rows)}
  </main>
</body>
</html>
"""
