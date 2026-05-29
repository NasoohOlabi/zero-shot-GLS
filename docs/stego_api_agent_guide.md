# Zero-shot GLS API Agent Guide

Use `/capacity_probe` for fair research capacity. Use `/hide` and `/reveal` only for UTF-8 text roundtrips.

## Core Rule

The main capacity metric is raw payload bits recovered under the hard visible word limit:

```text
payload_bits_encoded
```

Do not report framed totals as capacity. ZGLS uses a 16-bit payload-bit-length header, so:

```text
total_embedded_bits = payload_bits_encoded + protocol_overhead_bits
```

## Base URL

```text
http://127.0.0.1:9000
```

## Prompt Rule

`prompt` is the complete model context. Include the real post context, comment examples, style constraints, and output instruction directly in `prompt`.

Do not assume the server samples or retrieves examples.

## Capacity Probe

Call `POST /capacity_probe`.

```json
{
  "prompt": "complete prompt with examples",
  "max_words": 40,
  "quality_max_retries": 6,
  "payload_bits_candidates": [1, 2, 4, 8, 12, 16, 24, 32, 48, 64],
  "complete_sent": true,
  "max_new_tokens": 64,
  "threshold": 0.005,
  "temperature": 1.0,
  "temperature_alpha": 1.25,
  "max_bpw": 2
}
```

Interpretation:

- `best_success.payload_bits_exact` is the capacity score.
- `best_success.header_bits` is overhead.
- `best_success.total_target_bits` is diagnostic only.
- `trials` contains failed and successful candidates with `failure_reason`.

A trial is successful only if encoding, extraction, exact payload-bit match, word limit, and quality gate all pass.

## Legacy Hide/Reveal

`POST /hide` accepts UTF-8 `secret` text. Its bit accounting is explicit:

- `payload_bits`: UTF-8 payload bits
- `header_bits`: 16
- `total_target_bits`: payload plus header
- `total_used_bits`: framed bits embedded

`POST /reveal` returns UTF-8 text and also exposes `payload_bits` and `payload_bits_len`.

Use matching prompt and EGS parameters for reveal. Prefer passing `stego_token_ids` from `/hide`.

## Agent Checklist

- Use `/capacity_probe` for paper-style comparison.
- Keep `max_words` the same for both methods.
- Put the same post context and examples in each method's prompt.
- Record failed trials instead of discarding them.
- Compare by `payload_bits_encoded`, not `total_embedded_bits`.
