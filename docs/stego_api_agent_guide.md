# Zero-shot GLS API Agent Guide

Use `/capacity_probe` for fair research capacity. Use `/hide` and `/reveal` only for UTF-8 text roundtrips.

## Core Rule

The main capacity metric is raw payload bits recovered under the hard visible word limit:

```text
payload_bits_encoded
```

New `/hide` and `/capacity_probe` calls embed only those payload bits. The compatibility `header_bits` field is `0`, so:

```text
total_embedded_bits = payload_bits_encoded
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
  "payload_bits": "1011001110001111000011110000111100001111000011110000111100001111",
  "max_words": 40,
  "quality_max_retries": 6,
  "initial_payload_bits": 32,
  "payload_bits_candidates": [64, 32, 16, 8],
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
- `best_success.header_bits` is a compatibility field and should be `0`.
- `best_success.total_target_bits` equals the raw payload target.
- `payload_bits_candidates` are prefix lengths into `payload_bits`; omit them to try `initial_payload_bits`, then smaller halves and larger doublings within the supplied stream. If `initial_payload_bits` is omitted, the probe starts at the full stream length.
- `trials` contains failed and successful prefix lengths with `failure_reason`.

A trial is successful only if encoding, extraction, exact payload-bit match, word limit, and quality gate all pass.

## Hide/Reveal

`POST /hide` accepts the exact prompt/context string plus exactly one payload field:
UTF-8 `secret`, raw `payload_base64`, or raw `payload_bits`. Its bit accounting is explicit:

- `payload_bits`: UTF-8 payload bits
- `header_bits`: `0`
- `total_target_bits`: same as payload bits
- `total_used_bits`: raw payload bits embedded
- `remaining_bits`: exact unembedded suffix when generation stops early

For raw byte payloads, partial results are returned by default. If the remainder is not
byte-aligned, continue with `payload_bits=remaining_bits` rather than base64 bytes.

`POST /reveal` returns UTF-8 text and also exposes `payload_bits` and `payload_bits_len`. Pass the `/hide` response's `payload_bits` value back as `payload_bits_len` when revealing. If `payload_bits_len` is omitted, the server only tries the older 16-bit framed decode for legacy stegotext.

Use matching prompt and EGS parameters for reveal. Prefer passing `stego_token_ids` from `/hide`.

## Agent Checklist

- Use `/capacity_probe` for paper-style comparison.
- Keep `max_words` the same for both methods.
- Put the same post context and examples in each method's prompt.
- Record failed trials instead of discarding them.
- Compare by `payload_bits_encoded`; new outputs do not include hidden header overhead.
