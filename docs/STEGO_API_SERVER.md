# Stego API Server

This server exposes the ZGLS hide/reveal API plus the research capacity probe used for fair short-comment comparisons.

Source files:

- `scripts/stego_api_server.py`
- `src/zgls_api.py`

## Research Capacity Contract

The canonical capacity metric is:

```text
payload_bits_successfully_recovered_under_hard_visible_word_limit
```

New API outputs do not embed protocol framing overhead:

- `payload_bits = raw secret payload bits`
- `header_bits = 0`
- `total_target_bits = payload_bits`

Report `payload_bits` as the main capacity number. `total_target_bits` and `total_used_bits` are diagnostics for the same raw payload stream.

The visible word limit is hard. The server no longer silently expands `quality_max_words` for larger payloads.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\stego_api_server.py `
  --host 127.0.0.1 `
  --port 9000 `
  --backend llama_server `
  --server-url http://127.0.0.1:8081 `
  --server-model Qwen3.5-9B-Q4_K_M.gguf `
  --quality-min-words 8 `
  --quality-max-words 40 `
  --quality-max-retries 4 `
  --egs-mode huffman `
  --threshold 0.005 `
  --temperature 1.0 `
  --temperature-alpha 1.25 `
  --max-bpw 2 `
  --max-new-tokens 256
```

The request `prompt` is the complete model context. Put post context, examples, style instructions, and output constraints directly in the prompt.

## `POST /capacity_probe`

Use this endpoint for fair benchmark capacity.

Example request:

```json
{
  "prompt": "You are writing natural short movie comments.\n\nExamples:\n- The acting was strong but the pacing felt uneven.\n- It had a few nice scenes, though the ending felt rushed.\n\nWrite one new natural short movie comment.",
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

Success requires all of:

- raw payload fully embedded
- generated text has `word_count <= max_words`
- quality gate passes
- extraction succeeds
- recovered raw payload bits exactly match the candidate payload bits

Important response fields:

- `best_success`: largest successful trial by `payload_bits_exact`
- `trials`: one structured result per candidate
- `payload_bits_exact`: main capacity metric
- `header_bits`: compatibility field, currently `0`
- `total_target_bits`: same as `payload_bits_exact`
- `total_used_bits`: raw payload bits actually embedded
- `failure_reason`: `truncated`, `insufficient_used_bits`, `word_limit_exceeded`, `quality_gate_failed`, `decode_failed`, or `payload_mismatch`

## `POST /hide`

Encodes caller-supplied payload into next-token generation from the exact prompt you send.
Put examples, context, and instructions directly in `prompt`; the server does not own or
rewrite the prompt.

Provide exactly one payload field:

- `secret`: UTF-8 text, kept for compatibility
- `payload_base64`: raw payload bytes encoded as base64
- `payload_bits`: raw bits, useful for continuing a non-byte-aligned remainder

For raw payload requests, partial output is returned by default when the token budget ends
before the full payload is embedded. Continue with `remaining_bits` if `remaining_bits_len`
is non-zero. `remaining_payload_base64` is present only when the remainder is byte-aligned.

Important response accounting:

- `payload_bits`: UTF-8 payload bits
- `header_bits`: compatibility field, currently `0`
- `target_bits` / `total_target_bits`: raw payload target bits
- `used_bits` / `total_used_bits`: raw payload bits embedded by the encoder
- `embedded_bits`: useful payload bits embedded, capped at the target length
- `fully_embedded`: true when no payload remains
- `remaining_bits` / `remaining_bits_len`: exact unembedded payload suffix
- `quality_max_words`: hard accepted word ceiling

## `POST /reveal`

Decodes a UTF-8 secret from stegotext. Prefer the token-stable path by passing `stego_token_ids` from `/hide`, and pass `/hide`'s `payload_bits` value as `payload_bits_len`.

If `payload_bits_len` is omitted, the server tries the old 16-bit framed decode for legacy stegotext only.

The response includes:

- `secret`
- `payload_bytes`
- `payload_bits`
- `payload_bits_len`
- `decode_ok`
- `raw_bits_len`
- `warnings`

For non-byte-aligned research probes, use `/capacity_probe`; `/reveal` is intentionally text-oriented.

## Required Benchmark Columns

Use these columns for fair reports:

- `post_id`
- `sample_index`
- `method`
- `stegotext`
- `word_count`
- `payload_bits_encoded`
- `protocol_overhead_bits`
- `total_embedded_bits`
- `decode_ok`
- `quality_passed`
- `perplexity_gpt2`
- `kl_global_corpus`
- `jsd_global_corpus`
- `repetition_ratio`

Main metric:

```text
payload_bits_encoded
```

Diagnostic:

```text
total_embedded_bits = payload_bits_encoded + protocol_overhead_bits
```

## Reproducibility

Record these settings with every probe:

- prompt text
- `max_words`
- `payload_bits_candidates`
- `payload_seed`
- `max_new_tokens`
- `quality_max_retries`
- EGS `mode`, `threshold`, `temperature`, `temperature_alpha`, `max_bpw`
- model/backend id

Changing model, tokenizer, prompt, or EGS parameters can change extraction results.
