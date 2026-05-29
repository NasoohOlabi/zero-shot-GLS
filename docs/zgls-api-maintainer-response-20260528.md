# Response To ZLG Endpoint Capacity Spec

Status: implemented and superseding the earlier byte-oriented `/hide` interpretation.

The spec is correct: fair comparison should use payload bits recovered under the same hard visible comment-length limit. ZLG framed totals are useful diagnostics, but they are not the headline capacity metric because the frame includes protocol overhead.

The implemented source of truth is:

```text
POST /capacity_probe
```

The main benchmark metric is:

```text
payload_bits_encoded
```

The diagnostic framed total is:

```text
total_embedded_bits = payload_bits_encoded + protocol_overhead_bits
```

For the current ZLG API, `protocol_overhead_bits` is `16`.

## Implemented Contract

`/capacity_probe` accepts arbitrary candidate payload bit lengths, including non-byte-aligned lengths such as `1`, `11`, `12`, and `13` bits. Each candidate is converted into a framed ZLG payload with:

- `header_bits = 16`
- `payload_bits = candidate raw payload bits`
- `total_target_bits = header_bits + payload_bits`

The 16-bit header stores the payload bit length, not the payload byte length. This removes the old ambiguity where a UTF-8 text secret made it look as though byte-oriented payload capacity and framed ZLG capacity were the same quantity.

A trial is successful only when all of these are true:

- the full framed payload is embedded
- the generated visible text has `word_count <= max_words`
- the quality gate passes
- reveal/extraction succeeds
- the recovered payload bit string exactly equals the candidate payload bit string

`best_success` is the largest successful candidate by exact payload-bit length. Failed trials remain in `trials` with structured `failure_reason` values so reports can distinguish capacity failure from quality failure, truncation, decode failure, or payload mismatch.

## Endpoint Choice

The implemented design is Option A from the spec: add `/capacity_probe`.

`/hide` and `/reveal` remain operational as UTF-8 convenience endpoints, but they are not the benchmark interface. `/hide` now labels its accounting explicitly:

- `payload_bits`: UTF-8 payload bits, excluding the header
- `header_bits`: protocol overhead, currently `16`
- `target_bits` / `total_target_bits`: framed total, including header
- `used_bits` / `total_used_bits`: framed bits actually embedded

The server no longer performs silent adaptive expansion of the word limit. `quality_max_words` is an actual maximum accepted word count, and `/capacity_probe.max_words` is the hard visible budget for the benchmark.

## Answers To The Spec Questions

1. Does `used_bits` include the 16-bit header?

Yes. `used_bits` and `total_used_bits` are framed totals. They include the 16-bit header whenever the header was embedded.

2. Does `target_bits` always equal `header_bits + payload_bits`, or can it include other codec overhead?

For the current implementation, yes: `target_bits == total_target_bits == header_bits + payload_bits`. The only protocol overhead currently reported by the API is the 16-bit payload-length header.

3. Can the API accept raw bit payloads instead of UTF-8 text payloads?

Yes, through `/capacity_probe`. It generates deterministic raw bit payloads for each requested candidate length. This is the correct interface for paper-style capacity measurement.

4. If not, can it accept a `payload_bit_length` so we can test non-byte-aligned payloads like 11, 12, or 13 bits?

Yes. `/capacity_probe.payload_bits_candidates` directly accepts candidate bit lengths, including non-byte-aligned values. The implementation has tests for `1`, `11`, `12`, and `13` bit framing roundtrips.

5. Can the API enforce a hard visible length limit in words, not just quality retry filtering?

Yes. `/capacity_probe.max_words` is a hard acceptance limit. A trial with `word_count > max_words` fails with `failure_reason = "word_limit_exceeded"` even if extraction would otherwise work.

6. Can the API return the best valid payload size under a word limit automatically?

Yes. `/capacity_probe.best_success` returns the largest successful payload candidate under the configured word limit. If no candidate succeeds, `best_success` is `null` and the failed trials remain available for diagnosis.

7. For failed quality-gate responses, can it return decode-ready candidates as structured data without treating them as successful?

Yes. Failed trials are preserved in `trials`. A decode-ready candidate that fails the quality gate is still marked unsuccessful with `quality_passed = false` and a failure reason, rather than being counted as capacity.

8. Can `/reveal` report payload bits recovered separately from raw extracted bits?

Yes for UTF-8 `/reveal`. It reports `payload_bits`, `payload_bits_len`, and `raw_bits_len`. For non-byte-aligned payload probes, use `/capacity_probe`, because `/reveal` intentionally returns UTF-8 text and therefore requires byte-aligned payload bits.

9. What is the exact minimum overhead for any ZLG payload?

The current minimum protocol overhead is `16` bits. This is the payload-bit-length header.

10. Is `max_bpw` a hard cap on payload bits per generated token, or only a search/generation constraint?

`max_bpw` is an EGS search/generation constraint, not the benchmark capacity metric. The benchmark result must still be determined by exact recovered payload bits under the hard visible word limit.

## Reporting Columns

Use the report columns from the spec:

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

For ZLG rows, fill them from `/capacity_probe.best_success`:

- `payload_bits_encoded = best_success.payload_bits_exact`
- `protocol_overhead_bits = best_success.header_bits`
- `total_embedded_bits = best_success.total_used_bits`
- `decode_ok = best_success.decode_ok`
- `quality_passed = best_success.quality_passed`
- `word_count = best_success.word_count`
- `stegotext = best_success.stegotext`

If no candidate succeeds, set `payload_bits_encoded = 0`, keep the failed trial rows for audit, and do not substitute `total_target_bits` or `total_used_bits` as capacity.

## Files Updated

- `src/zgls_api.py`: bit-length framing, raw bit helpers, hard word-limit behavior, and `ZGLSClient.capacity_probe(...)`
- `scripts/stego_api_server.py`: `/capacity_probe` models, endpoint, help text, OpenAPI schema, and explicit `/hide` accounting
- `docs/STEGO_API_SERVER.md`: rewritten around the capacity-probe contract
- `docs/stego_api_agent_guide.md`: rewritten for benchmark agents
- `tests/test_metrics.py`: framing, accounting, OpenAPI, and capacity-selection tests

## Verification

The local test suite for the ZLG API contract passes:

```text
16 passed, 6 skipped
```

The skipped tests are optional real-backend tests gated on backend environment variables.
