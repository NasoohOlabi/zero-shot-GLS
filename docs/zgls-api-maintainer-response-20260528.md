# ZGLS API Maintainer Response

This note supersedes the earlier framed-accounting response. New `/hide` and `/capacity_probe` outputs embed raw payload bits only.

## Current Contract

Fair capacity is:

```text
payload_bits_successfully_recovered_under_hard_visible_word_limit
```

There is no hidden length header in new stegotext:

- `payload_bits`: raw secret payload bits
- `header_bits`: compatibility field, currently `0`
- `target_bits` / `total_target_bits`: raw payload bit target
- `used_bits` / `total_used_bits`: raw payload bits consumed by the encoder

For example, a successful 40-bit payload trial targets 40 bits, not 56.

## Reveal Contract

Headerless decoding needs the payload length as external metadata. When revealing a `/hide` output, pass the response's `payload_bits` value back as `payload_bits_len`.

If `payload_bits_len` is omitted, `/reveal` only attempts the older framed decode path for legacy stegotext.
