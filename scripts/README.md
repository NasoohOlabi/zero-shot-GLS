# Stego Pipeline

## Setup

First, download all models.
```shell
python download_models.py
```

## Stages

* stage1 (s1,  encode): Plaintext to enc_bits.
* stage2 (s2, encrypt): Enc_bits to stegotext.
* stage3 (s3, decrypt): Stegotext to dec_bits.
* stage4 (s4,  decode): Dec_bits to dec_plaintext.

## Stego modes

For now, we have only one mode:
* `--mode=cover`: mimic the cover text.

## Corpus Hints

For IMDB, we use hint `Movie Reviews`.

For Twitter, we use hint `Twitter`.

## API Server

For full docs of the stego HTTP server (`/health`, `/hide`, `/reveal`), see:

- `docs/STEGO_API_SERVER.md`

Prompt construction for `/hide` and `/reveal` is config-driven and loaded from:

- `config/workflow_llm_prompts.json` (`stego_encode.system_template`, `stego_encode.user_template`)

## Backend Note

The stage2/stage3 EGS pipeline requires a backend that exposes all of the following:

- prompt tokenization
- token-id detokenization
- next-token top-logprobs with token ids

LM Studio's current OpenAI-compatible local server is useful for prompt-side sampling tests,
but it does not provide the full tokenization + top-logprob surface this repo expects for fair
paper reproduction. Use a llama.cpp-style server that exposes `/tokenize`, `/detokenize`, and
token-level top-logprobs.

## Prompt Style Note

Stage2 and stage3 now support `--prompt-style` with `auto`, `paper`, and `qwen_plain`.

- `auto` is the default and selects `qwen_plain` when the backend model hint contains `qwen`.
- `paper` keeps the Appendix A.3-style prompt used in the paper.
- `qwen_plain` uses a plain `/no_think` completion prompt that avoids the XML/tag echo behavior
  observed with Qwen on local completion backends.

## Python API and Metric Sampling

The public Python API is in `src/zgls_api.py`. Use `ZGLSClient.hide()` and
`ZGLSClient.reveal()` for direct programmatic access, or run `stego_api_server.py`
for the HTTP API.

Generate statistically controlled PPL/KLD samples:

```powershell
.\.venv\Scripts\python.exe scripts\generate_zgls_metric_samples.py `
  --server-url http://127.0.0.1:8081 `
  --server-model Qwen3.5-9B-Q4_K_M.gguf `
  --cover-path datasets\imdb\imdb.csv `
  --cover-col plaintext `
  --corpus IMDB_about_movies `
  --min-samples 100 `
  --max-samples 2000 `
  --relative-ci-target 0.05 `
  --kld-cover-limit 1000 `
  --output-dir tmp_saves\runs\zgls_metric_samples `
  --force
```

Outputs:

- `samples.jsonl`: full raw sample records
- `samples.csv`: spreadsheet-friendly sample table
- `summary.json`: PPL/KLD confidence summary and stopping reason
