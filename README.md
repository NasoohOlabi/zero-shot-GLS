# Zero-shot Generative Linguistic Steganography

[![arXiv](https://img.shields.io/badge/arXiv-2403.10856-brightgreen.svg)](https://arxiv.org/abs/2403.10856)
[![star badge](https://img.shields.io/github/stars/leonardodalinky/zero-shot-GLS?style=social)](https://github.com/leonardodalinky/zero-shot-GLS)

This repo is the official implementation of NAACL'24 paper "[Zero-shot Generative Linguistic Steganography](https://aclanthology.org/2024.naacl-long.289/)".

## Setup

1. Create Conda environment and install the requirements:
```shell
conda create -n zgls python=3.10
conda activate zgls
#
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
```
2. Compile external Python module:
```shell
pip install external_tools/zgls-utils
```
3. Install pre-commit hooks:
```shell
pre-commit install
```
4. Create a temp folder in HDD for storing and sharing files, for example:
```shell
# DO NOT copy directly!
mkdir -p $SHARE_FOLDER/zgls
ln -s $SHARE_FOLDER/zgls tmp_saves
ln -s $SHARE_FOLDER/datasets/imdb datasets/imdb
ln -s $SHARE_FOLDER/datasets/twitter datasets/twitter
```

### Backend compatibility

For fair reproduction of the paper's stage2/stage3 hiding and extraction pipeline, the backend
must expose tokenization, detokenization, and token-level next-token top-logprobs. An
OpenAI-compatible local server alone is not enough if it omits those lower-level capabilities.

## Datasets

Check [datasets/](datasets/README.md) section for details.

## Usage

See [scripts/](scripts/README.md) section for details.

### Python API

The reusable ZGLS API lives in `src/zgls_api.py`. It wraps the llama-server backed
hide/reveal pipeline used by the HTTP server.

```powershell
$env:PYTHONPATH="src"
@'
from zgls_api import ZGLSClient, ZGLSConfig

client = ZGLSClient(ZGLSConfig(
    server_url="http://127.0.0.1:8081",
    server_model="Qwen3.5-9B-Q4_K_M.gguf",
    corpus="IMDB_about_movies",
))

cover = [
    "I expected this film to be better than it was.",
    "The performances were strong but the pacing felt uneven.",
]
hidden = client.hide(
    prompt="Write one natural short sentence about coffee.",
    secret="hello",
    cover_texts=cover,
)
revealed = client.reveal(
    prompt="Write one natural short sentence about coffee.",
    stegotext=hidden.stegotext,
    context_seed=hidden.context_seed,
    effective_prompt_hash=hidden.effective_prompt_hash,
    stego_token_ids=hidden.stego_token_ids,
    payload_bits_len=hidden.payload_bits,
    cover_texts=cover,
)
print(hidden.stegotext)
print(revealed.secret)
'@ | .\.venv\Scripts\python.exe -
```

### HTTP API

Run the FastAPI service from PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\stego_api_server.py `
  --host 127.0.0.1 `
  --port 9000 `
  --server-url http://127.0.0.1:8081 `
  --server-model Qwen3.5-9B-Q4_K_M.gguf `
  --corpus IMDB_about_movies
```

Then call `/hide` and `/reveal` as documented in `docs/STEGO_API_SERVER.md`.

## Evaluation

For details of metrics, steganalysis, and language evaluation, check [evaluate/](evaluate/README.md) section.

Generate confidence-oriented samples:

```powershell
.\.venv\Scripts\python.exe scripts\generate_zgls_metric_samples.py `
  --server-url http://127.0.0.1:8081 `
  --server-model Qwen3.5-9B-Q4_K_M.gguf `
  --cover-path datasets\imdb\imdb.csv `
  --cover-col plaintext `
  --corpus IMDB_about_movies `
  --min-samples 100 `
  --max-samples 2000 `
  --kld-cover-limit 1000 `
  --output-dir tmp_saves\runs\zgls_metric_samples `
  --force
```

Compute PPL and KLD summaries:

```powershell
.\.venv\Scripts\python.exe evaluate\ppl.py tmp_saves\runs\zgls_metric_samples\samples.csv `
  --ppl-col ppl `
  --output tmp_saves\runs\zgls_metric_samples\ppl.json `
  --force

.\.venv\Scripts\python.exe evaluate\kld.py `
  tmp_saves\runs\zgls_metric_samples\samples.csv `
  datasets\imdb\imdb.csv `
  --stego-col stegotext `
  --cover-col plaintext `
  --server-url http://127.0.0.1:8081 `
  --server-model Qwen3.5-9B-Q4_K_M.gguf `
  --output tmp_saves\runs\zgls_metric_samples\kld.json `
  --force
```

Run local metric tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_metrics.py
```

Run real llama-server integration tests:

```powershell
$env:ZGLS_SERVER_URL="http://127.0.0.1:8081"
$env:ZGLS_SERVER_MODEL="Qwen3.5-9B-Q4_K_M.gguf"
.\.venv\Scripts\python.exe -m pytest tests\test_real_llama_api.py
```

## Reference

```
@inproceedings{lin2024zgls,
    title = "Zero-shot Generative Linguistic Steganography",
    author = "Lin, Ke  and
      Luo, Yiyang  and
      Zhang, Zijian  and
      Ping, Luo",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.289",
    pages = "5168--5182",
    doi = "10.18653/v1/2024.naacl-long.289"
}
```
