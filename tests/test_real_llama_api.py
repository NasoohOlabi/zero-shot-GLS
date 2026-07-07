from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from stego_api_server import build_app
from zgls_api import EGSParams, ZGLSClient, ZGLSConfig, ZGLSError


COVER_TEXTS = [
    "I expected this film to be better than it was.",
    "The performances were strong but the pacing felt uneven.",
    "It had a few nice scenes, though the ending felt rushed.",
]


def real_cfg(**kwargs) -> ZGLSConfig:
    server_url = os.getenv("ZGLS_SERVER_URL")
    server_model = os.getenv("ZGLS_SERVER_MODEL")
    if not server_url or not server_model:
        pytest.skip("Set ZGLS_SERVER_URL and ZGLS_SERVER_MODEL to run real backend tests.")
    cfg = ZGLSConfig(
        server_url=server_url,
        server_model=server_model,
        seed=int(os.getenv("ZGLS_TEST_SEED", "42")),
        cover_path=os.getenv("ZGLS_COVER_PATH") or None,
        cover_col=os.getenv("ZGLS_COVER_COL", "plaintext"),
        corpus="IMDB_about_movies",
        quality_max_retries=1,
        max_new_tokens=128,
        egs=EGSParams(mode=kwargs.pop("mode", "huffman"), max_bpw=2),
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_python_api_roundtrip_edge_cases():
    client = ZGLSClient(real_cfg())
    cases = [
        ("Write one casual sentence about a movie.", "a"),
        ("  Write with whitespace around the instruction.  ", "cafe"),
        ("Say something short: wow?! really...", "edge-case!"),
    ]
    for prompt, secret in cases:
        hidden = client.hide(prompt=prompt, secret=secret, cover_texts=COVER_TEXTS)
        revealed = client.reveal(
            prompt=prompt,
            stegotext=hidden.stegotext,
            context_seed=hidden.context_seed,
            effective_prompt_hash=hidden.effective_prompt_hash,
            cover_texts=COVER_TEXTS,
            stego_token_ids=hidden.stego_token_ids,
            payload_bits_len=hidden.payload_bits,
        )
        assert revealed.decode_ok
        assert revealed.secret == secret


def test_python_api_complete_sentence_and_hash_warning():
    client = ZGLSClient(real_cfg())
    hidden = client.hide(
        prompt="Write one short movie reaction.",
        secret="hello",
        cover_texts=COVER_TEXTS,
        complete_sent=True,
    )
    revealed = client.reveal(
        prompt="Write one short movie reaction.",
        stegotext=hidden.stegotext,
        context_seed=hidden.context_seed,
        effective_prompt_hash="wrong",
        cover_texts=COVER_TEXTS,
        stego_token_ids=hidden.stego_token_ids,
        payload_bits_len=hidden.payload_bits,
    )
    assert "provided effective_prompt_hash does not match reconstructed prompt" in revealed.warnings


def test_python_api_low_max_tokens_truncates_or_fails_cleanly():
    client = ZGLSClient(real_cfg(max_new_tokens=1, quality_max_retries=0))
    with pytest.raises(ZGLSError) as exc:
        client.hide(prompt="Write one sentence.", secret="too long", cover_texts=COVER_TEXTS)
    assert exc.value.status_code == 422


def test_python_api_block_mode_if_backend_has_candidates():
    client = ZGLSClient(real_cfg(mode="block"))
    hidden = client.hide(prompt="Write one short movie reaction.", secret="a", cover_texts=COVER_TEXTS)
    revealed = client.reveal(
        prompt="Write one short movie reaction.",
        stegotext=hidden.stegotext,
        context_seed=hidden.context_seed,
        effective_prompt_hash=hidden.effective_prompt_hash,
        cover_texts=COVER_TEXTS,
        stego_token_ids=hidden.stego_token_ids,
        payload_bits_len=hidden.payload_bits,
    )
    assert revealed.secret == "a"


def test_http_contract_health_hide_reveal_and_errors():
    app = build_app(real_cfg())
    http = TestClient(app)
    health = http.get("/health")
    assert health.status_code == 200
    assert health.json()["header_bits"] == 0

    hide = http.post(
        "/hide",
        json={
            "prompt": "Write one natural short sentence about coffee.",
            "secret": "hello",
            "cover_texts": COVER_TEXTS,
        },
    )
    assert hide.status_code == 200, hide.text
    payload = hide.json()

    reveal = http.post(
        "/reveal",
        json={
            "prompt": "Write one natural short sentence about coffee.",
            "stegotext": payload["stegotext"],
            "stego_token_ids": payload["stego_token_ids"],
            "payload_bits_len": payload["payload_bits"],
            "context_seed": payload["context_seed"],
            "effective_prompt_hash": payload["effective_prompt_hash"],
            "cover_texts": COVER_TEXTS,
        },
    )
    assert reveal.status_code == 200, reveal.text
    assert reveal.json()["secret"] == "hello"

    empty_secret = http.post(
        "/hide",
        json={"prompt": "x", "secret": "", "cover_texts": COVER_TEXTS},
    )
    assert empty_secret.status_code == 400

    missing_stegotext = http.post("/reveal", json={"prompt": "x"})
    assert missing_stegotext.status_code == 422


def test_http_missing_cover_source_returns_400():
    app = build_app(real_cfg(cover_path=None))
    http = TestClient(app)
    resp = http.post("/hide", json={"prompt": "x", "secret": "a"})
    assert resp.status_code == 400
