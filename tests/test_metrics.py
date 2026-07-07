from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "evaluate"))

import zgls_api
from zgls_api import (
    CapacityTrialResult,
    HEADER_BITS,
    HideResult,
    RevealResult,
    SampleConfig,
    ZGLSClient,
    ZGLSConfig,
    best_capacity_success,
    bits_to_payload,
    bits_to_payload_bits,
    bootstrap_ci,
    bootstrap_kld_ci,
    bytes_to_base64,
    compute_metrics,
    kld_from_counts,
    payload_bits_to_bits,
    payload_to_bits,
    quality_pass,
    repetition_metrics,
)


def test_payload_roundtrip_utf8_without_header():
    bits = payload_to_bits("cafe")
    decoded, byte_len = bits_to_payload(bits, len(bits))
    assert decoded == "cafe"
    assert byte_len == len("cafe".encode("utf-8"))
    payload_bits, payload_len = bits_to_payload_bits(bits, len(bits))
    assert payload_len == 32
    assert len(payload_bits) == 32
    assert len(bits) == payload_len


@pytest.mark.parametrize("payload_len", [1, 11, 12, 13])
def test_payload_roundtrip_non_byte_aligned_bits_without_header(payload_len):
    payload = ("10" * payload_len)[:payload_len]
    bits = payload_bits_to_bits(payload)
    recovered, recovered_len = bits_to_payload_bits(bits, payload_len)
    assert recovered == payload
    assert recovered_len == payload_len
    assert len(bits) == payload_len
    assert HEADER_BITS == 0


def test_bootstrap_ci_is_deterministic():
    first = bootstrap_ci([1, 2, 3, 4], iters=100, seed=7)
    second = bootstrap_ci([1, 2, 3, 4], iters=100, seed=7)
    assert first == second
    assert first["count"] == 4
    assert first["mean"] == pytest.approx(2.5)


def test_kld_smoothing_avoids_infinity_for_disjoint_vocabularies():
    value = kld_from_counts({"a": 3}, {"b": 3}, alpha=1e-6)
    assert math.isfinite(value)
    assert value > 0


def test_bootstrap_kld_ci_is_deterministic():
    first = bootstrap_kld_ci(["red blue", "red"], ["green yellow"], iters=50, seed=11)
    second = bootstrap_kld_ci(["red blue", "red"], ["green yellow"], iters=50, seed=11)
    assert first == second
    assert first["kld_symmetric"] is not None
    assert first["ci_low"] is not None


def test_compute_metrics_counts_failures():
    metrics = compute_metrics(
        [
            {"success": True, "ppl": 2.0, "stegotext": "one two"},
            {"success": False, "error_category": "quality_gate_failed"},
        ],
        cover_texts=["one three"],
        bootstrap_iters=20,
        seed=1,
    )
    assert metrics.sample_count == 2
    assert metrics.success_count == 1
    assert metrics.failure_count == 1
    assert metrics.failure_rate == pytest.approx(0.5)


def test_quality_gate_rejects_prompt_and_control_artifacts():
    bad_text = "Your task is to output similar input.</OUTPUT><|im_end|>"
    metrics = repetition_metrics(bad_text)
    assert metrics["structural_artifact_count"] > 0
    assert metrics["prompt_echo_count"] > 0
    assert not quality_pass(
        metrics,
        max_repetition_ratio=0.65,
        max_single_token_share=0.35,
    )


def test_quality_gate_rejects_mojibake_and_non_ascii():
    mojibake = repetition_metrics("It is not about the money anymoreâ€”it is about being seen.")
    assert mojibake["mojibake_artifact_count"] > 0
    assert not quality_pass(
        mojibake,
        max_repetition_ratio=0.65,
        max_single_token_share=0.35,
    )

    non_ascii = repetition_metrics("It is not about the money anymore—it is about being seen.")
    assert non_ascii["has_non_ascii"] == 1
    assert not quality_pass(
        non_ascii,
        max_repetition_ratio=0.65,
        max_single_token_share=0.35,
    )


def test_quality_gate_rejects_malformed_joins_and_repeated_bigrams():
    malformed = repetition_metrics(
        "The scene was fine., The scene was fine too."
    )
    assert malformed["malformed_punctuation_count"] > 0
    assert not quality_pass(
        malformed,
        max_repetition_ratio=0.65,
        max_single_token_share=0.35,
    )

    joined = repetition_metrics(
        "It takes on a beautiful appearance FYThere is truly a strange join."
    )
    assert joined["malformed_punctuation_count"] > 0
    assert not quality_pass(
        joined,
        max_repetition_ratio=0.65,
        max_single_token_share=0.35,
    )

    repeated_bigram = repetition_metrics(
        "You can watch it over and over and find more jokes over and over."
    )
    assert repeated_bigram["max_bigram_repeat"] > 1
    assert not quality_pass(
        repeated_bigram,
        max_repetition_ratio=0.65,
        max_single_token_share=0.35,
    )


def test_quality_gate_word_ceiling_is_configurable_for_capacity():
    text = " ".join(f"word{i}" for i in range(45)) + "."
    metrics = repetition_metrics(text)
    assert metrics["word_count"] == 45
    assert not quality_pass(
        metrics,
        max_repetition_ratio=0.65,
        max_single_token_share=0.35,
    )
    assert quality_pass(
        metrics,
        max_words=60,
        max_repetition_ratio=0.65,
        max_single_token_share=0.35,
    )


def make_capacity_trial(payload_bits: int, success: bool, word_count: int) -> CapacityTrialResult:
    return CapacityTrialResult(
        payload_bits=payload_bits,
        payload_bytes=math.ceil(payload_bits / 8),
        payload_bits_exact=payload_bits,
        header_bits=HEADER_BITS,
        total_target_bits=payload_bits,
        total_used_bits=payload_bits,
        used_bits=payload_bits,
        success=success,
        decode_ok=success,
        secret_matches=success,
        quality_passed=success,
        word_count=word_count,
        stegotext="ok.",
        stego_token_ids=[],
        context_seed=1,
        effective_prompt_hash="hash",
        is_truncated=False,
        ppl=1.0,
        quality_metrics={},
        failure_reason=None if success else "quality_gate_failed",
        warnings=[],
        mode="huffman",
        params_used={},
    )


def test_capacity_accounting_and_best_success():
    small = make_capacity_trial(8, True, 18)
    failed_large = make_capacity_trial(64, False, 39)
    medium = make_capacity_trial(24, True, 25)
    assert medium.header_bits == 0
    assert medium.total_target_bits == medium.payload_bits_exact
    assert best_capacity_success([small, failed_large, medium]) == medium


def test_generate_samples_reveal_uses_headerless_payload_length():
    client = ZGLSClient.__new__(ZGLSClient)
    client.cover_text = ["one two three four five six seven eight."]
    client.tokenizer = None
    reveal_payload_lengths: list[int | None] = []

    def hide(**kwargs):
        return HideResult(
            stegotext="A simple clean sentence with enough distinct words today.",
            stego_token_ids=[1, 2, 3],
            context_seed=123,
            effective_prompt_hash="hash",
            used_bits=8,
            target_bits=8,
            bpw_estimate=1.0,
            payload_bytes=1,
            payload_bits=8,
            header_bits=HEADER_BITS,
            total_target_bits=8,
            total_used_bits=8,
            is_truncated=False,
            ppl=1.0,
            quality_passed=True,
            quality_metrics={},
            mode="huffman",
            params_used={},
        )

    def reveal(**kwargs):
        reveal_payload_lengths.append(kwargs.get("payload_bits_len"))
        return RevealResult(
            secret="x",
            payload_bytes=1,
            payload_bits="01111000",
            payload_bits_len=kwargs.get("payload_bits_len"),
            decode_ok=True,
            raw_bits_len=8,
            warnings=[],
            context_seed=123,
            effective_prompt_hash="hash",
            mode="huffman",
            params_used={},
        )

    client.hide = hide
    client.reveal = reveal

    result = ZGLSClient.generate_samples(
        client,
        SampleConfig(
            prompts=["Write one sentence."],
            secrets=["x"],
            min_samples=1,
            max_samples=1,
            bootstrap_iters=1,
        ),
    )

    assert reveal_payload_lengths == [8]
    assert result.samples[0]["roundtrip_ok"]


def test_hide_raw_base64_payload_returns_exact_partial_remainder(monkeypatch):
    class TokenizerOutput:
        input_ids = torch.tensor([[10, 11]], dtype=torch.long)

    class FakeTokenizer:
        def __call__(self, text, return_tensors):
            assert text == "complete prompt"
            assert return_tensors == "pt"
            return TokenizerOutput()

        def decode(self, token_ids):
            assert token_ids == [99, 100]
            return "Partial stego text."

    class FakeModel:
        device = torch.device("cpu")

    def fake_hide_bits(**kwargs):
        assert kwargs["bits"].bin == "0100000101000010"
        prompt_ids = kwargs["prompt_ids"]
        suffix = torch.tensor([[99, 100]], dtype=torch.long)
        return torch.cat([prompt_ids, suffix], dim=1), True, 9, 1.0

    monkeypatch.setattr(zgls_api.hide_extract, "hide_bits_with_prompt_ids_by_egs", fake_hide_bits)

    client = ZGLSClient.__new__(ZGLSClient)
    client.cfg = ZGLSConfig(quality_max_retries=0)
    client.tokenizer = FakeTokenizer()
    client.model = FakeModel()

    result = ZGLSClient.hide(
        client,
        prompt="complete prompt",
        payload_base64=bytes_to_base64(b"AB"),
        max_new_tokens=2,
        allow_partial=True,
        enforce_quality=False,
    )

    assert result.stegotext == "Partial stego text."
    assert result.embedded_bits == 9
    assert not result.fully_embedded
    assert result.remaining_bits == "1000010"
    assert result.remaining_bits_len == 7
    assert result.remaining_payload_base64 is None
    assert result.payload_base64 == bytes_to_base64(b"AB")


def test_http_models_expose_benchmark_overrides():
    sys.path.append(str(ROOT / "scripts"))
    from stego_api_server import (
        CapacityProbeRequest,
        CapacityProbeResponse,
        HideRequest,
        HideResponse,
        RevealRequest,
    )

    hide_schema = (
        HideRequest.model_json_schema()
        if hasattr(HideRequest, "model_json_schema")
        else HideRequest.schema()
    )
    hide_response_schema = (
        HideResponse.model_json_schema()
        if hasattr(HideResponse, "model_json_schema")
        else HideResponse.schema()
    )
    reveal_schema = (
        RevealRequest.model_json_schema()
        if hasattr(RevealRequest, "model_json_schema")
        else RevealRequest.schema()
    )
    capacity_schema = (
        CapacityProbeRequest.model_json_schema()
        if hasattr(CapacityProbeRequest, "model_json_schema")
        else CapacityProbeRequest.schema()
    )
    capacity_response_schema = (
        CapacityProbeResponse.model_json_schema()
        if hasattr(CapacityProbeResponse, "model_json_schema")
        else CapacityProbeResponse.schema()
    )
    for field in (
        "secret",
        "payload_base64",
        "payload_bits",
        "allow_partial",
        "enforce_quality",
        "max_new_tokens",
        "quality_max_retries",
        "quality_max_words",
        "threshold",
        "temperature",
        "temperature_alpha",
        "max_bpw",
    ):
        assert field in hide_schema["properties"]
    for field in (
        "embedded_bits",
        "fully_embedded",
        "remaining_bits_len",
        "remaining_bits",
        "payload_base64",
        "remaining_payload_base64",
    ):
        assert field in hide_response_schema["properties"]
    for field in ("payload_bits_len", "threshold", "temperature", "temperature_alpha", "max_bpw"):
        assert field in reveal_schema["properties"]
    for field in (
        "prompt",
        "payload_bits",
        "max_words",
        "payload_bits_candidates",
        "initial_payload_bits",
        "payload_seed",
        "quality_max_retries",
        "threshold",
        "temperature",
        "temperature_alpha",
        "max_bpw",
    ):
        assert field in capacity_schema["properties"]
    for field in ("best_success", "trials", "params_used"):
        assert field in capacity_response_schema["properties"]


def test_ppl_loader_rejects_invalid_by_default(tmp_path):
    import ppl

    path = tmp_path / "ppl.csv"
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["ppl"])
        writer.writeheader()
        writer.writerow({"ppl": "3.0"})
        writer.writerow({"ppl": "bad"})

    with pytest.raises(ValueError):
        ppl.load_ppl_values(str(path), "ppl", drop_invalid=False)
    assert ppl.load_ppl_values(str(path), "ppl", drop_invalid=True) == [3.0]
