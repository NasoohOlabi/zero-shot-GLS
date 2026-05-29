from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from bitstring import BitStream, ConstBitStream

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
sys.path.append(str(ROOT / "src"))

import bit2plain
import codec
import plain2bit
from zgls_api import (
    bits_to_payload,
    bits_to_payload_bits,
    payload_bits_to_bits,
    payload_to_bits,
)


class _TokenizerOutput:
    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids


class FakeTokenizer:
    """Small byte-level tokenizer with stable ids for deterministic codec tests."""

    def __init__(self) -> None:
        self.offset = 3
        self.vocab_size = self.offset + 256

    def __call__(
        self,
        text: str,
        return_tensors: str,
        max_length: int,
        truncation: bool,
    ) -> _TokenizerOutput:
        assert return_tensors == "pt"
        raw = text.encode("utf-8")
        if truncation:
            raw = raw[:max_length]
        ids = [byte + self.offset for byte in raw]
        return _TokenizerOutput(torch.tensor([ids], dtype=torch.long))

    def decode(self, token_ids: list[int]) -> str:
        raw = bytes(token_id - self.offset for token_id in token_ids)
        return raw.decode("utf-8")


class FakeConfig:
    bos_token_id = 0
    eos_token_id = 1


class FakeModel:
    """Minimal language-model shape required by codec.encode/decode."""

    def __init__(self, vocab_size: int) -> None:
        self.config = FakeConfig()
        self.device = torch.device("cpu")
        # The codec only needs a reproducible distribution over the full vocab.
        self._logits = torch.linspace(1.0, -1.0, steps=vocab_size, dtype=torch.float32)

    def __call__(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        logits = self._logits.repeat(input_ids.size(0), seq_len, 1)
        return type("FakeOutput", (), {"logits": logits})()


@pytest.fixture
def fake_codec_stack() -> tuple[FakeModel, FakeTokenizer]:
    tokenizer = FakeTokenizer()
    return FakeModel(tokenizer.vocab_size), tokenizer


@pytest.mark.parametrize(
    "message",
    [
        "hello",
        "encode/decode round trip",
        "line one\nline two\twith tab",
        "symbols: !@#$%^&*()[]{}",
        "utf-8: cafe \u2603 \u0633\u0644\u0627\u0645",
    ],
)
def test_plaintext_message_encode_decode_roundtrip(fake_codec_stack, message):
    model, tokenizer = fake_codec_stack

    encoded, encoded_without_ef = plain2bit.encode(
        model=model,
        tokenizer=tokenizer,
        plaintext=message,
        max_token_lengh=512,
        size_bits=12,
        ef_bits=4,
    )

    assert bit2plain.decode(model, tokenizer, encoded, size_bits=12, ef_bits=4) == message
    assert bit2plain.decode(model, tokenizer, encoded_without_ef, size_bits=12, ef_bits=4) == message


@pytest.mark.parametrize(
    "token_ids",
    [
        [104, 105],
        [115, 101, 99, 114, 101, 116],
        [10, 0, 255, 128, 64],
    ],
)
def test_token_id_codec_roundtrip(fake_codec_stack, token_ids):
    model, tokenizer = fake_codec_stack
    input_ids = torch.tensor([[token + tokenizer.offset for token in token_ids]], dtype=torch.long)

    bits = codec.encode_token_ids(model, input_ids, max_bits_len=4096, add_bos_token=True)
    decoded = codec.decode_bitstream(model, ConstBitStream(bits), remove_bos_token=True)

    assert decoded.tolist() == input_ids.tolist()


@pytest.mark.parametrize("enable_ef", [True, False])
@pytest.mark.parametrize("bit_string", ["1", "0", "1010101", "111100001010", "0" * 31 + "1"])
def test_bit_wrapping_roundtrip_preserves_exact_payload_bits(bit_string, enable_ef):
    raw_bits = ConstBitStream(bin=bit_string)

    wrapped = codec.wrap_bits(raw_bits, size_bits=8, ef_bits=4, enable_ef=enable_ef)
    padded = BitStream(wrapped)
    padded.append("0b0" * (8 - len(padded) % 8))
    recovered = codec.unwrap_bits(
        codec.base642bits(codec.bits2base64(padded)),
        size_bits=8,
        ef_bits=4,
    )

    assert recovered.bin == bit_string


@pytest.mark.parametrize(
    "message",
    [
        "short secret",
        "multi-line\nsecret",
        "utf-8 secret: \u2603 \u0633\u0644\u0627\u0645",
    ],
)
def test_api_payload_message_roundtrip_without_header(message):
    bits = payload_to_bits(message)

    decoded, byte_len = bits_to_payload(bits, len(bits))

    assert decoded == message
    assert byte_len == len(message.encode("utf-8"))


@pytest.mark.parametrize("payload_bits", ["1", "101", "0" * 15 + "1", "10101010101"])
def test_api_payload_bits_roundtrip_non_byte_aligned_without_header(payload_bits):
    bits = payload_bits_to_bits(payload_bits)

    recovered, bit_len = bits_to_payload_bits(bits, len(payload_bits))

    assert recovered == payload_bits
    assert bit_len == len(payload_bits)
