"""Public Python API for Zero-shot Generative Linguistic Steganography.

The API intentionally wraps the existing EGS hide/extract implementation without
changing its algorithmic behavior. It gives scripts, tests, and the HTTP server
one shared contract for configuration, hide/reveal calls, sampling, and metrics.
"""

from __future__ import annotations

import csv
import base64
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import torch
from bitstring import BitStream, ConstBitStream

import hide_extract
from llama_server import LlamaServerConfig, LlamaServerModel, LlamaServerTokenizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_CFG_PATH = ROOT / "config" / "workflow_llm_prompts.json"
HEADER_BITS = 0
LEGACY_HEADER_BITS = 16
MAX_PAYLOAD_BITS = (2**LEGACY_HEADER_BITS) - 1
MAX_PAYLOAD_BYTES = MAX_PAYLOAD_BITS // 8
WORD_RE = re.compile(r"[A-Za-z0-9']+")
STRUCTURAL_ARTIFACT_RE = re.compile(
    r"(<\|[^>]+?\|>|</?[^>\s]{1,40}[^>]*>|\\?```|\[/|/\]|"
    r"\b(?:INST|SYS|SYSTEM|USER|ASSISTANT|OUTPUT|CONTEXT|CORPUS)\b\s*[:>\]])",
    re.IGNORECASE,
)
PROMPT_ECHO_RE = re.compile(
    r"\b(?:your task|the input format|users? will input|output only|"
    r"example sentence|match input|similar input|do not use|do not write|"
    r"write exactly|sentence:)\b",
    re.IGNORECASE,
)
META_ARTIFACT_RE = re.compile(
    r"\b(?:repeated twice|as an ai|imdb|reddit-style|thread context)\b",
    re.IGNORECASE,
)
TERMINAL_PUNCT_RE = re.compile(r"""[.!?'"”’)\]]\s*$""")
MALFORMED_PUNCT_RE = re.compile(
    r"(?:[.!?]{2,}\s*[A-Z]|[a-z][.!?][A-Z]|[.!?][,;:]|"
    r"[.!?]['\"][A-Z]|[a-z][A-Z]{2}[A-Za-z]|\b[A-Z]{3,}[a-z]|\s+[,.!?]|"
    r"\([^)]*(?i:do\s+not)[^)]*\))",
)
MOJIBAKE_RE = re.compile(r"(?:â.|Ã.|Â.|�)")


class ZGLSError(RuntimeError):
    """Error raised by the public ZGLS API.

    ``status_code`` mirrors HTTP semantics so the FastAPI layer can preserve its
    previous behavior without embedding transport exceptions in the core API.
    """

    def __init__(self, message: str, *, status_code: int = 500, detail: Any | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail if detail is not None else message


@dataclass
class EGSParams:
    mode: str = "huffman"
    threshold: float = 1e-2
    temperature: float = 0.7
    temperature_alpha: float = 1.0
    max_bpw: int = 2


@dataclass
class ZGLSConfig:
    backend: str = "llama_server"
    server_url: str = "http://127.0.0.1:8081"
    server_model: str = "Qwen3.5-9B-Q4_K_M.gguf"
    seed: int = 42
    max_new_tokens: int = 256
    cover_path: str | None = None
    cover_col: str = "plaintext"
    corpus: str = "Unknown"
    n_cover: int = 2
    quality_max_retries: int = 4
    quality_min_words: int = 8
    quality_max_words: int = 40
    quality_max_repetition_ratio: float = 0.65
    quality_max_single_token_share: float = 0.35
    prompt_cfg_path: str = str(DEFAULT_PROMPT_CFG_PATH)
    egs: EGSParams = field(default_factory=EGSParams)


@dataclass
class HideResult:
    stegotext: str
    stego_token_ids: list[int]
    context_seed: int
    effective_prompt_hash: str
    used_bits: int
    target_bits: int
    bpw_estimate: float
    payload_bytes: int
    payload_bits: int
    header_bits: int
    total_target_bits: int
    total_used_bits: int
    is_truncated: bool
    ppl: float
    quality_passed: bool
    quality_metrics: dict[str, Any]
    mode: str
    params_used: dict[str, Any]
    embedded_bits: int = 0
    fully_embedded: bool = False
    remaining_bits_len: int = 0
    remaining_bits: str = ""
    payload_base64: str | None = None
    remaining_payload_base64: str | None = None


@dataclass
class RevealResult:
    secret: str | None
    payload_bytes: int | None
    payload_bits: str | None
    payload_bits_len: int | None
    decode_ok: bool
    raw_bits_len: int
    warnings: list[str]
    context_seed: int | None
    effective_prompt_hash: str | None
    mode: str
    params_used: dict[str, Any]


@dataclass
class CapacityTrialResult:
    payload_bits: int
    payload_bytes: int
    payload_bits_exact: int
    header_bits: int
    total_target_bits: int
    total_used_bits: int
    used_bits: int
    success: bool
    decode_ok: bool
    secret_matches: bool
    quality_passed: bool
    word_count: int
    stegotext: str | None
    stego_token_ids: list[int]
    context_seed: int | None
    effective_prompt_hash: str | None
    is_truncated: bool
    ppl: float | None
    quality_metrics: dict[str, Any]
    failure_reason: str | None
    warnings: list[str]
    mode: str
    params_used: dict[str, Any]


@dataclass
class CapacityProbeResult:
    best_success: CapacityTrialResult | None
    trials: list[CapacityTrialResult]
    params_used: dict[str, Any]


@dataclass
class SampleConfig:
    prompts: list[str]
    secrets: list[str]
    cover_texts: list[str] | None = None
    corpus: str | None = None
    min_samples: int = 100
    max_samples: int = 2000
    bootstrap_iters: int = 1000
    confidence: float = 0.95
    relative_ci_target: float = 0.05
    seed: int = 42
    complete_sent: bool = False
    kld_cover_limit: int = 1000
    progress_path: str | None = None


@dataclass
class MetricsResult:
    sample_count: int
    success_count: int
    failure_count: int
    failure_rate: float
    ppl: dict[str, float | int | None]
    kld: dict[str, float | int | None]
    stopping_reason: str | None = None


@dataclass
class SampleRunResult:
    samples: list[dict[str, Any]]
    metrics: MetricsResult


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)


def load_cover_text(cover_path: str, cover_col: str) -> list[str]:
    rows: list[str] = []
    with open(cover_path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if cover_col not in (reader.fieldnames or []):
            raise ZGLSError(
                f"cover column '{cover_col}' not found in {cover_path}",
                status_code=400,
            )
        for row in reader:
            text = (row.get(cover_col) or "").strip()
            if text:
                rows.append(text)
    if not rows:
        raise ZGLSError(f"cover text is empty: {cover_path}", status_code=400)
    return rows


def load_prompt_templates(path: str | Path = DEFAULT_PROMPT_CFG_PATH) -> tuple[str, str]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8-sig") as fp:
        cfg = json.load(fp)
    if "stego_encode" not in cfg:
        raise ZGLSError("Missing key: stego_encode in workflow_llm_prompts.json")
    stego = cfg["stego_encode"]
    system_template = stego.get("system_template")
    user_template = stego.get("user_template")
    if not isinstance(system_template, str) or not system_template.strip():
        raise ZGLSError("Missing key: stego_encode.system_template in workflow_llm_prompts.json")
    if not isinstance(user_template, str) or not user_template.strip():
        raise ZGLSError("Missing key: stego_encode.user_template in workflow_llm_prompts.json")
    return system_template, user_template


def build_effective_prompt(
    *,
    user_prompt: str,
    cover_text: list[str],
    corpus: str,
    n_cover: int,
    context_seed: int,
    system_template: str,
    user_template: str,
) -> str:
    rng = random.Random(context_seed)
    k = max(1, min(n_cover, len(cover_text)))
    sampled = rng.sample(cover_text, k)
    chain_section = ""
    if sampled:
        chain_section = "\n\nThread Context:\n" + "\n\n".join(sampled)

    values = {
        "tangent": "mood",
        "category": corpus,
        "target_category": corpus,
        "target_tangent": "mood",
        "target_source_quote": "",
        "best_match": "",
        "title": corpus,
        "author": "reddit_user",
        "selftext": user_prompt.strip(),
        "chain_section": chain_section,
    }
    return system_template.format(**values) + "\n\n" + user_template.format(**values)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def validate_payload_bits(payload_bits: str) -> str:
    if not isinstance(payload_bits, str):
        raise ZGLSError("payload bits must be a string", status_code=400)
    cleaned = payload_bits.strip()
    if not cleaned:
        raise ZGLSError("payload bits must not be empty", status_code=400)
    if any(bit not in "01" for bit in cleaned):
        raise ZGLSError("payload bits must contain only 0 and 1", status_code=400)
    if len(cleaned) > MAX_PAYLOAD_BITS:
        raise ZGLSError(
            f"payload too large: {len(cleaned)} bits (max={MAX_PAYLOAD_BITS})",
            status_code=400,
        )
    return cleaned


def payload_bits_to_framed_bits(payload_bits: str) -> ConstBitStream:
    """Legacy frame format used by older generated stegotext."""

    payload = validate_payload_bits(payload_bits)
    header = f"{len(payload):0{LEGACY_HEADER_BITS}b}"
    return ConstBitStream(bin=header + payload)


def framed_bits_to_payload_bits(bits: ConstBitStream) -> tuple[str, int]:
    if len(bits) < LEGACY_HEADER_BITS:
        raise ZGLSError(
            f"incomplete framed payload: need at least {LEGACY_HEADER_BITS} bits, got {len(bits)}",
            status_code=422,
        )

    payload_bit_length = int(bits[:LEGACY_HEADER_BITS].bin, 2)
    if payload_bit_length <= 0:
        raise ZGLSError(f"invalid payload bit length in header: {payload_bit_length}", status_code=422)
    if payload_bit_length > MAX_PAYLOAD_BITS:
        raise ZGLSError(
            f"header payload bit length out of range: {payload_bit_length}",
            status_code=422,
        )

    needed_bits = LEGACY_HEADER_BITS + payload_bit_length
    if len(bits) < needed_bits:
        raise ZGLSError(
            f"incomplete payload bits: need {needed_bits}, got {len(bits)}",
            status_code=422,
        )

    payload_bits = bits[LEGACY_HEADER_BITS:needed_bits].bin
    return payload_bits, payload_bit_length


def bytes_to_bits(raw: bytes) -> str:
    return "".join(f"{b:08b}" for b in raw)


def bytes_to_base64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def base64_to_bytes(payload_base64: str) -> bytes:
    if not isinstance(payload_base64, str):
        raise ZGLSError("payload_base64 must be a string", status_code=400)
    try:
        raw = base64.b64decode(payload_base64.encode("ascii"), validate=True)
    except Exception as e:
        raise ZGLSError(f"payload_base64 decode failed: {e}", status_code=400) from e
    if not raw:
        raise ZGLSError("payload_base64 must not decode to empty bytes", status_code=400)
    return raw


def payload_bytes_to_bits(raw: bytes) -> ConstBitStream:
    if not raw:
        raise ZGLSError("payload bytes must not be empty", status_code=400)
    payload_bits = bytes_to_bits(raw)
    if len(payload_bits) > MAX_PAYLOAD_BITS:
        raise ZGLSError(
            f"payload too large: {len(raw)} bytes ({len(payload_bits)} bits, max={MAX_PAYLOAD_BITS} bits)",
            status_code=400,
        )
    return payload_bits_to_bits(payload_bits)


def bit_string_to_base64_bytes(bit_string: str) -> str | None:
    if len(bit_string) % 8 != 0:
        return None
    raw = bytes(int(bit_string[idx : idx + 8], 2) for idx in range(0, len(bit_string), 8))
    return bytes_to_base64(raw)


def payload_bits_to_bits(payload_bits: str) -> ConstBitStream:
    return ConstBitStream(bin=validate_payload_bits(payload_bits))


def payload_to_bits(secret: str) -> ConstBitStream:
    raw = secret.encode("utf-8")
    try:
        return payload_bytes_to_bits(raw)
    except ZGLSError as exc:
        if not raw:
            raise ZGLSError("secret must not be empty", status_code=400) from exc
        raise


def payload_to_framed_bits(secret: str) -> ConstBitStream:
    raw_bits = payload_to_bits(secret).bin
    return payload_bits_to_framed_bits(raw_bits)


def hide_payload_to_bits(
    *,
    secret: str | None = None,
    payload_base64: str | None = None,
    payload_bits: str | None = None,
) -> tuple[ConstBitStream, int, str | None]:
    provided = [
        secret is not None,
        payload_base64 is not None,
        payload_bits is not None,
    ]
    if sum(provided) != 1:
        raise ZGLSError(
            "provide exactly one of secret, payload_base64, or payload_bits",
            status_code=400,
        )

    if secret is not None:
        raw = secret.encode("utf-8")
        return payload_bytes_to_bits(raw), len(raw), bytes_to_base64(raw)

    if payload_base64 is not None:
        raw = base64_to_bytes(payload_base64)
        return payload_bytes_to_bits(raw), len(raw), bytes_to_base64(raw)

    assert payload_bits is not None
    bits = payload_bits_to_bits(payload_bits)
    return bits, math.ceil(len(bits) / 8), bit_string_to_base64_bytes(bits.bin)


def bits_to_payload_bits(bits: ConstBitStream, payload_bits_len: int) -> tuple[str, int]:
    requested_len = int(payload_bits_len)
    if requested_len <= 0:
        raise ZGLSError("payload_bits_len must be positive", status_code=400)
    if requested_len > MAX_PAYLOAD_BITS:
        raise ZGLSError(
            f"payload_bits_len too large: {requested_len} bits (max={MAX_PAYLOAD_BITS})",
            status_code=400,
        )
    if len(bits) < requested_len:
        raise ZGLSError(
            f"incomplete payload bits: need {requested_len}, got {len(bits)}",
            status_code=422,
        )
    return bits[:requested_len].bin, requested_len


def bits_to_payload(bits: ConstBitStream, payload_bits_len: int) -> tuple[str, int]:
    payload_bit_string, payload_bit_length = bits_to_payload_bits(bits, payload_bits_len)
    if payload_bit_length % 8 != 0:
        raise ZGLSError(
            f"payload is not byte-aligned: {payload_bit_length} bits",
            status_code=422,
        )

    payload_bits = ConstBitStream(bin=payload_bit_string)
    bs = BitStream(payload_bits)
    buf = bytearray()
    while bs.pos < len(bs):
        buf.append(bs.read("uint:8"))
    try:
        text = bytes(buf).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ZGLSError(f"payload utf-8 decode failed: {e}", status_code=400) from e
    return text, payload_bit_length // 8


def framed_bits_to_payload(bits: ConstBitStream) -> tuple[str, int]:
    payload_bit_string, payload_bit_length = framed_bits_to_payload_bits(bits)
    if payload_bit_length % 8 != 0:
        raise ZGLSError(
            f"framed payload is not byte-aligned: {payload_bit_length} bits",
            status_code=422,
        )

    payload_bits = ConstBitStream(bin=payload_bit_string)
    bs = BitStream(payload_bits)
    buf = bytearray()
    while bs.pos < len(bs):
        buf.append(bs.read("uint:8"))
    try:
        text = bytes(buf).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ZGLSError(f"payload utf-8 decode failed: {e}", status_code=400) from e
    return text, payload_bit_length // 8


def repetition_metrics(text: str) -> dict[str, float | int]:
    words = WORD_RE.findall(text.lower())
    structural_artifacts = STRUCTURAL_ARTIFACT_RE.findall(text)
    prompt_echoes = PROMPT_ECHO_RE.findall(text)
    meta_artifacts = META_ARTIFACT_RE.findall(text)
    malformed_punctuation = MALFORMED_PUNCT_RE.findall(text)
    mojibake_artifacts = MOJIBAKE_RE.findall(text)
    unbalanced_quote_count = text.count('"') % 2
    if not words:
        return {
            "word_count": 0,
            "repetition_ratio": 1.0,
            "single_token_share": 1.0,
            "max_bigram_repeat": 0,
            "structural_artifact_count": len(structural_artifacts),
            "prompt_echo_count": len(prompt_echoes),
            "meta_artifact_count": len(meta_artifacts),
            "malformed_punctuation_count": len(malformed_punctuation),
            "mojibake_artifact_count": len(mojibake_artifacts),
            "unbalanced_quote_count": unbalanced_quote_count,
            "terminal_punctuation": False,
            "has_non_ascii": int(not text.isascii()),
        }
    counts = Counter(words)
    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    bigram_counts = Counter(bigrams)
    return {
        "word_count": len(words),
        "repetition_ratio": 1.0 - (len(counts) / len(words)),
        "single_token_share": max(counts.values()) / len(words),
        "max_bigram_repeat": max(bigram_counts.values()) if bigram_counts else 0,
        "structural_artifact_count": len(structural_artifacts),
        "prompt_echo_count": len(prompt_echoes),
        "meta_artifact_count": len(meta_artifacts),
        "malformed_punctuation_count": len(malformed_punctuation),
        "mojibake_artifact_count": len(mojibake_artifacts),
        "unbalanced_quote_count": unbalanced_quote_count,
        "terminal_punctuation": bool(TERMINAL_PUNCT_RE.search(text)),
        "has_non_ascii": int(not text.isascii()),
    }


def quality_pass(
    metrics: dict[str, Any],
    *,
    min_words: int = 8,
    max_words: int = 40,
    max_repetition_ratio: float,
    max_single_token_share: float,
) -> bool:
    return (
        metrics["word_count"] >= min_words
        and metrics["word_count"] <= max_words
        and metrics["repetition_ratio"] <= max_repetition_ratio
        and metrics["single_token_share"] <= max_single_token_share
        and metrics["max_bigram_repeat"] <= 1
        and metrics["structural_artifact_count"] == 0
        and metrics["prompt_echo_count"] == 0
        and metrics["meta_artifact_count"] == 0
        and metrics["malformed_punctuation_count"] == 0
        and metrics["mojibake_artifact_count"] == 0
        and metrics["unbalanced_quote_count"] == 0
        and metrics["has_non_ascii"] == 0
        and metrics["terminal_punctuation"]
    )


def deterministic_payload_bits(*, length: int, seed: int, label: str = "") -> str:
    if length <= 0:
        raise ZGLSError("payload bit candidate must be positive", status_code=400)
    rng_seed = int(hashlib.sha1(f"{seed}:{label}:{length}".encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(rng_seed)
    return "".join("1" if rng.randrange(2) else "0" for _ in range(length))


def best_capacity_success(trials: Iterable[CapacityTrialResult]) -> CapacityTrialResult | None:
    successes = [trial for trial in trials if trial.success]
    if not successes:
        return None
    return max(successes, key=lambda trial: (trial.payload_bits_exact, -trial.word_count))


def trim_to_first_sentence_ids(
    *,
    model: Any,
    prompt_len: int,
    min_end: int,
    out_ids: torch.Tensor,
    min_words: int = 8,
) -> torch.Tensor:
    assert out_ids.dim() == 2 and out_ids.size(0) == 1
    full_ids = out_ids[0].tolist()
    if len(full_ids) <= prompt_len:
        return out_ids
    best_end: int | None = None
    start_end = max(prompt_len + 1, min_end)
    for end in range(start_end, len(full_ids) + 1):
        suffix = model.detokenize(full_ids[prompt_len:end])
        metrics = repetition_metrics(suffix)
        if metrics["word_count"] >= min_words and metrics["terminal_punctuation"]:
            best_end = end
            break
    if best_end is None:
        return out_ids
    return out_ids[:, :best_end]


def bootstrap_ci(
    values: Iterable[float],
    *,
    iters: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stddev": None,
            "min": None,
            "max": None,
            "ci_low": None,
            "ci_high": None,
            "ci_half_width": None,
            "relative_ci_half_width": None,
        }

    rng = random.Random(seed)
    means = []
    for _ in range(max(1, iters)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(statistics.fmean(sample))
    means.sort()
    alpha = 1.0 - confidence
    low_idx = min(len(means) - 1, max(0, int((alpha / 2) * len(means))))
    high_idx = min(len(means) - 1, max(0, int((1 - alpha / 2) * len(means)) - 1))
    mean = statistics.fmean(vals)
    ci_low = means[low_idx]
    ci_high = means[high_idx]
    half_width = (ci_high - ci_low) / 2
    return {
        "count": len(vals),
        "mean": mean,
        "median": statistics.median(vals),
        "stddev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half_width": half_width,
        "relative_ci_half_width": abs(half_width / mean) if mean else None,
    }


def _token_counts_from_texts(texts: Iterable[str], tokenizer: Any | None = None) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        if tokenizer is not None:
            ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
            counts.update(str(i) for i in ids)
        else:
            counts.update(WORD_RE.findall(str(text).lower()))
    return counts


def _token_docs(texts: Iterable[str], tokenizer: Any | None = None) -> list[list[str]]:
    docs: list[list[str]] = []
    for text in texts:
        if tokenizer is not None:
            ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
            docs.append([str(i) for i in ids])
        else:
            docs.append(WORD_RE.findall(str(text).lower()))
    return docs


def _counts_from_docs(docs: Iterable[list[str]]) -> Counter:
    counts: Counter = Counter()
    for doc in docs:
        counts.update(doc)
    return counts


def kld_from_counts(p_counts: Counter, q_counts: Counter, *, alpha: float = 1e-6) -> float:
    vocab = set(p_counts) | set(q_counts)
    if not vocab:
        return 0.0
    p_total = sum(p_counts.values()) + alpha * len(vocab)
    q_total = sum(q_counts.values()) + alpha * len(vocab)
    total = 0.0
    for key in vocab:
        p = (p_counts.get(key, 0) + alpha) / p_total
        q = (q_counts.get(key, 0) + alpha) / q_total
        total += p * math.log(p / q)
    return total


def compute_kld(
    stego_texts: Iterable[str],
    cover_texts: Iterable[str],
    *,
    tokenizer: Any | None = None,
    alpha: float = 1e-6,
) -> dict[str, float]:
    stego_counts = _token_counts_from_texts(stego_texts, tokenizer)
    cover_counts = _token_counts_from_texts(cover_texts, tokenizer)
    stego_to_cover = kld_from_counts(stego_counts, cover_counts, alpha=alpha)
    cover_to_stego = kld_from_counts(cover_counts, stego_counts, alpha=alpha)
    return {
        "kld_stego_to_cover": stego_to_cover,
        "kld_cover_to_stego": cover_to_stego,
        "kld_symmetric": (stego_to_cover + cover_to_stego) / 2,
    }


def bootstrap_kld_ci(
    stego_texts: list[str],
    cover_texts: list[str],
    *,
    tokenizer: Any | None = None,
    alpha: float = 1e-6,
    iters: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | None]:
    if not stego_texts or not cover_texts:
        return {
            "count": 0,
            "kld_stego_to_cover": None,
            "kld_cover_to_stego": None,
            "kld_symmetric": None,
            "ci_low": None,
            "ci_high": None,
            "ci_half_width": None,
            "relative_ci_half_width": None,
        }

    stego_docs = _token_docs(stego_texts, tokenizer)
    cover_docs = _token_docs(cover_texts, tokenizer)
    stego_counts = _counts_from_docs(stego_docs)
    cover_counts = _counts_from_docs(cover_docs)
    stego_to_cover = kld_from_counts(stego_counts, cover_counts, alpha=alpha)
    cover_to_stego = kld_from_counts(cover_counts, stego_counts, alpha=alpha)
    point = {
        "kld_stego_to_cover": stego_to_cover,
        "kld_cover_to_stego": cover_to_stego,
        "kld_symmetric": (stego_to_cover + cover_to_stego) / 2,
    }
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(max(1, iters)):
        stego_sample = [stego_docs[rng.randrange(len(stego_docs))] for _ in stego_docs]
        cover_sample = [cover_docs[rng.randrange(len(cover_docs))] for _ in cover_docs]
        sample_stego_counts = _counts_from_docs(stego_sample)
        sample_cover_counts = _counts_from_docs(cover_sample)
        a_to_b = kld_from_counts(sample_stego_counts, sample_cover_counts, alpha=alpha)
        b_to_a = kld_from_counts(sample_cover_counts, sample_stego_counts, alpha=alpha)
        estimates.append((a_to_b + b_to_a) / 2)
    estimates.sort()
    alpha_tail = 1.0 - confidence
    low_idx = min(len(estimates) - 1, max(0, int((alpha_tail / 2) * len(estimates))))
    high_idx = min(len(estimates) - 1, max(0, int((1 - alpha_tail / 2) * len(estimates)) - 1))
    ci_low = estimates[low_idx]
    ci_high = estimates[high_idx]
    half_width = (ci_high - ci_low) / 2
    mean = point["kld_symmetric"]
    return {
        "count": len(stego_texts),
        **point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half_width": half_width,
        "relative_ci_half_width": abs(half_width / mean) if mean else None,
    }


def compute_metrics(
    samples: list[dict[str, Any]],
    *,
    cover_texts: list[str] | None = None,
    tokenizer: Any | None = None,
    bootstrap_iters: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    stopping_reason: str | None = None,
) -> MetricsResult:
    successes = [s for s in samples if s.get("success")]
    failures = len(samples) - len(successes)
    ppl_summary = bootstrap_ci(
        [float(s["ppl"]) for s in successes if s.get("ppl") is not None],
        iters=bootstrap_iters,
        confidence=confidence,
        seed=seed,
    )

    if cover_texts and successes:
        kld_summary = bootstrap_kld_ci(
            [str(s.get("stegotext", "")) for s in successes],
            cover_texts,
            tokenizer=tokenizer,
            iters=bootstrap_iters,
            confidence=confidence,
            seed=seed,
        )
    else:
        kld_summary = {
            "count": 0,
            "kld_stego_to_cover": None,
            "kld_cover_to_stego": None,
            "kld_symmetric": None,
            "ci_low": None,
            "ci_high": None,
            "relative_ci_half_width": None,
            "ci_half_width": None,
        }

    return MetricsResult(
        sample_count=len(samples),
        success_count=len(successes),
        failure_count=failures,
        failure_rate=(failures / len(samples)) if samples else 0.0,
        ppl=ppl_summary,
        kld=kld_summary,
        stopping_reason=stopping_reason,
    )


class ZGLSClient:
    """Reusable Python client for the llama-server backed ZGLS pipeline."""

    def __init__(self, cfg: ZGLSConfig):
        if cfg.backend != "llama_server":
            raise ZGLSError("Only llama_server backend is supported.")
        self.cfg = cfg
        self.cover_text = load_cover_text(cfg.cover_path, cfg.cover_col) if cfg.cover_path else []
        self.system_template, self.user_template = load_prompt_templates(cfg.prompt_cfg_path)
        self.model = LlamaServerModel(
            LlamaServerConfig(base_url=cfg.server_url, model=cfg.server_model, seed=cfg.seed)
        )
        self.tokenizer = LlamaServerTokenizer(self.model)
        self.model.eval()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": self.cfg.backend,
            "model": self.cfg.server_model,
            "seed": self.cfg.seed,
            "prompt_cfg_path": self.cfg.prompt_cfg_path,
            "generation_quality_status": "experimental",
            "prompt_mode": "request_prompt_is_complete_context",
            "cover_size": len(self.cover_text),
            "cover_source": ("file" if self.cfg.cover_path else "none"),
            "cover_source_note": "Cover files are only used by sampling/metrics helpers; hide/reveal use the request prompt as the complete model prompt.",
            "n_cover": self.cfg.n_cover,
            "header_bits": HEADER_BITS,
            "max_payload_bits": MAX_PAYLOAD_BITS,
            "max_payload_bytes": MAX_PAYLOAD_BYTES,
            "quality_min_words": self.cfg.quality_min_words,
            "quality_max_words": self.cfg.quality_max_words,
            "quality_max_retries": self.cfg.quality_max_retries,
            "max_new_tokens": self.cfg.max_new_tokens,
        }

    def _active_cover(self, cover_texts: list[str] | None) -> list[str]:
        request_cover = [x.strip() for x in (cover_texts or []) if x and x.strip()]
        active = request_cover if request_cover else self.cover_text
        if not active:
            raise ZGLSError(
                "no cover text available; pass cover_texts in request or start server with --cover-path",
                status_code=400,
            )
        return active

    def _effective_prompt(
        self,
        *,
        prompt: str,
    ) -> str:
        effective_prompt = prompt.strip()
        if not effective_prompt:
            raise ZGLSError("prompt must not be empty", status_code=400)
        return effective_prompt

    @torch.no_grad()
    def hide(
        self,
        *,
        prompt: str,
        secret: str | None = None,
        payload_base64: str | None = None,
        payload_bits: str | None = None,
        complete_sent: bool = False,
        cover_texts: list[str] | None = None,
        corpus: str | None = None,
        request_id: str | None = None,
        seed_offset: int | None = None,
        max_new_tokens: int | None = None,
        quality_max_retries: int | None = None,
        quality_max_words: int | None = None,
        threshold: float | None = None,
        temperature: float | None = None,
        temperature_alpha: float | None = None,
        max_bpw: int | None = None,
        allow_partial: bool = False,
        enforce_quality: bool = True,
    ) -> HideResult:
        bits, payload_bytes, normalized_payload_base64 = hide_payload_to_bits(
            secret=secret,
            payload_base64=payload_base64,
            payload_bits=payload_bits,
        )
        target_bits = int(len(bits))
        payload_bits_len = target_bits
        egs_params = replace(
            self.cfg.egs,
            threshold=self.cfg.egs.threshold if threshold is None else float(threshold),
            temperature=self.cfg.egs.temperature if temperature is None else float(temperature),
            temperature_alpha=(
                self.cfg.egs.temperature_alpha
                if temperature_alpha is None
                else float(temperature_alpha)
            ),
            max_bpw=self.cfg.egs.max_bpw if max_bpw is None else int(max_bpw),
        )
        effective_max_new_tokens = (
            self.cfg.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        )
        effective_quality_max_retries = (
            self.cfg.quality_max_retries
            if quality_max_retries is None
            else int(quality_max_retries)
        )
        configured_max_words = (
            self.cfg.quality_max_words if quality_max_words is None else int(quality_max_words)
        )
        last_fail_reason = "unknown"
        best_result: HideResult | None = None
        if seed_offset is None:
            request_id = request_id or str(time.time_ns())
            seed_offset = int(hashlib.sha1(request_id.encode("utf-8")).hexdigest()[:8], 16)

        for retry in range(effective_quality_max_retries + 1):
            context_seed = self.cfg.seed + int(seed_offset) + retry
            effective_prompt = self._effective_prompt(prompt=prompt)
            prompt_ids = self.tokenizer(effective_prompt, return_tensors="pt").input_ids.to(
                self.model.device
            )
            out_ids, is_truncated, used_bits, ppl = hide_extract.hide_bits_with_prompt_ids_by_egs(
                model=self.model,
                prompt_ids=prompt_ids,
                bits=bits,
                mode=egs_params.mode,
                threshold=egs_params.threshold,
                temperature=egs_params.temperature,
                temperature_alpha=egs_params.temperature_alpha,
                max_bpw=egs_params.max_bpw,
                max_new_tokens=effective_max_new_tokens,
                complete_sent=False,
            )
            encoded_out_ids = out_ids
            if complete_sent:
                prompt_len = prompt_ids.shape[1]
                out_ids = self.model.complete_tokens(
                    out_ids,
                    max_tokens=96,
                    temperature=egs_params.temperature,
                )
                out_ids = trim_to_first_sentence_ids(
                    model=self.model,
                    prompt_len=prompt_len,
                    min_end=encoded_out_ids.shape[1],
                    out_ids=out_ids,
                )

            prompt_len = prompt_ids.size(1)
            display_token_ids = out_ids[0, prompt_len:].tolist()
            stego_token_ids = encoded_out_ids[0, prompt_len:].tolist()
            stegotext = self.tokenizer.decode(display_token_ids)
            metrics = repetition_metrics(stegotext)
            embedded_bits = min(int(used_bits), target_bits)
            remaining_bits = bits[embedded_bits:target_bits].bin
            decode_ready = embedded_bits >= target_bits
            quality_ok = quality_pass(
                metrics,
                min_words=self.cfg.quality_min_words,
                max_words=configured_max_words,
                max_repetition_ratio=self.cfg.quality_max_repetition_ratio,
                max_single_token_share=self.cfg.quality_max_single_token_share,
            )
            passed = (
                (not bool(is_truncated))
                and decode_ready
                and (quality_ok or not enforce_quality)
            )
            bpw_estimate = int(used_bits) / max(1, int(metrics["word_count"]))
            candidate = HideResult(
                stegotext=stegotext,
                stego_token_ids=stego_token_ids,
                context_seed=context_seed,
                effective_prompt_hash=prompt_hash(effective_prompt),
                used_bits=int(used_bits),
                target_bits=target_bits,
                bpw_estimate=float(bpw_estimate),
                payload_bytes=payload_bytes,
                payload_bits=payload_bits_len,
                header_bits=HEADER_BITS,
                total_target_bits=target_bits,
                total_used_bits=int(used_bits),
                is_truncated=bool(is_truncated),
                ppl=float(ppl),
                quality_passed=passed,
                quality_metrics={
                    **metrics,
                    "decode_ready": decode_ready,
                    "retry": retry,
                    "quality_min_words": self.cfg.quality_min_words,
                    "quality_max_words": configured_max_words,
                    "payload_bits": payload_bits_len,
                    "header_bits": HEADER_BITS,
                    "total_target_bits": target_bits,
                },
                mode=egs_params.mode,
                params_used={
                    **asdict(egs_params),
                    "max_new_tokens": effective_max_new_tokens,
                    "quality_max_retries": effective_quality_max_retries,
                    "quality_min_words": self.cfg.quality_min_words,
                    "quality_max_words": configured_max_words,
                    "allow_partial": allow_partial,
                    "enforce_quality": enforce_quality,
                },
                embedded_bits=embedded_bits,
                fully_embedded=decode_ready,
                remaining_bits_len=len(remaining_bits),
                remaining_bits=remaining_bits,
                payload_base64=normalized_payload_base64,
                remaining_payload_base64=bit_string_to_base64_bytes(remaining_bits),
            )
            best_result = candidate
            if passed:
                return candidate
            if allow_partial and (is_truncated or not decode_ready):
                return candidate
            if decode_ready and not enforce_quality:
                return candidate
            if is_truncated:
                last_fail_reason = "truncated"
            elif not decode_ready:
                last_fail_reason = "insufficient_used_bits"
            else:
                last_fail_reason = "quality_gate_failed"

        raise ZGLSError(
            "quality_gate_failed",
            status_code=422,
            detail={
                "reason": "quality_gate_failed",
                "last_fail_reason": last_fail_reason,
                "failure_reason": last_fail_reason,
                "max_retries": effective_quality_max_retries,
                "payload_bits": payload_bits_len,
                "header_bits": HEADER_BITS,
                "total_target_bits": target_bits,
                "target_bits": target_bits,
                "payload_bytes": payload_bytes,
                "best_metrics": (best_result.quality_metrics if best_result else None),
                "best_candidate": (asdict(best_result) if best_result else None),
            },
        )

    @torch.no_grad()
    def capacity_probe(
        self,
        *,
        prompt: str,
        payload_bits: str,
        max_words: int = 40,
        payload_bits_candidates: list[int] | None = None,
        initial_payload_bits: int | None = None,
        complete_sent: bool = False,
        request_id: str | None = None,
        payload_seed: int | None = None,
        max_new_tokens: int | None = None,
        quality_max_retries: int | None = None,
        quality_max_words: int | None = None,
        threshold: float | None = None,
        temperature: float | None = None,
        temperature_alpha: float | None = None,
        max_bpw: int | None = None,
    ) -> CapacityProbeResult:
        probe_bits = validate_payload_bits(payload_bits)
        probe_bit_count = len(probe_bits)
        initial_candidate_bits = (
            probe_bit_count if initial_payload_bits is None else int(initial_payload_bits)
        )
        if initial_candidate_bits <= 0:
            raise ZGLSError("initial payload bits must be positive", status_code=400)
        if initial_candidate_bits > probe_bit_count:
            raise ZGLSError(
                f"initial payload bits must be <= supplied payload bitstream length ({probe_bit_count})",
                status_code=400,
            )
        candidates = (
            [
                bits
                for bits in (
                    [
                        initial_candidate_bits // (2**idx)
                        for idx in range(initial_candidate_bits.bit_length())
                        if initial_candidate_bits // (2**idx) > 0
                    ]
                    + [
                        min(probe_bit_count, initial_candidate_bits * (2**idx))
                        for idx in range(1, probe_bit_count.bit_length() + 1)
                        if initial_candidate_bits * (2**idx) <= probe_bit_count
                    ]
                )
                if bits > 0
            ]
            if payload_bits_candidates is None
            else payload_bits_candidates
        )
        candidates = list(dict.fromkeys(int(x) for x in candidates))
        if not candidates:
            raise ZGLSError("payload bit candidates must not be empty", status_code=400)
        if any(x <= 0 for x in candidates):
            raise ZGLSError("payload bit candidates must be positive", status_code=400)
        if any(x > MAX_PAYLOAD_BITS for x in candidates):
            raise ZGLSError(
                f"payload bit candidates must be <= {MAX_PAYLOAD_BITS}",
                status_code=400,
            )
        if any(x > probe_bit_count for x in candidates):
            raise ZGLSError(
                f"payload bit candidates must be <= supplied payload bitstream length ({probe_bit_count})",
                status_code=400,
            )

        hard_max_words = int(max_words)
        if hard_max_words <= 0:
            raise ZGLSError("max_words must be positive", status_code=400)
        gate_max_words = hard_max_words if quality_max_words is None else min(
            int(quality_max_words),
            hard_max_words,
        )
        egs_params = replace(
            self.cfg.egs,
            threshold=self.cfg.egs.threshold if threshold is None else float(threshold),
            temperature=self.cfg.egs.temperature if temperature is None else float(temperature),
            temperature_alpha=(
                self.cfg.egs.temperature_alpha
                if temperature_alpha is None
                else float(temperature_alpha)
            ),
            max_bpw=self.cfg.egs.max_bpw if max_bpw is None else int(max_bpw),
        )
        effective_max_new_tokens = (
            self.cfg.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        )
        effective_quality_max_retries = (
            self.cfg.quality_max_retries
            if quality_max_retries is None
            else int(quality_max_retries)
        )
        effective_prompt = self._effective_prompt(prompt=prompt)
        effective_hash = prompt_hash(effective_prompt)
        seed_base = self.cfg.seed if payload_seed is None else int(payload_seed)
        request_id = request_id or f"capacity:{effective_hash}:{seed_base}"
        request_seed_offset = int(hashlib.sha1(request_id.encode("utf-8")).hexdigest()[:8], 16)
        params_used = {
            **asdict(egs_params),
            "max_new_tokens": effective_max_new_tokens,
            "quality_max_retries": effective_quality_max_retries,
            "quality_min_words": self.cfg.quality_min_words,
            "quality_max_words": gate_max_words,
            "max_words": hard_max_words,
            "initial_payload_bits": initial_candidate_bits,
            "payload_seed": seed_base,
        }

        trials: list[CapacityTrialResult] = []
        for candidate_bits in candidates:
            candidate_payload_bits = probe_bits[:candidate_bits]
            raw_bits = payload_bits_to_bits(candidate_payload_bits)
            target_bits = len(raw_bits)
            best_trial: CapacityTrialResult | None = None

            for retry in range(effective_quality_max_retries + 1):
                context_seed = self.cfg.seed + request_seed_offset + candidate_bits * 1009 + retry
                prompt_ids = self.tokenizer(effective_prompt, return_tensors="pt").input_ids.to(
                    self.model.device
                )
                out_ids, is_truncated, used_bits, ppl = hide_extract.hide_bits_with_prompt_ids_by_egs(
                    model=self.model,
                    prompt_ids=prompt_ids,
                    bits=raw_bits,
                    mode=egs_params.mode,
                    threshold=egs_params.threshold,
                    temperature=egs_params.temperature,
                    temperature_alpha=egs_params.temperature_alpha,
                    max_bpw=egs_params.max_bpw,
                    max_new_tokens=effective_max_new_tokens,
                    complete_sent=False,
                )
                encoded_out_ids = out_ids
                if complete_sent:
                    prompt_len = prompt_ids.shape[1]
                    out_ids = self.model.complete_tokens(
                        out_ids,
                        max_tokens=96,
                        temperature=egs_params.temperature,
                    )
                    out_ids = trim_to_first_sentence_ids(
                        model=self.model,
                        prompt_len=prompt_len,
                        min_end=encoded_out_ids.shape[1],
                        out_ids=out_ids,
                    )

                prompt_len = prompt_ids.size(1)
                display_token_ids = out_ids[0, prompt_len:].tolist()
                stego_token_ids = encoded_out_ids[0, prompt_len:].tolist()
                stegotext = self.tokenizer.decode(display_token_ids)
                metrics = repetition_metrics(stegotext)
                word_count = int(metrics["word_count"])
                decode_ready = int(used_bits) >= target_bits
                quality_ok = (
                    not bool(is_truncated)
                    and decode_ready
                    and word_count <= hard_max_words
                    and quality_pass(
                        metrics,
                        min_words=self.cfg.quality_min_words,
                        max_words=gate_max_words,
                        max_repetition_ratio=self.cfg.quality_max_repetition_ratio,
                        max_single_token_share=self.cfg.quality_max_single_token_share,
                    )
                )

                warnings: list[str] = []
                decode_ok = False
                secret_matches = False
                if decode_ready:
                    hide_ids = torch.tensor(
                        [prompt_ids[0].tolist() + stego_token_ids],
                        dtype=torch.long,
                    ).to(self.model.device)
                    extracted_bits, is_succeed = hide_extract.extract_bits_with_prompt_ids_by_egs(
                        model=self.model,
                        prompt_ids=prompt_ids,
                        hide_ids=hide_ids,
                        mode=egs_params.mode,
                        threshold=egs_params.threshold,
                        temperature=egs_params.temperature,
                        temperature_alpha=egs_params.temperature_alpha,
                        max_bpw=egs_params.max_bpw,
                    )
                    if is_succeed:
                        try:
                            recovered_bits, recovered_len = bits_to_payload_bits(
                                ConstBitStream(extracted_bits),
                                candidate_bits,
                            )
                            decode_ok = recovered_len == candidate_bits
                            secret_matches = decode_ok and recovered_bits == candidate_payload_bits
                        except ZGLSError as exc:
                            warnings.append(str(exc.detail))
                    else:
                        warnings.append("extractor could not follow stegotext path completely")

                success = bool(quality_ok and decode_ok and secret_matches)
                if success:
                    failure_reason = None
                elif is_truncated:
                    failure_reason = "truncated"
                elif not decode_ready:
                    failure_reason = "insufficient_used_bits"
                elif word_count > hard_max_words:
                    failure_reason = "word_limit_exceeded"
                elif not quality_ok:
                    failure_reason = "quality_gate_failed"
                elif not decode_ok:
                    failure_reason = "decode_failed"
                else:
                    failure_reason = "payload_mismatch"

                trial = CapacityTrialResult(
                    payload_bits=candidate_bits,
                    payload_bytes=math.ceil(candidate_bits / 8),
                    payload_bits_exact=candidate_bits,
                    header_bits=HEADER_BITS,
                    total_target_bits=target_bits,
                    total_used_bits=int(used_bits),
                    used_bits=int(used_bits),
                    success=success,
                    decode_ok=decode_ok,
                    secret_matches=secret_matches,
                    quality_passed=quality_ok,
                    word_count=word_count,
                    stegotext=stegotext,
                    stego_token_ids=stego_token_ids,
                    context_seed=context_seed,
                    effective_prompt_hash=effective_hash,
                    is_truncated=bool(is_truncated),
                    ppl=float(ppl),
                    quality_metrics={
                        **metrics,
                        "decode_ready": decode_ready,
                        "retry": retry,
                        "quality_min_words": self.cfg.quality_min_words,
                        "quality_max_words": gate_max_words,
                        "max_words": hard_max_words,
                        "payload_bits": candidate_bits,
                        "header_bits": HEADER_BITS,
                        "total_target_bits": target_bits,
                    },
                    failure_reason=failure_reason,
                    warnings=warnings,
                    mode=egs_params.mode,
                    params_used=params_used,
                )
                best_trial = trial
                if success:
                    break

            assert best_trial is not None
            trials.append(best_trial)

        return CapacityProbeResult(
            best_success=best_capacity_success(trials),
            trials=trials,
            params_used=params_used,
        )

    @torch.no_grad()
    def reveal(
        self,
        *,
        prompt: str,
        stegotext: str,
        context_seed: int | None = None,
        effective_prompt_hash: str | None = None,
        cover_texts: list[str] | None = None,
        corpus: str | None = None,
        stego_token_ids: list[int] | None = None,
        payload_bits_len: int | None = None,
        threshold: float | None = None,
        temperature: float | None = None,
        temperature_alpha: float | None = None,
        max_bpw: int | None = None,
    ) -> RevealResult:
        warnings: list[str] = []
        egs_params = replace(
            self.cfg.egs,
            threshold=self.cfg.egs.threshold if threshold is None else float(threshold),
            temperature=self.cfg.egs.temperature if temperature is None else float(temperature),
            temperature_alpha=(
                self.cfg.egs.temperature_alpha
                if temperature_alpha is None
                else float(temperature_alpha)
            ),
            max_bpw=self.cfg.egs.max_bpw if max_bpw is None else int(max_bpw),
        )
        context_seed = self.cfg.seed if context_seed is None else int(context_seed)
        effective_prompt = self._effective_prompt(prompt=prompt)
        reconstructed_hash = prompt_hash(effective_prompt)
        if effective_prompt_hash and effective_prompt_hash != reconstructed_hash:
            warnings.append("provided effective_prompt_hash does not match reconstructed prompt")

        prompt_ids = self.tokenizer(effective_prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        if stego_token_ids:
            prefix = prompt_ids[0].tolist()
            hide_ids = torch.tensor([prefix + stego_token_ids], dtype=torch.long).to(self.model.device)
            warnings.append("decode used provided stego_token_ids (token-stable path)")
        else:
            warnings.append("decode used retokenized text path; tokenization drift may reduce reliability")
            hide_ids = self.tokenizer(effective_prompt + stegotext, return_tensors="pt").input_ids.to(
                self.model.device
            )

        extracted_bits, is_succeed = hide_extract.extract_bits_with_prompt_ids_by_egs(
            model=self.model,
            prompt_ids=prompt_ids,
            hide_ids=hide_ids,
            mode=egs_params.mode,
            threshold=egs_params.threshold,
            temperature=egs_params.temperature,
            temperature_alpha=egs_params.temperature_alpha,
            max_bpw=egs_params.max_bpw,
        )
        raw_bits_len = int(len(extracted_bits))
        if not is_succeed:
            warnings.append("extractor could not follow stegotext path completely")
            return RevealResult(
                secret=None,
                payload_bytes=None,
                payload_bits=None,
                payload_bits_len=None,
                decode_ok=False,
                raw_bits_len=raw_bits_len,
                warnings=warnings,
                context_seed=context_seed,
                effective_prompt_hash=reconstructed_hash,
                mode=egs_params.mode,
                params_used=asdict(egs_params),
            )

        raw_bits = ConstBitStream(extracted_bits)
        if payload_bits_len is None:
            warnings.append("payload_bits_len omitted; trying legacy 16-bit framed decode")
            payload_bit_string, payload_bits_len = framed_bits_to_payload_bits(raw_bits)
            secret, payload_bytes = framed_bits_to_payload(raw_bits)
        else:
            payload_bit_string, payload_bits_len = bits_to_payload_bits(raw_bits, payload_bits_len)
            secret, payload_bytes = bits_to_payload(raw_bits, payload_bits_len)
        return RevealResult(
            secret=secret,
            payload_bytes=payload_bytes,
            payload_bits=payload_bit_string,
            payload_bits_len=payload_bits_len,
            decode_ok=True,
            raw_bits_len=raw_bits_len,
            warnings=warnings,
            context_seed=context_seed,
            effective_prompt_hash=reconstructed_hash,
            mode=egs_params.mode,
            params_used=asdict(egs_params),
        )

    def generate_samples(self, sample_cfg: SampleConfig) -> SampleRunResult:
        rng = random.Random(sample_cfg.seed)
        samples: list[dict[str, Any]] = []
        stopping_reason = "max_samples"
        if not sample_cfg.prompts:
            raise ZGLSError("SampleConfig.prompts must not be empty", status_code=400)
        if not sample_cfg.secrets:
            raise ZGLSError("SampleConfig.secrets must not be empty", status_code=400)
        kld_cover_source = sample_cfg.cover_texts or self.cover_text
        if sample_cfg.kld_cover_limit > 0 and len(kld_cover_source) > sample_cfg.kld_cover_limit:
            kld_cover_texts = random.Random(sample_cfg.seed).sample(
                kld_cover_source,
                sample_cfg.kld_cover_limit,
            )
        else:
            kld_cover_texts = kld_cover_source
        progress_fp = None
        if sample_cfg.progress_path:
            progress_path = Path(sample_cfg.progress_path)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_fp = progress_path.open("a", encoding="utf-8")

        try:
            for idx in range(sample_cfg.max_samples):
                prompt = sample_cfg.prompts[idx % len(sample_cfg.prompts)]
                secret = sample_cfg.secrets[idx % len(sample_cfg.secrets)]
                seed_offset = rng.randrange(0, 2**31)
                row: dict[str, Any] = {
                    "sample_id": idx,
                    "prompt": prompt,
                    "secret_len_bytes": len(secret.encode("utf-8")),
                    "seed_offset": seed_offset,
                    "success": False,
                    "error_category": None,
                }
                try:
                    hidden = self.hide(
                        prompt=prompt,
                        secret=secret,
                        complete_sent=sample_cfg.complete_sent,
                        cover_texts=sample_cfg.cover_texts,
                        corpus=sample_cfg.corpus,
                        seed_offset=seed_offset,
                    )
                    revealed = self.reveal(
                        prompt=prompt,
                        stegotext=hidden.stegotext,
                        context_seed=hidden.context_seed,
                        effective_prompt_hash=hidden.effective_prompt_hash,
                        cover_texts=sample_cfg.cover_texts,
                        corpus=sample_cfg.corpus,
                        stego_token_ids=hidden.stego_token_ids,
                        payload_bits_len=hidden.payload_bits,
                    )
                    row.update(asdict(hidden))
                    row.update(
                        {
                            "roundtrip_ok": revealed.decode_ok and revealed.secret == secret,
                            "revealed_secret_len_bytes": (
                                len(revealed.secret.encode("utf-8")) if revealed.secret else None
                            ),
                            "reveal_warnings": revealed.warnings,
                        }
                    )
                    row["success"] = bool(row["roundtrip_ok"] and hidden.quality_passed)
                    if not row["success"]:
                        row["error_category"] = "roundtrip_mismatch"
                except ZGLSError as e:
                    row["error_category"] = str(e.detail)
                except Exception as e:  # keep sampling runs from dying on one bad sample
                    row["error_category"] = type(e).__name__
                    row["error_detail"] = str(e)
                samples.append(row)
                if progress_fp is not None:
                    progress_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress_fp.flush()

                if idx + 1 >= sample_cfg.min_samples:
                    metrics = compute_metrics(
                        samples,
                        cover_texts=kld_cover_texts,
                        tokenizer=self.tokenizer,
                        bootstrap_iters=sample_cfg.bootstrap_iters,
                        confidence=sample_cfg.confidence,
                        seed=sample_cfg.seed,
                    )
                    ppl_rel = metrics.ppl.get("relative_ci_half_width")
                    kld_rel = metrics.kld.get("relative_ci_half_width")
                    ppl_ok = ppl_rel is not None and float(ppl_rel) <= sample_cfg.relative_ci_target
                    kld_ok = kld_rel is None or float(kld_rel) <= sample_cfg.relative_ci_target
                    if ppl_ok and kld_ok:
                        stopping_reason = "relative_ci_target"
                        break
        finally:
            if progress_fp is not None:
                progress_fp.close()

        final_metrics = compute_metrics(
            samples,
            cover_texts=kld_cover_texts,
            tokenizer=self.tokenizer,
            bootstrap_iters=sample_cfg.bootstrap_iters,
            confidence=sample_cfg.confidence,
            seed=sample_cfg.seed,
            stopping_reason=stopping_reason,
        )
        return SampleRunResult(samples=samples, metrics=final_metrics)
