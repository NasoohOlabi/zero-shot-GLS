"""HTTP server for the Zero-shot GLS hide/reveal API.

Run:
    python scripts/stego_api_server.py --backend llama_server --host 127.0.0.1 --port 9000
"""

from __future__ import annotations

import argparse
import contextvars
import json
import logging
import os
import os.path as osp
import sys
import time
import uuid
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse, PlainTextResponse, Response

os.environ["HF_HOME"] = f"{osp.dirname(__file__)}/../tmp_saves/hg_cache"
sys.path.append(f"{osp.dirname(osp.abspath(__file__))}/../src")

import hide_extract
from zgls_trace import generate_generation_trace, render_generation_trace_html
from zgls_api import (
    CapacityProbeResult,
    EGSParams,
    HideResult,
    RevealResult,
    ZGLSClient,
    ZGLSConfig,
    ZGLSError,
)


LOGGER = logging.getLogger("stego_api")
REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("stego_req_id", default="unknown")
MAX_LOG_TEXT_LEN = 400


def _safe_preview(text: str, max_len: int = MAX_LOG_TEXT_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [truncated {len(text) - max_len} chars]"


def _parse_json_body(raw: bytes) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else {"_body": data}


def _summarize_body(body: dict | None) -> dict | None:
    if body is None:
        return None

    summarized: dict = {}
    for key, value in body.items():
        summarized[key] = _safe_preview(value) if isinstance(value, str) else value

    for key in ("secret", "stegotext", "prompt"):
        if key in summarized and isinstance(body.get(key), str):
            raw = body[key]
            summarized[key] = {
                "preview": _safe_preview(raw),
                "length_chars": len(raw),
                "length_bytes": len(raw.encode("utf-8")),
            }

    return summarized


class HideRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Complete model prompt/context. Include any examples or samples directly "
            "in this string; the server does not add cover samples."
        ),
        examples=[
            "You are writing natural short movie comments.\n\nExamples:\n- The pacing felt uneven, but the ending stayed with me.\n- The lead performance carried a thin plot.\n\nWrite one new natural short movie comment."
        ],
    )
    secret: str = Field(..., description="Secret text to hide (UTF-8).")
    complete_sent: bool = Field(default=False, description="Whether to auto-complete sentence end.")
    max_new_tokens: int | None = Field(
        default=None,
        ge=1,
        le=4096,
        description="Optional per-request generation token budget override.",
    )
    quality_max_retries: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional per-request quality retry budget override.",
    )
    quality_max_words: int | None = Field(
        default=None,
        ge=1,
        le=2000,
        description=(
            "Optional per-request visible word ceiling. This is a hard acceptance gate."
        ),
    )
    threshold: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Optional per-request EGS threshold override.",
    )
    temperature: float | None = Field(
        default=None,
        gt=0,
        le=5,
        description="Optional per-request generation temperature override.",
    )
    temperature_alpha: float | None = Field(
        default=None,
        gt=0,
        le=10,
        description="Optional per-request temperature alpha override.",
    )
    max_bpw: int | None = Field(
        default=None,
        ge=1,
        le=16,
        description="Optional per-request maximum bits per generated token.",
    )


class HideResponse(BaseModel):
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
    quality_metrics: dict
    mode: str
    params_used: dict


class RevealRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Exact same complete prompt/context string used for /hide.",
    )
    stegotext: str = Field(..., min_length=1, description="Generated stegotext.")
    context_seed: int | None = Field(
        default=None,
        description="Legacy replay metadata returned by /hide; prompt construction no longer depends on it.",
    )
    effective_prompt_hash: str | None = Field(
        default=None,
        description="Hash of the exact effective prompt returned by /hide; used only as a mismatch check.",
    )
    stego_token_ids: list[int] | None = Field(
        default=None,
        description="Optional exact token ids from /hide for stable decode.",
    )
    payload_bits_len: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Raw payload bit length returned by /hide as payload_bits. "
            "Required for new headerless stegotext; omitted requests use legacy framed decode."
        ),
    )
    threshold: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Optional EGS threshold override matching /hide.",
    )
    temperature: float | None = Field(
        default=None,
        gt=0,
        le=5,
        description="Optional temperature override matching /hide.",
    )
    temperature_alpha: float | None = Field(
        default=None,
        gt=0,
        le=10,
        description="Optional temperature alpha override matching /hide.",
    )
    max_bpw: int | None = Field(
        default=None,
        ge=1,
        le=16,
        description="Optional max bits per token override matching /hide.",
    )


class RevealResponse(BaseModel):
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
    params_used: dict


class CapacityProbeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Complete prompt with examples/context.")
    payload_bits: str = Field(
        ...,
        min_length=1,
        description="Caller-supplied raw bitstream. Capacity trials use prefixes of this exact stream.",
        pattern="^[01]+$",
    )
    max_words: int = Field(default=40, ge=1, le=2000, description="Hard visible word limit.")
    quality_max_words: int | None = Field(
        default=None,
        ge=1,
        le=2000,
        description="Optional quality gate word ceiling; capped at max_words.",
    )
    quality_max_retries: int | None = Field(default=None, ge=0, le=100)
    payload_bits_candidates: list[int] | None = Field(
        default=None,
        description=(
            "Optional prefix lengths to test against payload_bits. "
            "When omitted, probes start at the full bitstream length and halve downward."
        ),
    )
    initial_payload_bits: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional starting prefix length. Without explicit candidates, the probe tries n, "
            "then smaller halves and larger doublings within payload_bits."
        ),
    )
    payload_seed: int | None = Field(
        default=None,
        description="Seed for deterministic probe trial ordering/context; payload bits come from payload_bits.",
    )
    complete_sent: bool = Field(default=True)
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)
    threshold: float | None = Field(default=None, gt=0, lt=1)
    temperature: float | None = Field(default=None, gt=0, le=5)
    temperature_alpha: float | None = Field(default=None, gt=0, le=10)
    max_bpw: int | None = Field(default=None, ge=1, le=16)


class CapacityTrialResponse(BaseModel):
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
    quality_metrics: dict
    failure_reason: str | None
    warnings: list[str]
    mode: str
    params_used: dict


class CapacityProbeResponse(BaseModel):
    best_success: CapacityTrialResponse | None
    trials: list[CapacityTrialResponse]
    params_used: dict


class GenerationTraceRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Complete prompt/context to trace.",
        examples=[
            "/no_think\nWrite exactly one short, natural movie-review sentence.\nOutput only the sentence.\nExamples:\nThe acting was uneven, but the ending still worked.\nI liked the premise more than the execution.\nNew sentence:"
        ],
    )
    secret: str = Field(..., min_length=1, description="Secret text to hide and trace as UTF-8 bits.")
    threshold: float | None = Field(default=None, gt=0, lt=1)
    temperature: float | None = Field(default=None, gt=0, le=5)
    temperature_alpha: float | None = Field(default=None, gt=0, le=10)
    max_bpw: int | None = Field(default=None, ge=1, le=16)
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)
    max_steps: int = Field(default=64, ge=1, le=512)
    candidates_shown: int | None = Field(
        default=None,
        ge=1,
        le=10000,
        description="Optional cap on shown post-threshold candidates per step. Omit for full pruned distribution.",
    )


def _http_error(e: ZGLSError) -> HTTPException:
    return HTTPException(status_code=e.status_code, detail=e.detail)


def _model_schema(model: type[BaseModel]) -> dict:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def _help_payload(cfg: ZGLSConfig, client: ZGLSClient) -> dict:
    base_url = "http://<server-ip>:9000"
    return {
        "name": "Zero-shot GLS API",
        "version": "0.3.0",
        "summary": (
            "Hide and reveal UTF-8 secrets using the shared ZGLS Python API and a "
            "llama-server backend with token-level top-logprobs."
        ),
        "base_urls": {
            "same_machine": "http://127.0.0.1:9000",
            "lan_template": base_url,
        },
        "runtime": client.health(),
        "important_notes": [
            "Use the stego_token_ids returned by /hide when calling /reveal; this is the token-stable path.",
            "The prompt is the complete model context. Put all paper-style samples/examples directly in prompt.",
            "The server does not add, sample, or retrieve cover text for /hide or /reveal.",
            "Use the exact same prompt and EGS parameters for /reveal. effective_prompt_hash is a mismatch check, not a secret.",
            "quality_passed=true means the current server quality gate accepted the visible stegotext.",
            "Capacity is raw payload bits recovered under the hard word limit.",
            "New /hide and /capacity_probe outputs do not embed a length header. Pass payload_bits as payload_bits_len to /reveal.",
            "If payload_bits_len is omitted, /reveal tries the older 16-bit framed decode for legacy stegotext.",
        ],
        "quality_gate": {
            "minimum_words": cfg.quality_min_words,
            "default_maximum_words": cfg.quality_max_words,
            "maximum_words_note": (
                "The maximum word count is a hard gate. The server does not silently expand it "
                "for larger payloads."
            ),
            "rejects": [
                "high repetition",
                "repeated bigrams above the gate",
                "structural/control-token artifacts",
                "prompt/instruction echo",
                "meta artifacts",
                "malformed punctuation",
                "mojibake artifacts",
                "unbalanced quotes",
                "non-ASCII visible stegotext",
                "missing terminal punctuation",
                "truncated or under-filled payloads",
            ],
            "configured_limits": {
                "quality_max_retries": cfg.quality_max_retries,
                "quality_min_words": cfg.quality_min_words,
                "quality_max_words": cfg.quality_max_words,
                "max_new_tokens": cfg.max_new_tokens,
                "quality_max_repetition_ratio": cfg.quality_max_repetition_ratio,
                "quality_max_single_token_share": cfg.quality_max_single_token_share,
            },
        },
        "endpoints": {
            "GET /health": {
                "purpose": "Check backend, model, prompt mode, quality settings, and payload metadata.",
                "response": "Runtime metadata dictionary.",
            },
            "GET /help": {
                "purpose": "Return this structured API guide as JSON.",
                "response": "Machine-readable contracts, examples, and notes.",
            },
            "GET /help.md": {
                "purpose": "Return a human-readable Markdown API guide.",
                "response": "text/markdown.",
            },
            "POST /hide": {
                "purpose": "Encode a UTF-8 secret into generated stegotext. For fair capacity measurement use /capacity_probe.",
                "request_model": "HideRequest",
                "response_model": "HideResponse",
                "request_schema": _model_schema(HideRequest),
                "response_schema": _model_schema(HideResponse),
                "example_request": {
                    "prompt": (
                        "You are writing natural short movie comments.\n\n"
                        "Examples:\n"
                        "- The acting was strong but the pacing felt uneven.\n"
                        "- It had a few nice scenes, though the ending felt rushed.\n\n"
                        "Write one new natural short movie comment."
                    ),
                    "secret": "a",
                    "complete_sent": True,
                    "max_new_tokens": 512,
                    "quality_max_retries": 8,
                    "quality_max_words": 120,
                    "max_bpw": 5,
                },
                "example_response_fields": {
                    "stegotext": "Visible generated text carrying the hidden payload.",
                    "stego_token_ids": "Exact encoded token suffix; pass to /reveal for stable decoding.",
                    "context_seed": "Legacy replay metadata; prompt construction does not depend on it.",
                    "effective_prompt_hash": "SHA-1 hash of the exact effective prompt; pass to /reveal to detect prompt mismatches.",
                    "used_bits": "Number of payload bits actually embedded.",
                    "target_bits": "Number of raw payload bits required.",
                    "payload_bits": "Raw UTF-8 payload bits. Pass this back to /reveal as payload_bits_len.",
                    "header_bits": "Compatibility field; new outputs report 0.",
                    "total_target_bits": "Same as payload_bits for new headerless outputs.",
                    "total_used_bits": "Raw payload bits embedded by the encoder.",
                    "bpw_estimate": "Estimated embedded bits per visible word.",
                    "ppl": "Perplexity-style score from the ZGLS path.",
                    "quality_metrics": "Gate diagnostics for repetition, artifacts, punctuation, and encoding quality.",
                    "params_used": "Effective EGS and request override parameters needed for reveal/replay.",
                },
                "common_errors": {
                    "400": "Invalid input such as empty prompt, empty secret, or oversized secret.",
                    "422": "Payload could not be embedded cleanly within truncation/quality constraints.",
                    "500": "Unexpected backend/server failure.",
                },
            },
            "POST /capacity_probe": {
                "purpose": (
                    "Find the largest raw payload bit count that can be encoded, recovered, "
                    "and quality-accepted under a hard visible word limit."
                ),
                "request_model": "CapacityProbeRequest",
                "response_model": "CapacityProbeResponse",
                "request_schema": _model_schema(CapacityProbeRequest),
                "response_schema": _model_schema(CapacityProbeResponse),
                "example_request": {
                    "prompt": (
                        "You are writing natural short movie comments.\n\n"
                        "Examples:\n"
                        "- The acting was strong but the pacing felt uneven.\n"
                        "- It had a few nice scenes, though the ending felt rushed.\n\n"
                        "Write one new natural short movie comment."
                    ),
                    "payload_bits": "1011001110001111000011110000111100001111000011110000111100001111",
                    "max_words": 40,
                    "quality_max_retries": 6,
                    "initial_payload_bits": 32,
                    "payload_bits_candidates": [64, 32, 16, 8],
                    "complete_sent": True,
                    "max_new_tokens": 64,
                    "threshold": 0.005,
                    "temperature": 1.0,
                    "temperature_alpha": 1.25,
                    "max_bpw": 2,
                },
                "example_response_fields": {
                    "best_success": "Largest successful prefix of the supplied bitstream, or null.",
                    "trials": "One structured result per candidate prefix length.",
                    "payload_bits_exact": "Main capacity metric: raw recovered payload bits.",
                    "header_bits": "Compatibility field; new outputs report 0.",
                    "total_target_bits": "Same as payload_bits_exact for new headerless outputs.",
                    "total_used_bits": "Raw payload bits actually embedded by the encoder.",
                },
                "common_errors": {
                    "400": "Invalid prompt, word limit, or payload bit candidates.",
                    "422": "Reserved for malformed extracted data during legacy decode.",
                    "500": "Unexpected backend/server failure.",
                },
            },
            "POST /reveal": {
                "purpose": "Decode a hidden UTF-8 secret from stegotext.",
                "request_model": "RevealRequest",
                "response_model": "RevealResponse",
                "request_schema": _model_schema(RevealRequest),
                "response_schema": _model_schema(RevealResponse),
                "example_request": {
                    "prompt": (
                        "You are writing natural short movie comments.\n\n"
                        "Examples:\n"
                        "- The acting was strong but the pacing felt uneven.\n"
                        "- It had a few nice scenes, though the ending felt rushed.\n\n"
                        "Write one new natural short movie comment."
                    ),
                    "stegotext": "<stegotext from /hide>",
                    "stego_token_ids": ["<token ids from /hide>"],
                    "payload_bits_len": 8,
                    "context_seed": 12345,
                    "effective_prompt_hash": "<hash from /hide>",
                },
                "example_response_fields": {
                    "secret": "Decoded UTF-8 secret, or null on failure.",
                    "decode_ok": "True when extraction and payload decoding succeeded.",
                    "warnings": "Includes prompt-hash mismatch or retokenized-text fallback warnings.",
                    "raw_bits_len": "Number of extracted raw bits observed by the decoder.",
                },
                "common_errors": {
                    "400": "Invalid decoded payload framing or UTF-8.",
                    "422": "Malformed request body or missing required stegotext/prompt.",
                    "500": "Unexpected backend/server failure.",
                },
            },
            "POST /generation_trace.html": {
                "purpose": (
                    "Generate a self-contained HTML explanation of one headerless /hide run, "
                    "showing every token-selection step, pretext, post-threshold distribution, "
                    "Huffman code table, selected token, consumed bits, and selection reason."
                ),
                "request_model": "GenerationTraceRequest",
                "response_model": "text/html",
                "request_schema": _model_schema(GenerationTraceRequest),
                "example_request": {
                    "prompt": (
                        "/no_think\n"
                        "Write exactly one short, natural movie-review sentence.\n"
                        "Output only the sentence.\n"
                        "Examples:\n"
                        "The acting was uneven, but the ending still worked.\n"
                        "I liked the premise more than the execution.\n"
                        "New sentence:"
                    ),
                    "secret": "A",
                    "threshold": 0.005,
                    "temperature": 1.0,
                    "temperature_alpha": 1.25,
                    "max_bpw": 5,
                    "max_new_tokens": 64,
                    "max_steps": 32,
                },
                "common_errors": {
                    "400": "Invalid prompt, secret, or trace bounds.",
                    "500": "Unexpected backend/server failure.",
                },
            },
            "GET /docs": {
                "purpose": "FastAPI Swagger UI generated from the same request/response models.",
            },
            "GET /openapi.json": {
                "purpose": "Raw OpenAPI schema generated by FastAPI.",
            },
        },
        "powershell_examples": {
            "health": "Invoke-RestMethod http://<server-ip>:9000/health",
            "hide": (
                "$prompt = \"You are writing natural short movie comments.`n`nExamples:`n- The acting was strong but the pacing felt uneven.`n- It had a few nice scenes, though the ending felt rushed.`n`nWrite one new natural short movie comment.\"; "
                "$body = @{ prompt=$prompt; "
                "secret='a'; complete_sent=$true } | ConvertTo-Json; "
                "Invoke-RestMethod http://<server-ip>:9000/hide -Method Post "
                "-ContentType 'application/json' -Body $body"
            ),
            "reveal": (
                "$body = @{ prompt=$prompt; stegotext=$hide.stegotext; "
                "stego_token_ids=$hide.stego_token_ids; context_seed=$hide.context_seed; "
                "effective_prompt_hash=$hide.effective_prompt_hash; payload_bits_len=$hide.payload_bits } | ConvertTo-Json -Depth 8; "
                "Invoke-RestMethod http://<server-ip>:9000/reveal -Method Post "
                "-ContentType 'application/json' -Body $body"
            ),
        },
        "python_example": (
            "import requests\n"
            "base = 'http://<server-ip>:9000'\n"
            "prompt = '''You are writing natural short movie comments.\n\n"
            "Examples:\n"
            "- The acting was strong but the pacing felt uneven.\n"
            "- It had a few nice scenes, though the ending felt rushed.\n\n"
            "Write one new natural short movie comment.'''\n"
            "hide = requests.post(base + '/hide', json={\n"
            "    'prompt': prompt,\n"
            "    'secret': 'a',\n"
            "    'complete_sent': True,\n"
            "}).json()\n"
            "reveal = requests.post(base + '/reveal', json={\n"
            "    'prompt': prompt,\n"
            "    'stegotext': hide['stegotext'],\n"
            "    'stego_token_ids': hide['stego_token_ids'],\n"
            "    'payload_bits_len': hide['payload_bits'],\n"
            "    'context_seed': hide['context_seed'],\n"
            "    'effective_prompt_hash': hide['effective_prompt_hash'],\n"
            "}).json()\n"
        ),
    }


def _help_markdown(payload: dict) -> str:
    lines = [
        f"# {payload['name']}",
        "",
        payload["summary"],
        "",
        "## Base URLs",
        "",
        f"- Same machine: `{payload['base_urls']['same_machine']}`",
        f"- LAN: `{payload['base_urls']['lan_template']}`",
        "",
        "## Runtime",
        "",
    ]
    for key, value in payload["runtime"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Important Notes", ""])
    for note in payload["important_notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Endpoints", ""])
    for path, spec in payload["endpoints"].items():
        lines.extend([f"### `{path}`", "", spec.get("purpose", ""), ""])
        if "request_model" in spec:
            lines.append(f"- Request model: `{spec['request_model']}`")
        if "response_model" in spec:
            lines.append(f"- Response model: `{spec['response_model']}`")
        if "example_request" in spec:
            lines.extend(["", "Example request:", "```json", json.dumps(spec["example_request"], indent=2), "```"])
        if "common_errors" in spec:
            lines.extend(["", "Common errors:"])
            for code, text in spec["common_errors"].items():
                lines.append(f"- `{code}`: {text}")
        lines.append("")
    lines.extend(
        [
            "## PowerShell",
            "",
            "```powershell",
            payload["powershell_examples"]["hide"],
            "```",
            "",
            "## Python",
            "",
            "```python",
            payload["python_example"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_app(cfg: ZGLSConfig) -> FastAPI:
    client = ZGLSClient(cfg)
    app = FastAPI(
        title="Zero-shot GLS API",
        version="0.3.0",
        description="Hide and reveal UTF-8 secrets with the llama-server backed ZGLS EGS pipeline.",
    )

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        req_token = REQUEST_ID_CTX.set(req_id)
        started = time.perf_counter()

        raw_body = await request.body()
        parsed = _parse_json_body(raw_body)
        LOGGER.info(
            "req=%s event=request method=%s path=%s body=%s",
            req_id,
            request.method,
            request.url.path,
            _summarize_body(parsed),
        )

        async def receive():
            return {"type": "http.request", "body": raw_body, "more_body": False}

        request = Request(request.scope, receive)

        try:
            try:
                response = await call_next(request)
            except Exception:
                elapsed_ms = (time.perf_counter() - started) * 1000
                LOGGER.exception(
                    "req=%s event=response status=500 elapsed_ms=%.2f unhandled_error=true",
                    req_id,
                    elapsed_ms,
                )
                raise

            response.headers["x-request-id"] = req_id
            elapsed_ms = (time.perf_counter() - started) * 1000

            body_preview = None
            content_type = (response.headers.get("content-type") or "").lower()
            if "application/json" in content_type:
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                body_bytes = b"".join(chunks)
                parsed_response = _parse_json_body(body_bytes)
                body_preview = _summarize_body(parsed_response)
                response = Response(
                    content=body_bytes,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )

            LOGGER.info(
                "req=%s event=response status=%d elapsed_ms=%.2f body=%s",
                req_id,
                response.status_code,
                elapsed_ms,
                body_preview,
            )
            return response
        finally:
            REQUEST_ID_CTX.reset(req_token)

    @app.get("/health")
    def health():
        return client.health()

    @app.get("/help")
    def help_json():
        return _help_payload(cfg, client)

    @app.get("/help.md", response_class=PlainTextResponse)
    def help_markdown():
        return _help_markdown(_help_payload(cfg, client))

    @app.post(
        "/generation_trace.html",
        response_class=HTMLResponse,
        summary="Trace one token-by-token ZGLS generation",
        response_description="Self-contained HTML report.",
    )
    def generation_trace_html(req: GenerationTraceRequest):
        try:
            trace = generate_generation_trace(
                model=client.model,
                tokenizer=client.tokenizer,
                prompt=req.prompt,
                secret=req.secret,
                threshold=cfg.egs.threshold if req.threshold is None else float(req.threshold),
                temperature=cfg.egs.temperature if req.temperature is None else float(req.temperature),
                temperature_alpha=(
                    cfg.egs.temperature_alpha
                    if req.temperature_alpha is None
                    else float(req.temperature_alpha)
                ),
                max_bpw=cfg.egs.max_bpw if req.max_bpw is None else int(req.max_bpw),
                max_new_tokens=(
                    cfg.max_new_tokens if req.max_new_tokens is None else int(req.max_new_tokens)
                ),
                max_steps=int(req.max_steps),
                candidates_shown=req.candidates_shown,
            )
            return render_generation_trace_html(trace)
        except ZGLSError as e:
            raise _http_error(e) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"generation trace failed: {e}") from e

    @app.post("/hide", response_model=HideResponse)
    def hide(req: HideRequest):
        req_id = REQUEST_ID_CTX.get()
        try:
            result: HideResult = client.hide(
                prompt=req.prompt,
                secret=req.secret,
                complete_sent=req.complete_sent,
                request_id=req_id,
                max_new_tokens=req.max_new_tokens,
                quality_max_retries=req.quality_max_retries,
                quality_max_words=req.quality_max_words,
                threshold=req.threshold,
                temperature=req.temperature,
                temperature_alpha=req.temperature_alpha,
                max_bpw=req.max_bpw,
            )
            LOGGER.info(
                "req=%s op=hide payload_bytes=%d used_bits=%d truncated=%s quality_passed=%s",
                req_id,
                result.payload_bytes,
                result.used_bits,
                result.is_truncated,
                result.quality_passed,
            )
            return asdict(result)
        except ZGLSError as e:
            raise _http_error(e) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"hide failed: {e}") from e

    @app.post("/capacity_probe", response_model=CapacityProbeResponse)
    def capacity_probe(req: CapacityProbeRequest):
        req_id = REQUEST_ID_CTX.get()
        try:
            result: CapacityProbeResult = client.capacity_probe(
                prompt=req.prompt,
                payload_bits=req.payload_bits,
                max_words=req.max_words,
                quality_max_words=req.quality_max_words,
                quality_max_retries=req.quality_max_retries,
                payload_bits_candidates=req.payload_bits_candidates,
                initial_payload_bits=req.initial_payload_bits,
                payload_seed=req.payload_seed,
                complete_sent=req.complete_sent,
                request_id=req_id,
                max_new_tokens=req.max_new_tokens,
                threshold=req.threshold,
                temperature=req.temperature,
                temperature_alpha=req.temperature_alpha,
                max_bpw=req.max_bpw,
            )
            best_bits = result.best_success.payload_bits_exact if result.best_success else 0
            LOGGER.info(
                "req=%s op=capacity_probe best_payload_bits=%d trials=%d",
                req_id,
                best_bits,
                len(result.trials),
            )
            return asdict(result)
        except ZGLSError as e:
            raise _http_error(e) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"capacity_probe failed: {e}") from e

    @app.post("/reveal", response_model=RevealResponse)
    def reveal(req: RevealRequest):
        req_id = REQUEST_ID_CTX.get()
        try:
            result: RevealResult = client.reveal(
                prompt=req.prompt,
                stegotext=req.stegotext,
                context_seed=req.context_seed,
                effective_prompt_hash=req.effective_prompt_hash,
                stego_token_ids=req.stego_token_ids,
                payload_bits_len=req.payload_bits_len,
                threshold=req.threshold,
                temperature=req.temperature,
                temperature_alpha=req.temperature_alpha,
                max_bpw=req.max_bpw,
            )
            LOGGER.info(
                "req=%s op=reveal decode_ok=%s raw_bits=%d",
                req_id,
                result.decode_ok,
                result.raw_bits_len,
            )
            return asdict(result)
        except ZGLSError as e:
            raise _http_error(e) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"reveal failed: {e}") from e

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--backend", type=str, default="llama_server", choices=["llama_server"])
    p.add_argument("--server-url", type=str, default="http://127.0.0.1:8081")
    p.add_argument("--server-model", type=str, default="Qwen3.5-9B-Q4_K_M.gguf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cover-path", type=str, default=None)
    p.add_argument("--cover-col", type=str, default="plaintext")
    p.add_argument("--prompt-cfg-path", type=str, default=None)
    p.add_argument("--corpus", type=str, default="Unknown")
    p.add_argument("--n-cover", type=int, default=2)
    p.add_argument("--quality-max-retries", type=int, default=4)
    p.add_argument("--quality-min-words", type=int, default=8)
    p.add_argument("--quality-max-words", type=int, default=40)
    p.add_argument("--quality-max-repetition-ratio", type=float, default=0.65)
    p.add_argument("--quality-max-single-token-share", type=float, default=0.35)
    p.add_argument("--threshold", type=float, default=1e-2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--temperature-alpha", type=float, default=1.0)
    p.add_argument("--max-bpw", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--egs-mode", type=str, default="huffman", choices=list(hide_extract.MODE))
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> ZGLSConfig:
    return ZGLSConfig(
        backend=args.backend,
        server_url=args.server_url,
        server_model=args.server_model,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        cover_path=args.cover_path,
        cover_col=args.cover_col,
        prompt_cfg_path=args.prompt_cfg_path or ZGLSConfig.prompt_cfg_path,
        corpus=args.corpus,
        n_cover=args.n_cover,
        quality_max_retries=args.quality_max_retries,
        quality_min_words=args.quality_min_words,
        quality_max_words=args.quality_max_words,
        quality_max_repetition_ratio=args.quality_max_repetition_ratio,
        quality_max_single_token_share=args.quality_max_single_token_share,
        egs=EGSParams(
            mode=args.egs_mode,
            threshold=args.threshold,
            temperature=args.temperature,
            temperature_alpha=args.temperature_alpha,
            max_bpw=args.max_bpw,
        ),
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    app = build_app(config_from_args(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
