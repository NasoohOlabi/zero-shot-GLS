"""
Llama.cpp server backend.

We use an existing `llama-server` instance (OpenAI-compatible) to:
1) tokenize / detokenize text
2) get next-token distributions via `logprobs` for EGS

This avoids requiring local compilation of llama.cpp Python bindings on Windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from types import SimpleNamespace


@dataclass
class LlamaServerConfig:
    base_url: str = "http://127.0.0.1:8081"
    model: str = "Qwen3.5-9B-Q4_K_M.gguf"
    logprobs: int = 256
    timeout_s: int = 120
    retries: int = 2
    seed: int = 42


class LlamaServerCapabilityError(RuntimeError):
    pass


class LlamaServerModel:
    def __init__(self, cfg: LlamaServerConfig):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self._meta = self._get_model_meta()
        self.vocab_size = int(self._meta.get("meta", {}).get("n_vocab", 0)) or None

        class _Config:
            eos_token_id = None

        self.config = _Config()

    def eval(self):
        # HF models use .eval() to switch off dropout; server backend is already deterministic.
        return self

    def _url(self, path: str) -> str:
        return self.cfg.base_url.rstrip("/") + path

    def _get_model_meta(self) -> Dict[str, Any]:
        r = requests.get(self._url("/v1/models"), timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()
        # Find matching id
        for m in data.get("data", []):
            if m.get("id") == self.cfg.model:
                return m
        # fallback: return first
        return (data.get("data") or [{}])[0]

    def tokenize(self, text: str) -> torch.Tensor:
        r = requests.post(self._url("/tokenize"), json={"content": text}, timeout=self.cfg.timeout_s)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise LlamaServerCapabilityError(
                "Server backend does not expose /tokenize. This repo's EGS pipeline needs "
                "tokenize/detokenize plus token-level top-logprobs. LM Studio's OpenAI-compatible "
                "server is not sufficient for stage2/stage3 reproduction."
            ) from e
        payload = r.json()
        if payload.get("error"):
            raise LlamaServerCapabilityError(
                f"Server backend rejected /tokenize: {payload['error']}. This repo's EGS pipeline "
                "needs tokenize/detokenize plus token-level top-logprobs."
            )
        toks = payload.get("tokens", [])
        if not toks:
            raise LlamaServerCapabilityError(
                "Server backend returned no tokens from /tokenize. This repo cannot reproduce "
                "the hiding/extraction path without stable token ids."
            )
        return torch.tensor([toks], dtype=torch.long, device=self.device)

    def detokenize(self, token_ids: List[int]) -> str:
        r = requests.post(
            self._url("/detokenize"), json={"tokens": token_ids}, timeout=self.cfg.timeout_s
        )
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise LlamaServerCapabilityError(
                "Server backend does not expose /detokenize. This repo's EGS pipeline cannot "
                "decode stego token ids reliably without that endpoint."
            ) from e
        payload = r.json()
        if payload.get("error"):
            raise LlamaServerCapabilityError(
                f"Server backend rejected /detokenize: {payload['error']}. This repo cannot "
                "decode stego token ids reliably without that endpoint."
            )
        content = payload.get("content", "")
        if content == "":
            raise LlamaServerCapabilityError(
                "Server backend returned empty content from /detokenize. This repo cannot verify "
                "token-stable stegotext decoding in that configuration."
            )
        return content

    def next_token_top_logprobs(
        self,
        input_ids: torch.Tensor,
        temperature: float,
        logprobs: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Return a list of (token_id, prob) sorted descending by prob.
        """
        assert input_ids.dim() == 2 and input_ids.size(0) == 1
        prompt_tokens = input_ids[0].tolist()
        body = {
            "model": self.cfg.model,
            "prompt": prompt_tokens,
            "max_tokens": 1,
            "temperature": float(temperature),
            "logprobs": int(logprobs or self.cfg.logprobs),
            "seed": int(self.cfg.seed),
        }
        last_err: Exception | None = None
        tlp = []
        for _ in range(self.cfg.retries + 1):
            try:
                r = requests.post(
                    self._url("/v1/completions"),
                    json=body,
                    timeout=self.cfg.timeout_s,
                )
                r.raise_for_status()
                j = r.json()
                choice0 = (j.get("choices") or [{}])[0]
                logp = choice0.get("logprobs") or {}
                content = logp.get("content") or []
                if len(content) > 0:
                    tlp = content[0].get("top_logprobs") or []
                if tlp:
                    break
                if choice0.get("logprobs") is None:
                    last_err = LlamaServerCapabilityError(
                        "Server backend returned logprobs=null. This repo needs token-level "
                        "top_logprobs for enhanced greedy search, so OpenAI-compatible LM Studio "
                        "responses are not enough for fair reproduction."
                    )
                else:
                    last_err = RuntimeError(f"missing top_logprobs in completion response: {j}")
            except Exception as e:
                last_err = e
        if not tlp:
            raise RuntimeError(f"llama-server returned no top_logprobs after retries: {last_err}")
        out: List[Tuple[int, float]] = []
        for item in tlp:
            tid = int(item["id"])
            tok = item.get("token") or ""
            if tok.strip() == "":
                continue
            if "<think" in tok.lower() or "</think" in tok.lower():
                continue
            lp = float(item["logprob"])
            out.append((tid, float(torch.exp(torch.tensor(lp)).item())))
        # The server ordering can be nondeterministic for (near-)ties. Use a stable
        # deterministic ordering: prob desc, token_id asc.
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    def complete_tokens(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 32,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Append completion tokens after `input_ids` and return full ids tensor.
        """
        assert input_ids.dim() == 2 and input_ids.size(0) == 1
        prompt_tokens = input_ids[0].tolist()
        body = {
            "model": self.cfg.model,
            "prompt": prompt_tokens,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "logprobs": 1,
            "seed": int(self.cfg.seed),
        }
        r = requests.post(
            self._url("/v1/completions"),
            json=body,
            timeout=self.cfg.timeout_s,
        )
        r.raise_for_status()
        j = r.json()
        choice0 = (j.get("choices") or [{}])[0]
        logp = choice0.get("logprobs") or {}
        content = logp.get("content") or []
        new_ids = [int(item["id"]) for item in content if "id" in item]
        if not new_ids:
            # fallback: tokenize returned text if token ids are not present
            txt = choice0.get("text") or ""
            if txt:
                new_ids = self.tokenize(txt)[0].tolist()
        if not new_ids:
            return input_ids
        appended = torch.tensor([new_ids], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, appended], dim=1)


class LlamaServerTokenizer:
    """
    Minimal tokenizer shim to match the subset of HuggingFace tokenizer APIs used by scripts.
    """

    def __init__(self, model: LlamaServerModel):
        self.model = model

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
        # ignore truncation/max_length kwargs for now
        input_ids = self.model.tokenize(text)
        return SimpleNamespace(input_ids=input_ids)

    def decode(self, token_ids: List[int]) -> str:
        return self.model.detokenize(token_ids)
