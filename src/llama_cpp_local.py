"""
Local llama-cpp-python backend for GGUF models.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
from llama_cpp import Llama


@dataclass
class LlamaCppLocalConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    logprobs: int = 2048


class LlamaCppLocalModel:
    def __init__(self, cfg: LlamaCppLocalConfig):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self._llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            logits_all=False,
            verbose=False,
        )
        self.vocab_size = int(self._llm.n_vocab())

        class _Config:
            eos_token_id = None

        self.config = _Config()

    def eval(self):
        return self

    def tokenize(self, text: str) -> torch.Tensor:
        toks = self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        return torch.tensor([toks], dtype=torch.long, device=self.device)

    def detokenize(self, token_ids: List[int]) -> str:
        return self._llm.detokenize(token_ids, special=True).decode("utf-8", errors="ignore")

    def next_token_top_logprobs(
        self,
        input_ids: torch.Tensor,
        temperature: float,
        logprobs: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        assert input_ids.dim() == 2 and input_ids.size(0) == 1
        out = self._llm.create_completion(
            prompt=input_ids[0].tolist(),
            max_tokens=1,
            temperature=float(temperature),
            logprobs=int(logprobs or self.cfg.logprobs),
        )
        top = out["choices"][0]["logprobs"]["top_logprobs"][0]
        parsed = [(int(item["token_id"]), float(torch.exp(torch.tensor(item["logprob"])).item())) for item in top]
        parsed.sort(key=lambda x: (-x[1], x[0]))
        return parsed


class LlamaCppLocalTokenizer:
    def __init__(self, model: LlamaCppLocalModel):
        self.model = model

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
        input_ids = self.model.tokenize(text)
        return SimpleNamespace(input_ids=input_ids)

    def decode(self, token_ids: List[int]) -> str:
        return self.model.detokenize(token_ids)
