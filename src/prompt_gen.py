"""Generating prompts for LLMs."""
import contextlib
import csv
import logging
import random
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from prompt_template import COVER, COVER_QWEN_PLAIN

GEN_PROMPT_MODE_TYPE = Literal["cover", "sample"]
PROMPT_STYLE_TYPE = Literal["paper", "qwen_plain"]


@contextlib.contextmanager
def random_state(seed: int):
    # save state
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    try:
        yield
    finally:
        # restore state
        random.setstate(state["random"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["torch_cuda"])


@contextlib.contextmanager
def gen_prompt_ctx(
    mode: GEN_PROMPT_MODE_TYPE,
    cover: Path | str | None = None,
    cover_col: str = "plaintext",
    prompt_style: PROMPT_STYLE_TYPE = "paper",
):
    """Context manager for generating prompts.

    Args:
        mode (str): Mode.
        cover (Path | str, optional): Cover text file. Defaults to None.
        cover_col (str, optional): Cover column name. Defaults to "plaintext".
    """
    if mode in ["cover", "sample"]:
        assert cover is not None, "Cover text is required for 'cover' mode."
        cover = Path(cover)
        assert cover.exists(), f"Cover text file '{cover}' does not exist."
        logging.info(f"Loading cover text from '{cover}'.")
        with cover.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert cover_col in reader.fieldnames, f"Cover column '{cover_col}' not in {cover}."
            cover_text: list[str] = [row[cover_col] for row in reader]
        if mode == "cover":
            yield partial(
                cover_mode_prompt_gen, cover_text=cover_text, prompt_style=prompt_style
            )
        elif mode == "sample":
            yield partial(sample_mode_prompt_gen, cover_text=cover_text)
    else:
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")


def cover_mode_prompt_gen(
    n_ctx: int,
    seed: int,
    cover_text: list[str],
    corpus: str = "Unknown",
    prompt_style: PROMPT_STYLE_TYPE = "paper",
    **kwargs,
) -> str:
    """Generate prompts for cover mode.

    Args:
        n_ctx (int): Number of context sentences.
        seed (int): Random seed.
        cover_text (list[str]): Cover text.
        corpus (str, optional): Corpus name. Defaults to "Unknown".
    """
    with random_state(seed):
        # choose `n_ctx` context text
        context = random.sample(cover_text, n_ctx)
        context = "\n\n".join(context)
        template = COVER if prompt_style == "paper" else COVER_QWEN_PLAIN
        return template.substitute(corpus=corpus, context=context)


def sample_mode_prompt_gen(seed: int, cover_text: list[str], **kwargs) -> str:
    """Sample context text from cover text.

    Args:
        seed (int): Random seed.
        cover_text (list[str]): Cover text.
    """
    with random_state(seed):
        context = random.choice(cover_text)
        return context


def resolve_prompt_style(prompt_style: str, model_hint: str | None = None) -> PROMPT_STYLE_TYPE:
    if prompt_style != "auto":
        assert prompt_style in ("paper", "qwen_plain"), f"Unknown prompt style: {prompt_style}"
        return prompt_style

    hint = (model_hint or "").lower()
    if "qwen" in hint:
        return "qwen_plain"
    return "paper"
