"""Shared fixtures: everything here is CPU-only and download-free.

- ToySAE: a real JumpReLU autoencoder (random weights) satisfying SAELike.
- FakeTokenizer: word-level tokenizer with HF-compatible offset mapping.
- make_chain: assembles a valid ChainRecord from raw text.
"""

from __future__ import annotations

import re

import pytest
import torch

from src.config import EventsConfig, GenerationConfig, ProbeConfig
from src.generation import compute_token_char_offsets, find_trace_span
from src.io_utils import ChainRecord

DEFAULT_LEXICON = [
    "wait", "actually", "hold on", "let me recheck", "that's wrong",
    "i made a mistake", "let me reconsider", "scratch that",
]


class ToySAE(torch.nn.Module):
    def __init__(self, d_model: int = 8, d_sae: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, d_sae, generator=g))
        self.b_enc = torch.nn.Parameter(torch.randn(d_sae, generator=g) * 0.1)
        self.W_dec = torch.nn.Parameter(torch.randn(d_sae, d_model, generator=g) * 0.3)
        self.b_dec = torch.nn.Parameter(torch.randn(d_model, generator=g) * 0.1)
        self.theta = 0.05  # JumpReLU threshold
        self.requires_grad_(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.theta)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


class FakeTokenizer:
    """Word-level tokenizer: one token per `\\S+\\s*` chunk. Round-trips text
    exactly, supports HF-style offset mapping, so the fast path of
    compute_token_char_offsets is exercised."""

    eos_token_id = 0
    pad_token_id = 0

    def __init__(self):
        self._tok2id = {"<eos>": 0}
        self._id2tok = {0: "<eos>"}

    def _chunks(self, text: str) -> list[str]:
        return re.findall(r"\S+\s*|\s+", text)

    def _id(self, chunk: str) -> int:
        if chunk not in self._tok2id:
            i = len(self._tok2id)
            self._tok2id[chunk] = i
            self._id2tok[i] = chunk
        return self._tok2id[chunk]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(c) for c in self._chunks(text)]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self._id2tok[i] for i in ids if not (skip_special_tokens and i == 0))

    def apply_chat_template(self, messages, add_generation_prompt: bool = True,
                            tokenize: bool = True):
        text = "".join(f"<{m['role']}> {m['content']} " for m in messages)
        if add_generation_prompt:
            text += "<assistant> "
        return self.encode(text) if tokenize else text

    def __call__(self, text: str, return_offsets_mapping: bool = False,
                 add_special_tokens: bool = False) -> dict:
        ids, offsets, pos = [], [], 0
        for c in self._chunks(text):
            ids.append(self._id(c))
            offsets.append((pos, pos + len(c)))
            pos += len(c)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


@pytest.fixture
def toy_sae() -> ToySAE:
    return ToySAE()


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def events_cfg() -> EventsConfig:
    return EventsConfig(lexicon=DEFAULT_LEXICON)


@pytest.fixture
def gen_cfg() -> GenerationConfig:
    return GenerationConfig(
        probe=ProbeConfig(template="\n</think>\n\nFinal Answer: \\boxed{"),
    )


def make_chain(
    tokenizer: FakeTokenizer,
    completion_text: str,
    chain_id: str = "p0__c0",
    gold: str = "42",
    prompt_text: str = "Solve the problem. ",
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> ChainRecord:
    prompt_ids = tokenizer.encode(prompt_text)
    completion_ids = tokenizer.encode(completion_text)
    text, offsets = compute_token_char_offsets(tokenizer, completion_ids)
    assert text == completion_text
    span = find_trace_span(text, offsets, len(prompt_ids), think_open, think_close)
    return ChainRecord(
        chain_id=chain_id,
        problem_id=chain_id.split("__")[0],
        dataset="test",
        prompt=prompt_text,
        gold_answer=gold,
        seed=0,
        token_ids=prompt_ids + completion_ids,
        prompt_token_len=len(prompt_ids),
        text=text,
        token_char_offsets=offsets,
        trace_token_span=span,
    )
