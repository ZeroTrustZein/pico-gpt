"""Character-level tokenizer: the whole dataset is one long string."""
from __future__ import annotations

import pickle
from pathlib import Path

import torch


class CharTokenizer:
    """Bidirectional char <-> id mapping built from a corpus."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"vocab_size": self.vocab_size, "itos": self.itos}, f)

    @classmethod
    def load(cls, path) -> "CharTokenizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        tok = cls.__new__(cls)
        tok.vocab_size = d["vocab_size"]
        tok.itos = d["itos"]
        tok.stoi = {ch: i for i, ch in d["itos"].items()}
        return tok


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device):
    """Sample random (context, target) pairs from a 1-D token tensor."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
