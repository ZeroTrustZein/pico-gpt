"""Byte-level BPE tokenizer, implemented from scratch (no tiktoken/sentencepiece).

Pipeline: text -> UTF-8 bytes -> greedy BPE merges -> token ids.
All 256 byte values are always representable, so any string round-trips.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _get_pair_counts(ids: list[int]) -> Counter:
    counts: Counter = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Trainable byte-level BPE. Vocab layout: [0..255] = raw bytes, then merges."""

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        # merges[i] maps to token id 256 + i
        self.merges: list[tuple[int, int]] = list(merges) if merges else []
        self._rebuild()

    def _rebuild(self) -> None:
        # vocab: id -> bytes
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            vocab[256 + i] = vocab[a] + vocab[b]
        self.vocab = vocab

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        n_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))
        self.merges = []
        self._rebuild()
        for step in range(n_merges):
            counts = _get_pair_counts(ids)
            if not counts:
                break
            pair, _ = max(counts.items(), key=lambda kv: kv[1])
            new_id = 256 + step
            ids = _merge(ids, pair, new_id)
            self.merges.append(pair)
            if verbose:
                print(f"merge {step}: {pair} -> {new_id} "
                      f"({self.vocab[pair[0]]!r}+{self.vocab[pair[1]]!r})")
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self._rebuild()

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        # apply merges in training order (priority = merge index)
        for step, pair in enumerate(self.merges):
            new_id = 256 + step
            if pair[0] in ids and pair[1] in ids:
                ids = _merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path) -> None:
        Path(path).write_text(
            json.dumps({"merges": [list(p) for p in self.merges]}), encoding="utf-8"
        )

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(p) for p in d["merges"]])
