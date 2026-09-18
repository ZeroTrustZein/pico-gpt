"""Prepare a corpus with the byte-level BPE tokenizer.

Downloads tiny-shakespeare if absent, trains BPE to the requested vocab
size, encodes, splits 90/10, saves corpus + tokenizer.
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import torch

from picogpt.bpe import BPETokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare BPE corpus")
    ap.add_argument("--input", default="data/input.txt")
    ap.add_argument("--out", default="data")
    ap.add_argument("--vocab-size", type=int, default=512)
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"downloading tiny-shakespeare -> {input_path}")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, input_path)

    text = input_path.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} chars")

    t0 = time.time()
    tok = BPETokenizer()
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]
    tok.train(train_text, vocab_size=args.vocab_size)
    print(f"BPE trained: vocab {tok.vocab_size} in {time.time() - t0:.1f}s")

    train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
    n = len(train_data) + len(val_data)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train_data, "val": val_data}, out / "corpus_bpe.pt")
    tok.save(out / "tokenizer_bpe.json")
    print(f"saved: train {len(train_data):,} tokens | val {len(val_data):,} tokens "
          f"(chars/token: {len(text) / n:.2f})")


if __name__ == "__main__":
    main()
