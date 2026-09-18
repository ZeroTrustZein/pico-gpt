"""Prepare a corpus: download tiny-shakespeare if absent, tokenize, split."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import torch

from picogpt.data import CharTokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(input_path: Path, out_dir: Path) -> None:
    text = input_path.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} chars")

    tok = CharTokenizer(text)
    print(f"vocab size: {tok.vocab_size}")

    ids = tok.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train_data, "val": val_data}, out_dir / "corpus.pt")
    tok.save(out_dir / "tokenizer.pkl")
    np.save(out_dir / "ids.npy", np.array(ids, dtype=np.int32))
    print(f"saved: train {len(train_data):,} tokens | val {len(val_data):,} tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the training corpus")
    ap.add_argument("--input", default="data/input.txt")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"downloading tiny-shakespeare -> {input_path}")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, input_path)

    prepare(input_path, Path(args.out))


if __name__ == "__main__":
    main()
