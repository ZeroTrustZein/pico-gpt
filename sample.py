"""Sample from a trained checkpoint (char or BPE tokenizer, auto-detected)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from picogpt.bpe import BPETokenizer
from picogpt.data import CharTokenizer
from picogpt.model import GPT, GPTConfig


def load_tokenizer(path: str):
    if str(path).endswith(".json"):
        return BPETokenizer.load(path)
    return CharTokenizer.load(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text with pico-gpt")
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.pkl")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    torch.set_num_threads(4)
    ckpt = torch.load(args.ckpt, weights_only=True, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = load_tokenizer(args.tokenizer)
    x = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    y = model.generate(x, args.tokens, temperature=args.temperature,
                       top_k=args.top_k)
    print(tok.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
