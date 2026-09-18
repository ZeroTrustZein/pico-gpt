"""Train pico-gpt on the prepared corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from picogpt.model import GPT, GPTConfig
from picogpt.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train pico-gpt")
    ap.add_argument("--data", default="data/corpus.pt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--iters", "--max_iters", dest="iters", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-iters", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    corpus = torch.load(args.data, weights_only=True)
    vocab_size = max(int(corpus[split].max()) for split in ("train", "val")) + 1

    model_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(model_cfg)
    print(f"model parameters: {model.num_params():,}")

    train_cfg = TrainConfig(
        max_iters=args.iters,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        out_dir=Path(args.out),
        learning_rate=args.lr,
        grad_accum_steps=args.grad_accum,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
    )
    history = train(model, {"train": corpus["train"], "val": corpus["val"]},
                    train_cfg)

    out = Path(args.out)
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved {out / 'history.json'}")


if __name__ == "__main__":
    main()
