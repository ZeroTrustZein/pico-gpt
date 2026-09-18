"""Create a loss curve and seeded text samples from a completed run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sample import load_tokenizer
from picogpt.model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out_run")
    parser.add_argument("--tokenizer", default="data/tokenizer.pkl")
    args = parser.parse_args()
    out = Path(args.out)
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(out / "ckpt.pt", weights_only=True, map_location="cpu")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the checkpoint")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    validation = "val_exact" if "val_exact" in history else "val"
    for split in ("train", validation):
        axes[0].plot(history["iters"], history[split], label=split, marker=".")
    axes[0].set(xlabel="Optimizer steps", ylabel="Cross entropy (nats/token)",
                title="pico-gpt | Shakespeare")
    axes[0].legend()
    axes[1].plot(history["iters"], history["lr"], color="teal")
    axes[1].set(xlabel="Optimizer steps", ylabel="Learning rate", title="Warmup + cosine decay")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=160)
    plt.close(fig)

    samples = []
    for temperature in (0.6, 0.8, 1.0):
        torch.manual_seed(42)
        prompt = "ROMEO:"
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        result = model.generate(ids, 500, temperature=temperature, top_k=20)
        samples.append(f"Temperature {temperature}, seed 42\n{tokenizer.decode(result[0].tolist())}")
    (out / "samples.txt").write_text("\n\n---\n\n".join(samples), encoding="utf-8")
    summary = {
        "steps": history["iters"][-1],
        "initial_validation_loss": history[validation][0],
        "final_validation_loss": history[validation][-1],
        "evaluation": "All validation targets, disjoint contexts including final partial window" if validation == "val_exact" else "Sampled validation batches",
        "best_validation_loss": checkpoint["val"],
        "best_checkpoint_step": checkpoint["iter"],
        "token_perplexity": math.exp(checkpoint["val"]),
        "total_parameters": model.num_params(non_embedding=False),
        "config": checkpoint["config"],
        "limitations": "Shakespeare-only experimental language model, not a chatbot. Validation selects checkpoints; it is not an untouched test set. Token perplexities are not comparable across tokenizers.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {out / 'loss_curve.png'}, {out / 'samples.txt'}, {out / 'summary.json'}")


if __name__ == "__main__":
    main()
