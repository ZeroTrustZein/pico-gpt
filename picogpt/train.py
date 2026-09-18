"""Training loop: AdamW, cosine LR with warmup, grad accumulation, checkpointing.

Evaluation is exact: full-corpus loss over fixed non-overlapping windows
(no sampled batches), so val numbers are comparable across runs.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .data import get_batch
from .model import GPT, GPTConfig


@dataclass
class TrainConfig:
    max_iters: int = 5000
    batch_size: int = 64
    grad_accum_steps: int = 1      # effective batch = batch_size * grad_accum_steps
    eval_interval: int = 250
    eval_iters: int = 50           # sampled-batch estimate (quick signal)
    eval_max_windows: int = 0      # exact full-corpus eval: 0 = all windows
    learning_rate: float = 1e-3
    warmup_iters: int = 100
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cpu"
    out_dir: Path = field(default_factory=lambda: Path("out"))

    def __post_init__(self):
        if min(self.max_iters, self.batch_size, self.eval_interval,
               self.eval_iters, self.grad_accum_steps) <= 0:
            raise ValueError("iterations, batch size, accumulation and eval settings must be positive")
        if self.warmup_iters < 0 or self.learning_rate <= 0:
            raise ValueError("warmup must be nonnegative and learning rate positive")


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Linear warmup -> cosine decay to min_lr_frac * base."""
    warmup = min(cfg.warmup_iters, cfg.max_iters // 10)
    if it >= cfg.max_iters:
        return cfg.learning_rate * cfg.min_lr_frac
    if it < warmup:
        return cfg.learning_rate * (it + 1) / warmup
    progress = (it - warmup) / (cfg.max_iters - warmup)
    return cfg.learning_rate * (
        cfg.min_lr_frac + 0.5 * (1 - cfg.min_lr_frac)
        * (1 + math.cos(math.pi * progress))
    )


@torch.no_grad()
def exact_eval(model: GPT, data: torch.Tensor, batch_size: int,
               max_windows: int = 0) -> float:
    """Mean cross-entropy over every non-overlapping window of the corpus.

    Deterministic and complete: the same data always yields the same number,
    unlike sampled-batch estimates. Windows are processed in batches.
    """
    if len(data) < 2 or batch_size < 1 or max_windows < 0:
        raise ValueError("evaluation needs at least two tokens and valid batch/window limits")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    T = model.config.block_size
    end = len(data) - 1
    if max_windows:
        end = min(end, max_windows * T)
    total_nll = 0.0
    total_tokens = 0
    try:
        # Include the final partial window; each target is counted exactly once.
        for start in range(0, end, batch_size * T):
            stops = list(range(start, min(start + batch_size * T, end), T))
            for length in sorted({min(T, end - s) for s in stops}):
                group = [s for s in stops if min(T, end - s) == length]
                x = torch.stack([data[s:s + length] for s in group]).to(device)
                y = torch.stack([data[s + 1:s + length + 1] for s in group]).to(device)
                _, loss = model(x, y)
                count = y.numel()
                total_nll += loss.item() * count
                total_tokens += count
        return total_nll / total_tokens
    finally:
        model.train(was_training)


@torch.no_grad()
def estimate_loss(model, data, cfg: TrainConfig) -> dict:
    """Quick sampled-batch estimate (the classic nanoGPT-style eval)."""
    model.eval()
    out = {}
    for name, d in data.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(d, model.config.block_size, cfg.batch_size, cfg.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(model: GPT, data: dict, cfg: TrainConfig, log=print) -> dict:
    """Train `model` on `data` (dict of train/val token tensors). Returns history."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    model.to(cfg.device)
    model.train()

    history = {"train": [], "val": [], "val_exact": [], "lr": [], "iters": []}
    best_val = float("inf")
    t0 = time.time()

    for it in range(cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, data, cfg)
            val_exact = exact_eval(model, data["val"], cfg.batch_size,
                                   cfg.eval_max_windows)
            if not all(math.isfinite(v) for v in losses.values()) \
                    or not math.isfinite(val_exact):
                raise FloatingPointError(f"non-finite evaluation loss at iteration {it}")
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            history["val_exact"].append(val_exact)
            history["lr"].append(lr)
            history["iters"].append(it)
            log(
                f"iter {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | val_exact {val_exact:.4f} | "
                f"lr {lr:.2e} | {time.time() - t0:6.1f}s"
            )
            if val_exact < best_val:
                best_val = val_exact
                torch.save(
                    {"model": model.state_dict(), "config": asdict(model.config),
                     "optimizer": opt.state_dict(), "iter": it, "val": best_val},
                    cfg.out_dir / "ckpt.pt",
                )
            history_path = cfg.out_dir / "history.json"
            temporary = history_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
            temporary.replace(history_path)

        if it == cfg.max_iters:
            break

        opt.zero_grad(set_to_none=True)
        # gradient accumulation: K micro-batches per optimizer step
        for micro in range(cfg.grad_accum_steps):
            x, y = get_batch(data["train"], model.config.block_size,
                             cfg.batch_size, cfg.device)
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at iteration {it}")
            (loss / cfg.grad_accum_steps).backward()
            if micro == 0 and it % 25 == 0:
                elapsed = time.time() - t0
                log(f"progress {it}/{cfg.max_iters} | loss {loss.item():.4f} | "
                    f"elapsed {elapsed:.1f}s")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    log(f"done in {time.time() - t0:.1f}s | best val_exact {best_val:.4f}")
    return history
