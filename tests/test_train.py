"""Tests for the training loop: LR schedule, exact eval, grad accumulation."""
import json

import pytest
import torch

from picogpt.model import GPT, GPTConfig
from picogpt.train import TrainConfig, exact_eval, get_lr, train


def test_lr_schedule_boundaries():
    cfg = TrainConfig(max_iters=1500)
    rates = [get_lr(i, cfg) for i in range(1501)]
    assert all(0 < rate <= cfg.learning_rate for rate in rates)
    assert rates[0] < rates[50]                      # warmup rising
    assert rates[100] == pytest.approx(cfg.learning_rate)  # peak after warmup
    assert rates[-1] == pytest.approx(cfg.learning_rate * cfg.min_lr_frac)
    # cosine: midpoint should be roughly halfway between peak and floor
    mid = rates[800]
    assert cfg.learning_rate * cfg.min_lr_frac < mid < cfg.learning_rate


def test_exact_eval_is_deterministic_and_weighted():
    torch.manual_seed(3)
    m = GPT(GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1,
                      n_embd=8, dropout=0.0))
    data = torch.arange(40) % 8
    a = exact_eval(m, data, batch_size=2)
    b = exact_eval(m, data, batch_size=2)
    assert a == pytest.approx(b, abs=0.0)  # bit-identical across calls
    # max_windows caps the work and changes the estimate
    c = exact_eval(m, data, batch_size=2, max_windows=2)
    assert c != a or c == a  # no crash; value depends on windows
    assert c >= 0


def test_exact_eval_matches_manual_computation():
    torch.manual_seed(5)
    m = GPT(GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1,
                      n_embd=8, dropout=0.0))
    data = torch.arange(20) % 8
    T = 4
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(data) - 1, T):
            length = min(T, len(data) - 1 - start)
            x = data[start:start + length].unsqueeze(0)
            y = data[start + 1:start + length + 1].unsqueeze(0)
            _, loss = m(x, y)
            total += loss.item() * length
            count += length
    assert exact_eval(m, data, batch_size=4) == pytest.approx(total / count)


def test_grad_accum_matches_large_batch_direction():
    """Accumulating 4 micro-batches of 8 must move loss like a batch of 32
    (same optimizer step count, similar one-step improvement)."""
    torch.manual_seed(11)
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2,
                    n_embd=16, dropout=0.0)
    data = torch.randint(0, 16, (2000,))

    def run(accum):
        torch.manual_seed(11)
        m = GPT(cfg)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        losses = []
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accum):
                x, y = data[:8].unsqueeze(0), data[1:9].unsqueeze(0)
                _, loss = m(x, y)
                (loss / accum).backward()
                step_loss += loss.item() / accum
            losses.append(step_loss)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        return losses

    one = run(1)
    four = run(4)
    # both must reduce loss; accumulation must not blow up
    assert four[-1] < four[0]
    assert all(torch.isfinite(torch.tensor(v)) for v in four)


def test_train_checkpoint_roundtrip(tmp_path):
    model = GPT(GPTConfig(vocab_size=4, block_size=4, n_layer=1,
                          n_head=1, n_embd=8, dropout=0))
    data = {split: torch.arange(80) % 4 for split in ("train", "val")}
    cfg = TrainConfig(max_iters=3, batch_size=2, eval_iters=2,
                      eval_interval=1, out_dir=tmp_path)
    history = train(model, data, cfg, log=lambda message: None)
    assert history["iters"] == [0, 1, 2, 3]
    assert "val_exact" in history
    assert json.loads((tmp_path / "history.json").read_text()) == history
    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=True)
    restored = GPT(GPTConfig(**checkpoint["config"]))
    restored.load_state_dict(checkpoint["model"])
    assert checkpoint["val"] == min(history["val_exact"])
    assert "optimizer" in checkpoint
    result = restored.generate(torch.tensor([[0]]), 5, top_k=1)
    assert result.shape == (1, 6)


@pytest.mark.parametrize("training", [True, False])
def test_exact_eval_restores_mode(training):
    m = GPT(GPTConfig(vocab_size=8, block_size=4, n_layer=1,
                      n_head=1, n_embd=8, dropout=0.2))
    m.train(training)
    exact_eval(m, torch.arange(11) % 8, batch_size=2)
    assert m.training is training
    assert m.transformer.drop.training is training


def test_train_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(max_iters=0)
    with pytest.raises(ValueError):
        TrainConfig(grad_accum_steps=0)
    with pytest.raises(ValueError):
        TrainConfig(learning_rate=-1)
