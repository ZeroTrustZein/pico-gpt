"""Equivalence tests: manual vs SDPA attention, naive vs KV-cache generation."""
import time

import torch
import pytest

from picogpt.model import GPT, GPTConfig


def make_model(attn_mode, seed=1337):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2,
                    n_embd=32, dropout=0.0, attn_mode=attn_mode)
    return GPT(cfg)


@pytest.mark.parametrize("attn_mode", ["manual", "sdpa"])
def test_causal_masking(attn_mode):
    """Changing a FUTURE token must not change logits at earlier positions."""
    m = make_model(attn_mode)
    x = torch.randint(0, 32, (1, 8))
    y = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        logits1, _ = m(x, y)
        x2 = x.clone()
        x2[0, 5] = (x2[0, 5] + 7) % 32
        logits2, _ = m(x2, y)
    assert torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-5)
    assert not torch.allclose(logits1[0, 5:], logits2[0, 5:], atol=1e-1)


def test_sdpa_matches_manual():
    """The fast path must be numerically identical to the reference path."""
    m_manual = make_model("manual")
    m_sdpa = make_model("sdpa")
    m_sdpa.load_state_dict(m_manual.state_dict())
    x = torch.randint(0, 32, (2, 12))
    with torch.no_grad():
        l1, loss1 = m_manual(x, torch.zeros(2, 12, dtype=torch.long))
        l2, loss2 = m_sdpa(x, torch.zeros(2, 12, dtype=torch.long))
    assert torch.allclose(l1, l2, atol=1e-5)
    assert torch.allclose(loss1, loss2, atol=1e-6)


def test_kv_cache_logits_match_naive_step_by_step():
    """The rigorous cache check: at every decode step, the cached single-token
    logits must match a full-sequence forward over the same prefix."""
    torch.manual_seed(7)
    m = make_model("sdpa")
    prompt = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        logits, cache = m.prefill(prompt)
        full, _ = m(prompt, targets=None)
        assert torch.allclose(logits, full[:, -1, :], atol=1e-5)
        seq = prompt
        for step in range(10):
            token = logits.argmax(-1, keepdim=True)  # greedy continuation
            cached_last, cache = m.step(token, cache)
            seq = torch.cat([seq, token], dim=1)
            full, _ = m(seq, targets=None)
            # cached logits after ingesting token == naive logits over seq+token
            assert torch.allclose(cached_last, full[:, -1, :], atol=1e-4), f"step {step}"
            logits = full[:, -1, :]


def test_kv_cache_greedy_generation_matches_naive():
    """End-to-end: greedy decoding (temp->0) must be identical for both paths.
    Sampled equality is NOT asserted: the two paths hit different SDPA kernel
    shapes, so ~1e-6 logit noise can legally flip a near-tie multinomial draw."""
    torch.manual_seed(7)
    m = make_model("sdpa")
    prompt = torch.randint(0, 32, (1, 6))
    torch.manual_seed(123)
    a = m.generate(prompt.clone(), 10, temperature=0.01, use_cache=False)
    torch.manual_seed(123)
    b = m.generate(prompt.clone(), 10, temperature=0.01, use_cache=True)
    assert torch.equal(a, b)


def test_generation_past_block_size_stays_bounded():
    """Cache rebuilds on window shifts must preserve naive greedy output."""
    torch.manual_seed(7)
    m = make_model("sdpa")
    prompt = torch.randint(0, 32, (1, 4))
    out = m.generate(prompt.clone(), 60, temperature=1.0, top_k=10, use_cache=True)
    assert out.shape == (1, 64)
    assert ((out >= 0) & (out < 32)).all()
    cached = m.generate(prompt.clone(), 60, top_k=1, use_cache=True)
    naive = m.generate(prompt.clone(), 60, top_k=1, use_cache=False)
    assert torch.equal(cached, naive)


@pytest.mark.parametrize("attn_mode", ["manual", "sdpa"])
def test_long_prompt_sampled_cache_regression(attn_mode):
    m = make_model(attn_mode)
    prompt = torch.arange(40).reshape(1, 40) % 32
    torch.manual_seed(42)
    naive = m.generate(prompt.clone(), 240, top_k=10, use_cache=False)
    torch.manual_seed(42)
    cached = m.generate(prompt.clone(), 240, top_k=10, use_cache=True)
    assert torch.equal(cached, naive)
    assert torch.equal(cached[:, :40], prompt)


def test_prefill_step_api():
    m = make_model("manual")
    prompt = torch.randint(0, 32, (1, 6))
    logits, cache = m.prefill(prompt)
    assert logits.shape == (1, 32)
    assert len(cache) == m.config.n_layer
    logits2, cache = m.step(torch.tensor([[5]]), cache)
    assert logits2.shape == (1, 32)


def test_generate_shapes_and_state():
    m = make_model("sdpa")
    x = torch.randint(0, 32, (2, 4))
    out = m.generate(x.clone(), max_new_tokens=10, top_k=5)
    assert out.shape == (2, 14)
    assert torch.equal(out[:, :4], x)


def test_temperature_extremes():
    m = make_model("sdpa")
    x = torch.randint(0, 32, (1, 4))
    torch.manual_seed(42)
    a = m.generate(x.clone(), 8, temperature=0.01)
    torch.manual_seed(999)
    b = m.generate(x.clone(), 8, temperature=0.01)
    assert torch.equal(a, b)


def test_kv_cache_is_faster():
    """The whole point of the cache: fewer FLOPs per token."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=128, n_layer=4, n_head=4,
                    n_embd=128, dropout=0.0)
    m = GPT(cfg)
    prompt = torch.randint(0, 64, (1, 100))

    torch.manual_seed(1)
    t0 = time.perf_counter()
    m.generate(prompt.clone(), 60, temperature=1.0, use_cache=False)
    t_naive = time.perf_counter() - t0

    torch.manual_seed(1)
    t0 = time.perf_counter()
    m.generate(prompt.clone(), 60, temperature=1.0, use_cache=True)
    t_cached = time.perf_counter() - t0

    assert t_cached < t_naive


def test_loss_near_ln_vocab_at_init():
    m = make_model("sdpa")
    x = torch.randint(0, 32, (4, 16))
    y = torch.randint(0, 32, (4, 16))
    _, loss = m(x, y)
    assert abs(loss.item() - torch.log(torch.tensor(32.0)).item()) < 0.15


def test_overfit_single_batch():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2,
                    n_embd=32, dropout=0.0)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 16, (8, 8))
    y = torch.randint(0, 16, (8, 8))
    first = None
    for _ in range(200):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2


def test_weight_tying():
    m = make_model("sdpa")
    assert m.transformer.wte.weight is m.lm_head.weight


def test_config_validation():
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, n_embd=30, n_head=4)
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=10, attn_mode="bogus")
