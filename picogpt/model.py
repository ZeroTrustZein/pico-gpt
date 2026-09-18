"""pico-gpt: a character-level GPT, implemented from scratch in PyTorch.

No transformers library, no nn.Transformer — just tensors and matmuls.
Two attention backends, kept for verification:
  - "manual": explicit matmul + masked softmax (the reference)
  - "sdpa":   torch scaled_dot_product_attention (fast path)
and two generation paths:
  - naive: re-run the full prefix every step
  - kv-cache: prefill once, then one token per step
Equivalence between all of these is enforced in tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256      # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False         # following GPT-2/3 & nanoGPT: no bias in linears/LN
    attn_mode: str = "sdpa"    # "sdpa" or "manual"

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.attn_mode not in ("sdpa", "manual"):
            raise ValueError(f"unknown attn_mode: {self.attn_mode!r}")


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with optional KV cache.

    Q, K, V come from one fused linear projection. With `past_kv` set, the
    new keys/values are concatenated onto the cached ones so each decode
    step processes a single token instead of the whole prefix.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attn_mode = config.attn_mode
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # split into heads: (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v)

        T_total = k.size(2)   # past + present key positions
        # query i sits at absolute position (T_total - T + i) and may attend
        # to key positions <= its own. diagonal offset T_total - T handles
        # both the no-cache case (offset 0) and cached decoding (offset > 0).
        offset = T_total - T
        if self.attn_mode == "sdpa":
            if offset == 0:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                mask = torch.tril(
                    torch.ones(T, T_total, dtype=torch.bool, device=x.device),
                    diagonal=offset,
                )
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                )
        else:  # manual reference path
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal_mask = torch.triu(
                torch.ones(T, T_total, dtype=torch.bool, device=x.device),
                diagonal=offset + 1,
            )
            att = att.masked_fill(causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.attn_dropout, training=self.training)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)   # re-assemble heads
        return self.resid_dropout(self.c_proj(y)), present


class FeedForward(nn.Module):
    """Position-wise MLP: Linear -> GELU -> Linear, with residual dropout."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: x = x + attn(ln(x)); x = x + mlp(ln(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x, past_kv=None):
        a, present = self.attn(self.ln_1(x), past_kv)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT(nn.Module):
    """The full model: token+position embeddings -> N blocks -> LN -> head."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: the output head shares weights with the token embedding
        self.transformer.wte.weight = self.lm_head.weight

        # init: normal(0, 0.02) everywhere; residual projections scaled by 1/sqrt(2*L)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self.criterion = nn.CrossEntropyLoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        """Training/full-sequence path. Returns (logits, loss)."""
        logits_all, loss, _ = self._run(idx, targets=targets, want_all=True)
        return logits_all, loss

    def _run(self, idx, targets=None, want_all=False, past_kv=None):
        """Core transformer pass.

        Returns (logits, loss, presents) where logits cover all positions
        when want_all, else only the last position.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )
        device = idx.device
        past_len = 0 if past_kv is None else past_kv[0][0].size(2)

        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)          # (B, T, C)
        pos_emb = self.transformer.wpe(pos)          # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)

        presents = []
        for i, block in enumerate(self.transformer.h):
            x, present = block(
                x, past_kv[i] if past_kv is not None else None
            )
            presents.append(present)
        x = self.transformer.ln_f(x)

        if want_all:
            logits = self.lm_head(x)                 # (B, T, vocab)
        else:
            logits = self.lm_head(x[:, [-1], :])     # (B, 1, vocab)

        loss = None
        if targets is not None:
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss, presents

    # -- KV-cache generation API -------------------------------------------

    @torch.no_grad()
    def prefill(self, idx):
        """Process a prompt, return (logits_last, kv_cache)."""
        logits, _, presents = self._run(idx, want_all=False)
        return logits[:, -1, :], presents

    @torch.no_grad()
    def step(self, token, kv_cache):
        """One cached decode step for a single new token."""
        logits, _, presents = self._run(token, want_all=False, past_kv=kv_cache)
        return logits[:, -1, :], presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, use_cache: bool = True):
        """Autoregressive sampling with temperature and top-k."""
        self.eval()
        kv_cache = None
        for _ in range(max_new_tokens):
            if use_cache:
                if kv_cache is None or kv_cache[0][0].size(2) >= self.config.block_size:
                    # Learned absolute positions shift when the window slides.
                    # Recompute: trimming cached keys would retain stale positions/context.
                    logits, kv_cache = self.prefill(idx[:, -self.config.block_size:])
                else:
                    logits, kv_cache = self.step(idx[:, -1:], kv_cache)
            else:
                logits, _ = self(idx[:, -self.config.block_size:], targets=None)
                logits = logits[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
