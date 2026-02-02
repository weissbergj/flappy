# src/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def _rotate_half(x):
    # x: (..., d)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, D)
    # cos,sin: (1, 1, T, D)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class SwiGLU(nn.Module):
    def __init__(self, d_model, h_f):
        super().__init__()
        self.w1 = nn.Linear(d_model, h_f, bias=False)
        self.w2 = nn.Linear(d_model, h_f, bias=False)
        self.w3 = nn.Linear(h_f, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def ffn_hidden_size(d_model: int) -> int:
    # paper's rule: hf = ceil((8*d_model/3)/64)*64
    return ((8 * d_model // 3 + 63) // 64) * 64


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,D)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE on q,k (cos/sin already sliced to T in LM forward)
        q, k = apply_rope(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,T,T)
        att = att.float()
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1).to(dtype=q.dtype)
        y = att @ v  # (B,H,T,D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class BiSelfAttention(nn.Module):
    """Same thing but no causal mask (for diffusion MDM)."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.float()
        att = F.softmax(att, dim=-1).to(dtype=q.dtype)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, causal: bool):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads) if causal else BiSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        hf = ffn_hidden_size(d_model)
        self.mlp = SwiGLU(d_model, hf)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model=512, n_layers=8, n_heads=8, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, "RoPE requires even d_head"

        self.tok = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([Block(d_model, n_heads, causal=causal) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        # LM head: weight tying with token embedding (F.linear(x, self.tok.weight) in forward)

        # RoPE cache (cos/sin) with shape (1,1,seq_len,d_head)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        t = torch.arange(seq_len).float()
        freqs = torch.einsum("t,f->tf", t, inv_freq)  # (T, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)       # (T, d_head)
        cos = emb.cos()[None, None, :, :]             # (1,1,T,d_head)
        sin = emb.sin()[None, None, :, :]
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.seq_len

        x = self.tok(idx)  # (B,T,C)

        cos = self.rope_cos[:, :, :T, :].to(dtype=x.dtype)
        sin = self.rope_sin[:, :, :T, :].to(dtype=x.dtype)

        for blk in self.blocks:
            x = blk(x, cos, sin)

        x = self.norm_f(x)
        logits = F.linear(x, self.tok.weight)  # (B,T,V) weight tying
        return logits
