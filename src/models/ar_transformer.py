import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B,T,C)
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=-1)

        # (B, nh, T, hd)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # SDPA causal: fast on H100
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True,
        )  # (B, nh, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = mlp_mult * d_model
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.drop(x)

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_mult=4, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ARTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)  # simplest; you can swap RoPE later
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying (optional, standard)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        # idx: (B,T)
        B, T = idx.shape
        assert T <= self.seq_len
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)
        return logits
