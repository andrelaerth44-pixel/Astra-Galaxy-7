import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, seq_len, theta):
    head_dim = q.shape[-1]
    device = q.device
    dtype = q.dtype

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :].to(dtype)
    sin = emb.sin()[None, None, :, :].to(dtype)

    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout, rope_theta):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.rope_theta = rope_theta

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, t, self.rope_theta)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, dim, heads, intermediate_dim, dropout, rope_theta):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout, rope_theta)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, intermediate_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AstraGalaxy7(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg["vocab_size"]
        self.max_seq_len = cfg["max_seq_len"]
        dim = cfg["hidden_size"]

        self.token_embedding = nn.Embedding(self.vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(
                dim,
                cfg["num_heads"],
                cfg["intermediate_size"],
                cfg["dropout"],
                cfg["rope_theta"],
            )
            for _ in range(cfg["num_layers"])
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, self.vocab_size, bias=False)

        # Weight tying reduces parameters and is standard for decoder LMs.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds {self.max_seq_len}")

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
