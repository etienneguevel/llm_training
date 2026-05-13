from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mask(n):
    mask = torch.ones((n, n)).triu(diagonal=1).bool()

    return mask

@lru_cache(1)
def compute_rotations(n: int, d: int, base: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
    thetas = base ** (-(torch.arange(0, d, 2)) / d) # (d/2)
    ind_t = torch.arange(n).unsqueeze(-1).to(thetas.device) # (n, 1)

    cos_t = torch.cos(ind_t * thetas.unsqueeze(0)).repeat_interleave(2, -1) # (n, d) 
    sin_t = torch.sin(ind_t * thetas.unsqueeze(0)).repeat_interleave(2, -1) # (n, d) 

    return cos_t, sin_t


def rotate_half_inverted(x: torch.Tensor):
    # bs, n, d
    x1 = x[..., ::2] # bs, n, d/2
    x2 = x[..., 1::2] # bs, n, d/2

    o = torch.stack([-x2, x1], dim=-1).flatten(-2) # bs, n, d
    return o


def apply_rotation(x: torch.Tensor, max_length: int, d: int, end: int, start: int = 0) -> torch.Tensor:
    # Compute the angles and select the part interesting us
    cos, sin = compute_rotations(max_length, d)
    cos = cos[start:end, :].to(x.device)
    sin = sin[start:end, :].to(x.device)

    x_inv = rotate_half_inverted(x)

    return x * cos + x_inv * sin
    

class RMSnorm(nn.Module):

    def __init__(self, dim: int, epsilon: float = 1e-9):
        super().__init__()
        self.dim = torch.tensor([dim])
        self.epsilon = torch.tensor([epsilon])
        self.g = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = ((x ** 2).sum(-1) / self.dim + self.epsilon).sqrt().unsqueeze(-1)
        return (x / rms) * self.g


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d: int,
        heads: int,
        kv_heads: int,
        decoder: bool,
        max_length: int,
        max_bsz: int
    ):
        super().__init__()
        self.decoder = decoder
        self.max_length = max_length
        self.max_bsz = max_bsz

        # Assert that the number of heads divide dim
        assert (d % heads) == 0, f"Please give a number of heads {heads} that divides d {d}"
        self.dim_heads = d // heads
        self.num_heads = heads
        self.q = nn.Linear(d, d)

        # Assert that the number of kv_heads divide q heads
        assert (heads % kv_heads) == 0, f"Please give a number of kv heads {kv_heads} that divides heads {heads}"
        self.kv_heads = kv_heads
        self.kv_ratio = heads // kv_heads
        self.kv = nn.Linear(d, 2 * self.dim_heads * kv_heads)

        # Make a normalization layer
        self.norm = nn.RMSNorm(d)

        # Make a linear output projection
        self.Wo = nn.Linear(d, d)

        # Make a buffer to register the KV cache
        self.register_buffer(
            "kv_cache",
            torch.zeros((max_bsz, max_length, 2 * self.dim_heads * kv_heads))
        )

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        # x.shape : bsz, n, d 
        bsz, n, d = x.size()
        
        # normalize the input
        x = self.norm(x)

        # Compute queries
        q = self.q(x) # bsz, n, d
        q = q.unflatten(-1, (self.num_heads, self.dim_heads)) # bsz, n, heads, dim_heads
        q = q.transpose(-3, -2) # bsz, heads, n, dim_head

        # Compute kv and cache them
        kv = self.kv(x) # bsz, n, 2 * kv_heads * dim_heads
        self.kv_cache[:bsz, pos:pos+n, :] = kv

        # Uncache
        kv = self.kv_cache[:bsz, :pos+n, :]
        kv = kv.unflatten(-1, (self.kv_heads, 2 * self.dim_heads)).transpose(-3, -2) # bsz, kv_heads, pos+n, 2 * dim_heads
        kv = kv.repeat_interleave(self.kv_ratio, -3) # bsz, heads, pos+n, 2 * dim_heads
        k, v = kv.chunk(2, -1) # (bsz, heads, pos+n, dim_heads), (bsz, heads, pos+n, dim_heads)

        # Compute the attention
        q = apply_rotation(q, self.max_length, self.dim_heads, pos+n, pos)
        k = apply_rotation(k, self.max_length, self.dim_heads, pos+n)
        attn = (q @ k.transpose(-2, -1) / self.dim_heads ** 0.5) # (bsz, heads, n, pos+n)

        # Mask the attention if necessary
        if (self.decoder) & (n > 1):
            mask = create_mask(x.size(-2)).to(x.device)
            attn = attn.masked_fill(mask, -torch.inf)

        attn = attn.softmax(-1) # (bsz, heads, n, pos+n)

        # Compute the matrix update
        o = attn @ v # (bsz, heads, n, dim_heads)
        o = o.transpose(-3, -2).flatten(-2, -1) # (bsz, n, d)
        
        o = self.Wo(o) # (bsz, n, d)

        return o
    

class TPMultiHeadAttention(nn.Module):

    def __init__(
        self,
        d: int,
        heads: int,
        kv_heads: int,
        decoder: bool,
        world_size: int,
        max_length: int,
        max_bsz: int
    ):
        super().__init__()
        self.decoder = decoder
        self.max_length = max_length
        self.max_bsz = max_bsz

        # Assert that the number of heads divide dim
        assert (d % heads) == 0, f"Please give a number of heads {heads} that divides d {d}"
        self.dim_heads = d // heads
        
        assert (heads % world_size) == 0
        self.world_size = world_size
        self.num_heads = heads // world_size

        self.q = nn.Linear(d, self.num_heads * self.dim_heads)

        # Assert that the number of kv_heads divide q heads
        assert (heads % kv_heads) == 0, f"Please give a number of kv heads {kv_heads} that divides heads {heads}"
        self.kv_ratio = heads // kv_heads
        self.kv_heads = kv_heads // world_size
        self.kv = nn.Linear(d, 2 * self.dim_heads * self.kv_heads)

        # Make a normalization layer
        self.norm = nn.RMSNorm(d)

        # Make a linear output projection
        self.Wo = nn.Linear(self.num_heads * self.dim_heads, d)

        # Make a buffer to register the KV cache
        self.register_buffer(
            "kv_cache",
            torch.zeros((max_bsz, max_length, 2 * self.dim_heads * kv_heads))
        )


    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        # x.shape : bsz, n, d 
        # normalize the input
        bsz, n, _ = x.size()
        x = self.norm(x)
        mask = create_mask(x.size(-2))

        q = self.q(x) # bsz, n, d
        q = q.unflatten(-1, (self.num_heads, self.dim_heads)) # bsz, n, heads, dim_heads
        q = q.transpose(-3, -2) # bsz, heads, n, dim_heads

        # Compute kv and cache them
        kv = self.kv(x) # bsz, n, 2 * kv_heads * dim_heads
        self.kv_cache[:bsz, pos:pos+n, :] = kv

        # Uncache
        kv = self.kv_cache[:bsz, :pos+n, :]
        kv = kv.unflatten(-1, (self.kv_heads, 2 * self.dim_heads)).transpose(-3, -2) # bsz, kv_heads, pos+n, 2 * dim_heads
        kv = kv.repeat_interleave(self.kv_ratio, -3) # bsz, heads, pos+n, 2 * dim_heads
        k, v = kv.chunk(2, -1) # (bsz, heads, pos+n, dim_heads), (bsz, heads, pos+n, dim_heads)

        # Compute the attention
        q = apply_rotation(q, self.max_length, self.dim_heads, pos+n, pos)
        k = apply_rotation(k, self.max_length, self.dim_heads, pos+n)
        attn = (q @ k.transpose(-2, -1) / torch.tensor([self.dim_heads]).sqrt())

        # Mask the attention if necessary
        if (self.decoder) & (n > 1):
            attn = attn.masked_fill(mask, -torch.inf)

        attn = attn.softmax(-1) # (bsz, heads, n, n)

        # Compute the matrix update
        o = attn @ v # (bsz, heads, n, dim_heads)
        o = o.transpose(-3, -2).flatten(-2) # (bsz, n, dim_heads * heads)
        
        o = self.Wo(o) # (bsz, n, d)
        torch.distributed.all_reduce(o)

        return o



class Ffn(nn.Module):
    
    def __init__(self, d: int, ffn_dim: int, out_dim: int):
        super().__init__()

        self.d = d
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim

        # Make the first part of the network
        self.W1 = nn.Linear(d, ffn_dim)
        self.W2 = nn.Linear(d, ffn_dim)

        # Make the second part
        self.W3 = nn.Linear(ffn_dim, out_dim)

        # Make a normalization layer
        self.norm = nn.RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        o = self.W3(F.silu(self.W1(x)) * self.W2(x))
        
        return o


class TPFfn(nn.Module):

    def __init__(self, d: int, ffn_dim: int, world_size: int):
        super().__init__()
        self.d = d
        self.ffn_dim = ffn_dim
        self.world_size = world_size

        self.tp_dim = ffn_dim // world_size

        self.up = nn.Linear(d, self.tp_dim)
        self.gate = nn.Linear(d, self.tp_dim)

        self.down = nn.Linear(self.tp_dim, d)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (bs, n, d)
        u = self.up(x) # (bs, n, tp_dim)
        g = self.gate(x) # (bs, n, tp_dim)

        o = self.down(F.silu(u) * g) # (bs, n, d)
        torch.distributed.all_reduce(o) # (bs, n, d)

        return o


class Gate(nn.Module):

    def __init__(self, d: int, num_experts: int, active_experts: int):
        super().__init__()

        self.d = d
        self.num_experts = num_experts
        self.active_experts = active_experts

        self.weights = nn.Parameter(torch.randn(d, num_experts))


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        scores = (x @ self.weights) # (bs, n, num_experts)
        scores, experts = scores.topk(self.active_experts) # (bs, n, active_experts), (bs, n, active_experts)

        scores = scores.softmax(-1) # (bs, n, active_experts)

        return scores, experts
    

class MoE(nn.Module):

    def __init__(
        self,
        d,
        expert_dim,
        num_experts,
        active_experts,
    ):
        super().__init__()
        self.d = d
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.active_experts = active_experts

        # Make the layers for the network
        self.gate = Gate(d, num_experts, active_experts)
        self.experts = nn.ModuleList([Ffn(d, expert_dim, d) for _ in range(num_experts)])


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        # (bs, n, d)
        dims = x.size()

        x = x.view(-1, self.d)
        scores, routed_experts = self.gate(x) # (bs * n, active_experts), (bs * n, active_experts)

        y = torch.zeros_like(x) # (bs * n, d)

        for i, expert in enumerate(self.experts):
            idx, pos = torch.where(routed_experts == i) # (num_tokens_expert), (num_tokens_expert)
            out_expert = expert(x[idx]) # (num_tokens_expert, d)

            update = out_expert * scores[idx, pos] # (num_tokens_expert, d)

            y[idx] += update

        y = y.view(dims)
        return y



class TransformerLayer(nn.Module):

    def __init__(
        self,
        d: int,
        heads: int,
        kv_heads: int,
        ffn_dim: int,
        decoder: bool,
        max_length: int,
        max_bsz: int
    ):
        super().__init__()
        self.max_length = max_length
        self.mha = MultiHeadAttention(d, heads, kv_heads, decoder, max_length, max_bsz)
        self.ffn = Ffn(d, ffn_dim, d)

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        # do the attention part
        x = x + self.mha(x, pos)

        # do the ffn part
        x = x + self.ffn(x)

        return x


class TPTransformerLayer(nn.Module):

    def __init__(
        self,
        d: int,
        heads: int,
        kv_heads: int,
        ffn_dim: int,
        decoder: bool,
        max_length: int,
        max_bsz: int,
        world_size: int,
    ):
        super().__init__()
        self.max_length = max_length
        self.world_size = world_size

        self.mha = TPMultiHeadAttention(d, heads, kv_heads, decoder, world_size, max_length, max_bsz)
        self.ffn = TPFfn(d, ffn_dim, world_size)

    def forward(self, x: torch.Tensor, pos: int = 0):
        # do the attention part
        x = x + self.mha(x, pos)

        # do the ffn part
        x = x + self.ffn(x)

        return x


class TransformerModel(nn.Module):

    def __init__(
        self,
        d: int,
        heads: int,
        kv_heads: int,
        ffn_dim: int,
        num_layers: int,
        voc_size: int,
        decoder: bool,
        max_length: int,
        max_bsz: int,
    ):
        
        super().__init__()
        self.max_length = max_length
        self.max_bsz = max_bsz
        self.embedding = nn.Embedding(voc_size, d)

        # Make the transformer layers
        self.attention_blocks = nn.ModuleList([TransformerLayer(d, heads, kv_heads, ffn_dim, decoder, max_length, max_bsz) for _ in range(num_layers)])

        # Make the projection head
        self.last_norm = RMSnorm(d)
        self.projection = nn.Linear(d, voc_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.embedding(x)
        for l in self.attention_blocks:
            x = l(x)

        return self.projection(self.last_norm(x))
    
