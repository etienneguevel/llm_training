import torch
import torch.nn as nn


def make_attention_mask(indices):
    n = indices.sum().item()
    mask = torch.zeros((n, n))
    idx = 0
    for i in indices:
        mask[idx:idx+i, idx:idx+i] = 1
        idx += i
    
    mask = mask.triu().T
    return mask


class AttentionLayer(nn.Module):

    def __init__(self, dim: int, num_heads: int, kv_heads: int, decoder: bool):
        super().__init__()

        assert (dim % num_heads) == 0 # assert that the number of heads divide the dim
        assert (num_heads % kv_heads) == 0 # assert that the number of KV heads divide the number of Q heads

        self.kv_ratio =  num_heads // kv_heads
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.decoder = decoder

        self.norm = nn.RMSNorm(dim)

        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, int(self.head_dim * self.kv_heads))
        self.V = nn.Linear(dim, int(self.head_dim * self.kv_heads))

        self.Wo = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor):
        # tokens: bs (batch_size), n (context_length), d (dim)
        tokens = self.norm(tokens)

        q = self.Q(tokens).unflatten(-1, (self.num_heads, self.head_dim)).permute(0, 2, 1, 3) # bs, num_heads, n, head_dim
        k = self.K(tokens).unflatten(-1, (self.kv_heads, self.head_dim)).permute(0, 2, 1, 3) # bs, kv_heads, n, head_dim
        v = self.V(tokens).unflatten(-1, (self.kv_heads, self.head_dim)).permute(0, 2, 1, 3) # bs, kv_heads, n, head_dim

        # Compute the attention
        if tokens.is_nested:
            k = torch.cat([k for _ in range(self.kv_ratio)], dim=1)
            v = torch.cat([v for _ in range(self.kv_ratio)], dim=1)
            a = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.decoder)

        else:
            a = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.decoder, enable_gqa=True)


        # Do the output projection
        a = a.transpose(1, 2).flatten(start_dim=2) # bs, n, d
        a = self.Wo(a)

        return a


class Ffn(nn.Module):

    def __init__(self, dim: int, inside_dim: int):
        super().__init__()
        self.W1 = nn.Linear(dim, inside_dim)
        self.activation = nn.SiLU()
        self.W2 = nn.Linear(dim, inside_dim)
        self.Wout = nn.Linear(inside_dim, dim)

        self.norm = nn.RMSNorm(dim)

    def forward(self, tokens: torch.Tensor):
        # tokens: bs (batch_size), n (context_length), d (dim)
        tokens = self.norm(tokens) # bs, n, d

        emb = self.activation(self.W1(tokens)) * self.W2(tokens) # bs, n, d
        output = self.Wout(emb) # bs, n, d

        return output


class TransformerLayer(nn.Module):
    def __init__(
            self, 
            dim: int,
            ffn_dim: int,
            num_heads: int,
            kv_heads: int,
            decoder:bool,
        ):
        super().__init__()

        self.attention_block = AttentionLayer(dim, num_heads, kv_heads, decoder=decoder)
        self.ffn_block = Ffn(dim, ffn_dim)

    def forward(self, tokens: torch.Tensor):
        # tokens: bs (batch_size), n (context_length), d (dim)
        tokens = tokens + self.attention_block(tokens)
        tokens = tokens + self.ffn_block(tokens)

        return tokens