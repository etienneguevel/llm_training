import torch
import torch.nn as nn

def make_mask(tokens: torch.Tensor):
    bs, n, _ = tokens.shape
    
    mask = torch.stack(
            [torch.Tensor([j <= i for j in range(n)]) for i in range(n)]
        )[None, None, ...].to(bool)
    
    return mask

class AttentionLayerDecoder(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, kv_heads: int):
        super().__init__()

        assert (hidden_dim % num_heads) == 0 # assert that the number of heads divide the hidden_dim
        assert (num_heads % kv_heads) == 0 # assert that the number of KV heads divide the number of Q heads

        self.kv_ratio =  num_heads // kv_heads
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.norm = nn.RMSNorm(hidden_dim)

        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, int(self.head_dim * self.kv_heads))
        self.V = nn.Linear(hidden_dim, int(self.head_dim * self.kv_heads))

        self.Wo = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor, return_attention_map: bool = False):
        # tokens: bs (batch_size), n (context_length), d (hidden_dim)
        tokens = self.norm(tokens)

        q = self.Q(tokens).unflatten(-1, (self.num_heads, self.head_dim)).permute(0, 2, 1, 3) # bs, num_heads, n, head_dim
        k = self.K(tokens).unflatten(-1, (self.kv_heads, self.head_dim)).permute(0, 2, 1, 3) # bs, kv_heads, n, head_dim
        v = self.V(tokens).unflatten(-1, (self.kv_heads, self.head_dim)).permute(0, 2, 1, 3) # bs, kv_heads, n, head_dim

        # Duplicate the k and v heads to match the number of q heads
        k = torch.cat([k for i in range(self.kv_ratio)], dim=1) # bs, num_heads, n, head_dim
        v = torch.cat([v for i in range(self.kv_ratio)], dim=1) # bs, num_heads, n, head_dim
        print(k.shape)

        # Compute the attention
        mask = make_mask(tokens)  # 1, n, n

        attn = q @ k.transpose(2, 3) # bs, num_heads, n, n
        attn = attn.masked_fill(mask, -1e9).softmax(-1) # bs, num_heads, n, n

        a = (attn @ v).transpose(1, 2).flatten(start_dim=2) # bs, n, d

        # Do the output projection
        a = self.Wo(a)

        if return_attention_map:
            return a, attn

        return a


class Ffn(nn.Module):

    def __init__(self, hidden_dim: int, inside_dim: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, inside_dim)
        self.activation = nn.SiLU()
        self.W2 = nn.Linear(hidden_dim, inside_dim)
        self.Wout = nn.Linear(inside_dim, hidden_dim)

        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor):
        # tokens: bs (batch_size), n (context_length), d (hidden_dim)
        tokens = self.norm(tokens) # bs, n, d

        emb = self.activation(self.W1(tokens)) * self.W2(tokens) # bs, n, d
        output = self.Wout(emb) # bs, n, d

        return output


class TransformerLayer(nn.Module):
    def __init__(
            self, 
            hidden_dim,
            ffn_dim,
            num_heads,
            kv_heads,
        ):
        super().__init__()

        self.attention_block = AttentionLayerDecoder(hidden_dim, num_heads, kv_heads)
        self.ffn_block = Ffn(hidden_dim, ffn_dim)

    def forward(self, tokens: torch.Tensor):
        # tokens: bs (batch_size), n (context_length), d (hidden_dim)
        tokens = tokens + self.attention_block(tokens)
        tokens = tokens + self.ffn_block(tokens)

        return tokens