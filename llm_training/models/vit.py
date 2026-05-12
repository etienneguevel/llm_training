import torch
import torch.nn as nn

from llm_training.layers.embeddings import ConvEmbedding
from llm_training.layers.transformer_layer import TransformerLayer


class VisionTransformer(nn.Module):

    def __init__(
        self, 
        dim: int, 
        ffn_dim: int, 
        num_heads: int, 
        kv_heads: int, 
        emb_size: int, 
        in_channels: int,
        num_layers: int,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.emb_size = emb_size

        assert (num_heads % kv_heads) == 0
        self.num_heads = num_heads
        self.kv_heads = kv_heads

        self.embedding_layer = ConvEmbedding(emb_size, dim, in_channels)
        self.layers = nn.ModuleList(
            [TransformerLayer(dim, ffn_dim, num_heads, kv_heads, decoder=False) for _ in range(num_layers)]
        )

        self.last_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor):
        x = self.embedding_layer(tokens)

        for l in self.layers:
            x = l(x)

        logits = self.head(self.last_norm(x))

        return logits
    
