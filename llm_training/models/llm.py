import torch
import torch.nn as nn

from llm_training.layers.embeddings import EmbeddingLayer, TiedUnembeddingLayer
from llm_training.layers.transformer_layer import TransformerLayer


class LlmTransformer(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_heads: int,
        ffn_dim:int,
        voc_size: int,
        num_layers: int,
        tied_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = EmbeddingLayer(voc_size, dim)
        self.layers = nn.ModuleList([
            TransformerLayer(dim, ffn_dim, num_heads, kv_heads) for i in range(num_layers)
        ])
        self.last_norm = nn.RMSNorm(dim)

        if tied_embeddings:
            self.head = TiedUnembeddingLayer(self.embedding)

        else:
            self.head = nn.Linear(dim, voc_size)


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)

        for layer in self.layers:
            x = layer(x)

        logits = self.head(self.last_norm(x))

        return logits