import torch
import torch.nn as nn

from llm_training.layers.transformer_layer import TransformerLayer


class LlmTransformer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        kv_heads: int,
        ffn_dim:int,
        voc_size: int,
        num_layers: int,
    ):
        super().__init__()
        layers = [nn.Linear(voc_size, hidden_dim)]
        layers = layers + [
            TransformerLayer(hidden_dim, ffn_dim, num_heads, kv_heads) for i in range(num_layers)
        ]
        layers.append(nn.Linear(hidden_dim, voc_size))

        self.layers = nn.ModuleList(layers)


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            tokens = layer(tokens)

        return tokens