import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):

    def __init__(self, voc_size: int, hidden_dim: int):
        super().__init__()

        self.voc_size = voc_size
        self.embedding = nn.Linear(voc_size, hidden_dim)

    def forward(self, words: torch.Tensor):
        # bs (batch_size), n (context_length), voc_size

        return self.embedding(words) # bs, n, d (hidden_dim)
