import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):

    def __init__(self, voc_size: int, dim: int):
        super().__init__()

        self.voc_size = voc_size
        self.weights = nn.Parameter(torch.empty(voc_size, dim))

    def forward(self, words: torch.Tensor):
        # bs (batch_size), n (context_length), voc_size
        x = F.embedding(words, self.weights)

        return x # bs, n, dim


class TiedUnembeddingLayer(nn.Module):

    def __init__(self, embedding_layer: EmbeddingLayer):
        super().__init__()
        self.embedding_layer = embedding_layer

    def forward(self, x: torch.Tensor):

        return x @ self.embedding_layer.weights.T

class ConvEmbedding(nn.Module):

    def __init__(self, num_pixels: int, dim: int, in_channels: int = 3):
        self.num_pixels = num_pixels
        self.dim = dim

        self.weights = nn.Conv2d(
            in_channels,
            dim,
            num_pixels,
            num_pixels,
        )

    def forward(self, x: torch.Tensor):

        x = self.weights(x) # bs, n1, n2, dim
        x = x.flatten(start_dim=1, end_dim=2) # bs, n1*n2, dim

        return x