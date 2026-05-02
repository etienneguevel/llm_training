import torch

from llm_training.layers.embeddings import EmbeddingLayer, TiedUnembeddingLayer

VOC_SIZE = 10000
DIM = 512

def test_tied_embeddings():
    emb = EmbeddingLayer(VOC_SIZE, DIM)
    assert emb.weights.shape == (VOC_SIZE, DIM)

    unemb = TiedUnembeddingLayer(emb)

    x = torch.randn((10, 1000, DIM))
    log = unemb(x)

    assert log.shape == (10, 1000, VOC_SIZE)