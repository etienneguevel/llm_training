import torch

from llm_training.models.llm import LlmTransformer


HIDDEN_DIM = 512
NUM_HEADS = 16
KV_HEADS = 4
VOC_SIZE = 10000
NUM_LAYERS = 8

def test_llm_forward():
    model = LlmTransformer(
        HIDDEN_DIM,
        NUM_HEADS,
        KV_HEADS,
        HIDDEN_DIM * 4,
        VOC_SIZE,
        NUM_LAYERS
    )

    input_tensor = torch.randn((64, 100, VOC_SIZE))
    output = model(input_tensor)

    assert output.shape == (64, 100, VOC_SIZE)