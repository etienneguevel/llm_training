import torch

from llm_training.models.llm import LlmTransformer


HIDDEN_DIM = 512
NUM_HEADS = 16
KV_HEADS = 4
VOC_SIZE = 10000
NUM_LAYERS = 8

BATCH_SIZE = 2
CONTEXT_LENGTH = 1000

MAX_LENGTH = 10000
MAX_BSZ = 4


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

def test_llm_forward():
    model = LlmTransformer(
        HIDDEN_DIM,
        NUM_HEADS,
        KV_HEADS,
        HIDDEN_DIM * 4,
        VOC_SIZE,
        NUM_LAYERS,
        VOC_SIZE - 1,
        MAX_LENGTH,
        MAX_BSZ
    ).to(device)

    input_tensor = torch.randint(0, VOC_SIZE, (BATCH_SIZE, CONTEXT_LENGTH)).to(device)
    output = model(input_tensor)

    assert output.shape == (BATCH_SIZE, CONTEXT_LENGTH, VOC_SIZE)

def test_llm_forward_tiedembeddings():
    model = LlmTransformer(
        HIDDEN_DIM,
        NUM_HEADS,
        KV_HEADS,
        HIDDEN_DIM * 4,
        VOC_SIZE,
        NUM_LAYERS,
        VOC_SIZE - 1,
        MAX_LENGTH,
        MAX_BSZ,
        True
    ).to(device)

    input_tensor = torch.randint(0, VOC_SIZE, (BATCH_SIZE, CONTEXT_LENGTH)).to(device)
    output = model(input_tensor)

    assert output.shape == (BATCH_SIZE, CONTEXT_LENGTH, VOC_SIZE)


def test_llm_generate():
    model = LlmTransformer(
        HIDDEN_DIM,
        NUM_HEADS,
        KV_HEADS,
        HIDDEN_DIM * 4,
        VOC_SIZE,
        NUM_LAYERS,
        VOC_SIZE - 1,
        MAX_LENGTH,
        MAX_BSZ
    ).to(device)

    input_tensor = torch.randint(0, VOC_SIZE, (BATCH_SIZE, CONTEXT_LENGTH)).to(device)

    output = model.generate(input_tensor, 10, verbose=True)