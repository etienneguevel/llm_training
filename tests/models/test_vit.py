import torch

from llm_training.models.vit import VisionTransformer


DIM = 512
FFN_DIM = 3 * DIM
NUM_HEADS = KV_HEADS = 8
IN_CHANNELS = 3
EMB_SIZE = 16
NUM_LAYERS = 8

def test_vit_forward():

    vit = VisionTransformer(
        DIM, 
        FFN_DIM, 
        NUM_HEADS, 
        KV_HEADS, 
        EMB_SIZE,
        IN_CHANNELS,
        NUM_LAYERS
    )

    BSZ = 2
    IMG_SIZE = 128

    input = torch.randn((BSZ, IN_CHANNELS, IMG_SIZE, IMG_SIZE))
    output = vit(input)

    assert isinstance(output, torch.Tensor)
    assert output.size(-1) == DIM
    