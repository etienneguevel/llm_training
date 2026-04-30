import torch
from llm_training.layers.transformer_layer import AttentionLayerDecoder, Ffn, TransformerLayer


HIDDEN_DIM = 512
NUM_HEADS = 16
    

def test_basic_forward():
    attention_layer = AttentionLayerDecoder(
        HIDDEN_DIM,
        NUM_HEADS,
        NUM_HEADS // 4,
    )

    input_tensor = torch.randn((16, 100, HIDDEN_DIM))
    output_tensor = attention_layer(input_tensor)

    assert output_tensor.shape == (16, 100, HIDDEN_DIM)


def test_attention_softmaxed():
    attention_layer = AttentionLayerDecoder(
        HIDDEN_DIM,
        NUM_HEADS,
        NUM_HEADS
    )

    input_tensor = torch.randn((16, 100, HIDDEN_DIM))
    output_tensor, attn_map = attention_layer.forward(input_tensor, True)

    assert output_tensor.shape == (16, 100, HIDDEN_DIM)
    assert ((attn_map.sum(-1) - 1.).abs() <= 1e-3).all()


def test_ffn_forward():
    ffn_layer = Ffn(HIDDEN_DIM, 2 * HIDDEN_DIM)
    input_tensor = torch.randn((16, 100, HIDDEN_DIM))

    output_tensor = ffn_layer(input_tensor)

    assert output_tensor.shape == (16, 100, HIDDEN_DIM)


def test_transformer_layer():
    transformer_layer = TransformerLayer(
        HIDDEN_DIM,
        2 * HIDDEN_DIM,
        NUM_HEADS,
        NUM_HEADS // 4
    )

    input_tensor = torch.randn((16, 100, HIDDEN_DIM))
    output_tensor = transformer_layer(input_tensor)

    assert output_tensor.shape == (16, 100, HIDDEN_DIM)