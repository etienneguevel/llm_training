import torch
from torch.nn.utils.rnn import pad_sequence


def collate_tensors(tokens: list[torch.Tensor], eos_token: int, pad_token: int):
    
    samples = pad_sequence(tokens, batch_first=True, padding_value=pad_token)
    labels = pad_sequence([torch.cat(
        (t[1:], torch.tensor([eos_token]))
    ) for t in tokens], batch_first=True, padding_value=pad_token)

    return samples, labels