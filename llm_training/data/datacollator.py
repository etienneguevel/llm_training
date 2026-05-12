import torch
from torch.nn.utils.rnn import pad_sequence


def collate_tensors(tokens: list[torch.Tensor], eos_token: int, pad_token: int):
    
    samples = pad_sequence(tokens, batch_first=True, padding_value=pad_token)
    labels = pad_sequence([torch.cat(
        (t[1:], torch.tensor([eos_token]))
    ) for t in tokens], batch_first=True, padding_value=-100)

    return samples, labels

def collate_tensors_jagged(tokens: list[torch.Tensor], eos_token: int):

    samples = torch.nested.nested_tensor(tokens, layout=torch.jagged)
    labels = torch.nested.nested_tensor([torch.cat(
        (t[1:], torch.tensor([eos_token]))
    ) for t in tokens], layout=torch.jagged)

    return samples, labels

def collate_tensors_stacked(tokens: list[torch.Tensor], eos_token: int):
    labels = torch.cat(
        [torch.cat([t[1:], torch.tensor([eos_token])]) for t in tokens]
    )
    samples = torch.cat(tokens)
    indices = torch.tensor( [t.size(0) for t in tokens])

    return samples, labels, indices