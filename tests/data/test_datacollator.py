from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm_training.data.datacollator import collate_tensors
from llm_training.data.datasets import WikiDataset

def test_collate_tensors():

    BSZ = 32

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    context_length = 1000
    dataset = WikiDataset(context_length, tokenizer)

    tokens = [dataset[i] for i in range(BSZ)]
    pad_token = tokenizer(tokenizer.pad_token)["input_ids"][0]
    eos_token = tokenizer(tokenizer.eos_token)["input_ids"][0]
    sample, label = collate_tensors(tokens, eos_token, pad_token)

    assert isinstance(sample, torch.Tensor)
    assert sample.size(0) == BSZ

    collate_fn = partial(collate_tensors, eos_token=eos_token, pad_token=pad_token)
    dataloader = DataLoader(dataset, BSZ, collate_fn=collate_fn)

    samples, labels,  = next(iter(dataloader))
    assert isinstance(samples, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert samples.shape == (BSZ, context_length)
    assert labels.shape == (BSZ, context_length)
