import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm_training import BASE_DIR
from llm_training.data.datasets import WikiDataset, WikiDatasetPacked


def test_wikidataset():

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    context_length = 1000
    dataset = WikiDataset(context_length, tokenizer)

    sample = dataset[random.randint(0, len(dataset))]
    assert isinstance(sample, torch.Tensor)


def test_wikidataset_packed():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    context_length = 1000

    dataset = WikiDatasetPacked(context_length, tokenizer, BASE_DIR.parent / "data" / "wikidata")

    tokens, boundaries = dataset[random.randint(0, len(dataset))]
