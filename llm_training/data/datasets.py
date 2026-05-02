import torch
from torch.utils.data import Dataset

from datasets import load_dataset


class WikiDataset(Dataset):
    path = "allenai/dolmino-mix-1124"
    name = "wiki"

    def __init__(self, context_length: int, tokenizer):
        super().__init__()
        self.data = load_dataset(self.path, self.name)["train"]
        self.context_length = context_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]["text"]
        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

        return tokens[:min((len(tokens), self.context_length))]
