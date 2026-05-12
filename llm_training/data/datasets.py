
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

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


class WikiDatasetPacked(Dataset):
    path = "allenai/dolmino-mix-1124"
    name = "wiki"

    def __init__(self, context_length: int, tokenizer, cache_path: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.cache_path = Path(cache_path)

        if not self.cache_path.exists():
            token_iterator = WikiDataset(context_length, tokenizer)
            eos_id = tokenizer(tokenizer.eos_token)["input_ids"][0]
            chunks, chunks_boundaries = self._pack_documents(token_iterator, context_length, eos_id, self.cache_path)
            self.tokens = chunks
            self.flat_boundaries = chunks_boundaries

        else:
            self.tokens = np.load(self.cache_path / "tokens.npy", mmap_mode='r')
            self.flat_boundaries = np.load(self.cache_path / "boundaries.npy",allow_pickle=True)


    def _pack_documents(self, token_iterator, seq_len, eos_token_id, output_path):
        """
        token_iterator: yields 1D numpy arrays of token IDs (one per document)
        seq_len: target packed sequence length (e.g. 4096)
        Writes a memory-mapped file of shape (n_chunks, seq_len) and a
        parallel file of document boundaries per chunk.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        buffer = []           # current chunk being built
        boundaries = [0]      # offsets within the current chunk where docs end
        chunks = []
        chunk_boundaries = []

        for doc in tqdm(token_iterator):
            # Append doc + EOS
            doc_with_eos = np.concatenate([doc, [eos_token_id]])
            remaining = doc_with_eos

            while len(buffer) + len(remaining) >= seq_len:
                take = seq_len - len(buffer)
                buffer.extend(remaining[:take])
                boundaries.append(len(buffer))  # = seq_len
                chunks.append(np.array(buffer, dtype=np.int32))
                chunk_boundaries.append(np.array(boundaries, dtype=np.int32))

                buffer = []
                boundaries = [0]
                remaining = remaining[take:]

            if len(remaining) > 0:
                buffer.extend(remaining)
                boundaries.append(len(buffer))

        # Stack token chunks into a single (n_chunks, seq_len) array
        tokens_array = np.stack(chunks)
        np.save(output_path / "tokens.npy", tokens_array)

        # Boundaries are ragged (variable number of docs per chunk),
        # so save as an object array
        boundaries_array = np.empty(len(chunk_boundaries), dtype=object)
        for i, b in enumerate(chunk_boundaries):
            boundaries_array[i] = b
        np.save(output_path / "boundaries.npy", boundaries_array, allow_pickle=True)

        return tokens_array, chunk_boundaries
    

    def __len__(self):
        return len(self.tokens)
    

    def __getitem__(self, index):
        tokens = self.tokens[index]
        flat_boundaries = self.flat_boundaries[index]

        return tokens, flat_boundaries
