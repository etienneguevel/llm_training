import torch
import torch.nn as nn

from llm_training.layers.embeddings import EmbeddingLayer, TiedUnembeddingLayer
from llm_training.layers.attention import TransformerLayer, TPTransformerLayer

from tqdm import tqdm
from transformers import AutoTokenizer


class LlmTransformer(nn.Module):

    def __init__(
        self,
        d: int,
        heads: int,
        kv_heads: int,
        ffn_dim:int,
        voc_size: int,
        num_layers: int,
        eos_token: int,
        max_length: int,
        max_bsz: int,
        tied_embeddings: bool = False,
        tp_size: int | None = None,
    ):
        super().__init__()
        self.embedding = EmbeddingLayer(voc_size, d)
        self.eos_token = eos_token
        self.max_length = max_length
        self.max_bsz = max_bsz
        self.layers = nn.ModuleList([
            TransformerLayer(d, heads, kv_heads, ffn_dim, True, max_length, max_bsz)
            if not tp_size else (
                TPTransformerLayer(d, heads, kv_heads, ffn_dim, True, max_length, max_bsz, tp_size)
            )
            for i in range(num_layers)
        ])
        self.last_norm = nn.RMSNorm(d)

        if tied_embeddings:
            self.head = TiedUnembeddingLayer(self.embedding)

        else:
            self.head = nn.Linear(d, voc_size)


    def forward(self, tokens: torch.Tensor, pos: int = 0) -> torch.Tensor:
        # (bs, n)
        x = self.embedding(tokens)

        for layer in self.layers:
            x = layer(x, pos)

        logits = self.head(self.last_norm(x)) # (bs, n, voc_size)

        return logits
    
    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_tokens: int, verbose: bool = False) -> torch.Tensor:
        # (bs, n)
        bs, n = tokens.size()
        eos = torch.zeros(bs, device=tokens.device)
        pos = 0
        pbar = tqdm(range(max_tokens), disable=not verbose)
        all_tokens = tokens

        for _ in pbar:
            print(all_tokens)
            _, n = tokens.size()
            logits = self.forward(tokens, pos)
            new = logits[:, -1:, :].argmax(-1) # (bs, 1)

            # Check if the new tokens are eos tokens, and break if all elements in the batch are done
            eos += (new == self.eos_token).int().squeeze(-1)
            if eos.bool().all():
                break
            
            pos = n
            all_tokens = torch.cat([all_tokens, new], dim=-1) # (bs, n+1)
            tokens = new

        return all_tokens
    

    @classmethod
    def init_from_config(cls, cfg):
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
        eos_token = tokenizer(tokenizer.eos_token)["input_ids"][0]
        voc_size = len(tokenizer)

        return cls(
            cfg.model.dim,
            cfg.model.num_heads,
            cfg.model.kv_heads,
            cfg.model.ffn_dim,
            voc_size,
            cfg.model.num_layers,
            eos_token,
            cfg.model.max_length,
            cfg.model.max_bsz,
            cfg.model.tied_embeddings,
            cfg.distributed.tp_size
        )