import os
from functools import partial

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from llm_training import BASE_DIR
from llm_training.data.datacollator import collate_tensors, collate_tensors_jagged
from llm_training.data.datasets import WikiDataset
from llm_training.models.llm import LlmTransformer
from llm_training.train.utils import setup_model_resuming, get_args, get_device


def train(cfg):
    # Choose the device to train on
    device = get_device()

    # Create the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)

    # Load the dataset
    dataset = WikiDataset(cfg.train.context_length, tokenizer)

    # Create the loader
    pad_token = tokenizer(tokenizer.pad_token)["input_ids"][0]  # Will be ignored by CE
    eos_token = tokenizer(tokenizer.eos_token)["input_ids"][0]
    voc_size = len(tokenizer)

    if cfg.train.datacollator == "jagged":
        collate_fn = partial(collate_tensors_jagged, eos_token=eos_token)

    else:
        collate_fn = partial(collate_tensors, pad_token=pad_token, eos_token=eos_token)

    loader = DataLoader(dataset, cfg.train.batch_size, collate_fn=collate_fn)

    # Init the model, setup the optimizer
    model = LlmTransformer(
        cfg.model.dim,
        cfg.model.num_heads,
        cfg.model.kv_heads,
        cfg.model.ffn_dim,
        voc_size,
        cfg.model.num_layers,
        cfg.model.tied_embeddings,
    )
    print(model)

    optimizer = AdamW(model.parameters(), cfg.train.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setup the saving dirs
    setup_model_resuming(model, optimizer, cfg.train.save_path)

    # Setup the scaler for mixed_precision
    data_type = torch.bfloat16 if cfg.train.mixed_precision else torch.float32

    # Launch the training
    model = model.to(device)
    model.train()

    tqdm_bar = tqdm(loader, total=cfg.train.tokens_aim)
    token_count = 0
    token_count_total = 0
    log_loss = total_loss = 0

    for batch in tqdm_bar:
        # Unpack the batch
        batch = tuple(el.to(device) for el in batch)
        samples, labels = batch

        token_bsz = (samples != pad_token).sum().item()
        token_count += token_bsz

        with torch.autocast(device, data_type, enabled=cfg.train.mixed_precision):
            logits = model(samples)
            loss = loss_fn(logits.view(-1, voc_size), labels.view(-1))

        total_loss += loss.item()

        loss.backward()
        tqdm_bar.update(token_bsz)
        tqdm_bar.set_postfix({"num_tokens": token_bsz, "token_count": token_count, "loss": log_loss})
        del loss, logits, samples, labels, batch
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

        if token_count >= cfg.train.global_batch_size:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            token_count_total += token_count
            token_count = 0
            log_loss = total_loss
            total_loss = 0

            if token_count_total >= cfg.train.tokens_aim:
                break
                


def main():
    args = get_args()
    cfg_path = args.config_file

    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config.yaml")
    if os.path.exists(cfg_path):
        training_config = OmegaConf.load(cfg_path)
        cfg = OmegaConf.merge([cfg, training_config])

    else:
        print(f"No config found at {cfg_path}, executing with default_config.")

    train(cfg)


if __name__ == "__main__":
    main()
