import os
from argparse import ArgumentParser
from functools import partial

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from llm_training import BASE_DIR
from llm_training.data.datacollator import collate_tensors
from llm_training.data.datasets import WikiDataset
from llm_training.models.llm import LlmTransformer


def get_args():

    args_parser = ArgumentParser()
    args_parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the config file to use for the training.",
        default="",
    )

    return args_parser.parse_args()


def train(cfg):
    # Choose the device to train on
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.mps.is_available() else "cpu")
    )

    # Create the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)

    # Load the dataset
    dataset = WikiDataset(cfg.train.context_length, tokenizer)

    # Create the loader
    pad_token = -100  # Will be ignored by CE
    eos_token = tokenizer(tokenizer.eos_token)["input_ids"][0]

    collate_fn = partial(collate_tensors, pad_token=pad_token, eos_token=eos_token)

    loader = DataLoader(dataset, cfg.train.batch_size, True, collate_fn=collate_fn)

    # Init the model, setup the optimizer
    model = LlmTransformer(
        cfg.model.dim,
        cfg.model.num_heads,
        cfg.model.kv_heads,
        cfg.model.ffn_dim,
        tokenizer.vocab_size,
        cfg.model.num_layers,
        cfg.model.tied_embeddings,
    )
    print(model)

    optimizer = AdamW(model.parameters(), cfg.train.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setup the saving dirs
    saving_dir = cfg.train.save_path
    if os.path.isdir(saving_dir):
        print(f"Found an existing folder at {saving_dir}")
        ckpts = [i for i in os.listdir(saving_dir) if i.endswith(".pt")]
        if len(ckpts) > 0:
            ckpts.sort(key=lambda el: int(el.replace(".pt", "")), reverse=True)
            last_ckpt = ckpts[0]
            print(
                f"Found checkpoints, loading from {os.path.join(saving_dir, last_ckpt)}"
            )

            state_dict = torch.load(os.path.join(saving_dir, last_ckpt))
            model.load_state_dict(state_dict["model"])

            optimizer.load_state_dict(state_dict["optimizer"])

    else:
        os.makedirs(saving_dir)

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
            # Do the forward pass
            logits = model(samples)
            # Do the backward pass
            loss = loss_fn(logits.flatten(0, 1), labels.flatten())
            total_loss += loss.item()

        loss.backward()
        tqdm_bar.update(token_bsz)
        tqdm_bar.set_postfix({"num_tokens": token_bsz, "token_count": token_count, "loss": log_loss})
        del loss, logits, samples, labels
        torch.mps.empty_cache()

        if token_count >= cfg.train.global_batch_size:
            optimizer.step()
            optimizer.zero_grad()

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
