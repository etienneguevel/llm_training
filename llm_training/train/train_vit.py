import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from llm_training import BASE_DIR
from llm_training.data.transforms import image_transform_training
from llm_training.models.vit import VisionTransformer
from llm_training.train.utils import get_device, setup_model_resuming, get_args


def train(cfg):
    # Get the training device
    device = get_device()
    print(f"Training on device: {device}")

    # Make the dataset and loader
    train_tf = image_transform_training(cfg.train.resize_shape, cfg.train.croping_shape)

    train_dataset = ImageFolder(cfg.data.path, train_tf)
    train_loader = DataLoader(train_dataset, cfg.train.batch_size)

    # Make the model
    model = VisionTransformer(
        cfg.model.dim,
        cfg.model.ffn_dim,
        cfg.model.num_heads,
        cfg.model.kv_heads,
        cfg.model.emb_size,
        cfg.model.in_channels,
        cfg.model.num_layers,
    ).to(device)


    # Make the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), cfg.train.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Resume training if there are already checkpoints
    setup_model_resuming(model, optimizer, cfg.train.save_path)
    model.train()


    for b in tqdm(train_loader):
        b = tuple(el.to(device) for el in b)
        inp, label = b

        logits = model(inp)
        print(logits.shape, label.shape)
        loss = loss_fn(logits, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def main():

    args = get_args()

    cfg_path = args.config_file

    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config_vit.yaml")
    if os.path.exists(cfg_path):
        training_config = OmegaConf.load(cfg_path)
        cfg = OmegaConf.merge([cfg, training_config])

    else:
        print(f"No config found at {cfg_path}, executing with default_config.")

    train(cfg)


if __name__ == "__main__":
    main()
