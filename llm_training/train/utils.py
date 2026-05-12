import os
from argparse import ArgumentParser
import torch

def get_args():

    args_parser = ArgumentParser()
    args_parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the config file to use for the training.",
        default="",
    )

    return args_parser.parse_args()


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.mps.is_available() else "cpu")
    )

    return device


def setup_model_resuming(model, optimizer, save_path):
    saving_dir = save_path
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

