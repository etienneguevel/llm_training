import os
from argparse import ArgumentParser

from omegaconf import OmegaConf
from transformers import AutoTokenizer

import llm_training.distributed as dist
from llm_training.models.llm import LlmTransformer


def get_args():

    args_parser = ArgumentParser()
    args_parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the config file to use for the training.",
        default="",
    )

    args_parser.add_argument(
        "--txt",
        type=str,
        help="Text to use, or path to a file containing the text to use.",
    )

    return args_parser.parse_args()


def main():

    args = get_args()
    cfg = OmegaConf.load(args.config_file)

    if ((tp:=cfg.distributed.tp_size) is not None) | ((dp:=cfg.distributed.dp_size) is not None):
        dist.enable()
        dp = 1 if dp is None else dp
        tp = 1 if tp is None else tp
        
        assert (dp * tp) == dist.get_global_size()

    model = LlmTransformer.init_from_config(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)

    print(model)

    # Open the text
    if os.path.exists(args.txt):
        with open(args.txt) as f:
            txt = f.read()

    else:
        txt = args.txt

    tokens = tokenizer(txt, return_tensors="pt")["input_ids"]

    # Generate the new tokens
    output_tokens = model.generate(tokens, 20)
    print(f"shape of the outpu_tokens {output_tokens.shape} {output_tokens}")
    output_txt = tokenizer.decode(output_tokens)
    
    print(output_txt)

    return output_txt


if __name__ == "__main__":
    main()
