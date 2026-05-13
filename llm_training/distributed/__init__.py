import os
from datetime import timedelta

import torch
import torch.distributed as dist


_MAIN_RANK = 0

def enable(timeout_minutes: int = 20, main_rank: int = 0):
    dist.init_process_group(
        backend="nccl",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
        timeout=timedelta(minutes=timeout_minutes)
    )

    set_main_rank(main_rank)


def is_enabled() -> bool:
    
    return dist.is_available() and dist.is_initialized()


def get_global_size() -> int:

    return dist.get_world_size() if is_enabled() else 0


def get_global_rank() -> int:

    return dist.get_rank() if is_enabled() else 0


def set_main_rank(main_rank: int) -> None:

    assert main_rank <= get_global_size()
    global _MAIN_RANK
    _MAIN_RANK = main_rank


def is_main_rank() -> bool:

    return get_global_rank() == _MAIN_RANK
