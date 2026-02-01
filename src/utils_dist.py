import os
import torch
import torch.distributed as dist

def ddp_setup():
    """
    Uses torchrun env vars.
    Returns (is_ddp, rank, local_rank, world_size, device).
    """
    if "RANK" not in os.environ:
        return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return True, rank, local_rank, world_size, device

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
