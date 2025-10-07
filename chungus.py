import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"[Rank {rank}/{world_size}] Hello from {os.uname().nodename} using {device}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
