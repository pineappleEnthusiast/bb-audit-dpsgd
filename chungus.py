# import os
# import torch
# import torch.distributed as dist

# def main():
#     dist.init_process_group(backend="nccl")

#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     device = torch.device("cuda", local_rank)

#     print(f"[Rank {rank}/{world_size}] Hello from {os.uname().nodename} using {device}")

#     dist.destroy_process_group()

# if __name__ == "__main__":
#     main()

import os
import torch
import torch.distributed as dist

def main():
    # Initialize process group
    dist.init_process_group("nccl")

    # Get rank information
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Set the device - should use local_rank for CUDA device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"[Rank {rank}/{world_size}] Hello from node {os.uname().nodename}, local_rank={local_rank}")
    print(f"[Rank {rank}] Using device: {device}")
    print(f"[Rank {rank}] CUDA available: {torch.cuda.is_available()}")
    print(f"[Rank {rank}] CUDA device count: {torch.cuda.device_count()}")
    print(f"[Rank {rank}] Current CUDA device: {torch.cuda.current_device()}")
    print(f"[Rank {rank}] Device name: {torch.cuda.get_device_name(local_rank)}")
    
    # Allocate some memory on each GPU
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y  # Simple matrix multiplication to ensure GPU is working
    
    # Print memory stats
    print(f"[Rank {rank}] Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
    print(f"[Rank {rank}] Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e6:.2f} MB")
    
    # Each GPU prints a different message
    if rank == 0:
        print("GPU 0: Training model A")
    elif rank == 1:
        print("GPU 1: Training model B")
    else:
        print(f"GPU {rank}: Doing something else")
    
    # Sync all processes before destroying the process group
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
