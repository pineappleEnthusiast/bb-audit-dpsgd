import torch
import torch.nn as nn
import torch.distributed as dist
import os
import argparse


# Simple toy model
class ToyModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def train_single_model(model, local_rank, model_id, epochs=5):
    """Train a single model on the given GPU."""
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    rank = dist.get_rank()
    print(f"[Rank {rank}, Local Rank {local_rank}] Training model {model_id}")
    
    for epoch in range(epochs):
        # Generate dummy data
        x = torch.randn(32, 10).to(device)
        y = torch.randn(32, 5).to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"[Rank {rank}] Model {model_id}, Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print(f"[Rank {rank}] Finished training model {model_id}")
    return model


def distribute_models(num_models, world_size):
    """Distribute models across available processes."""
    models_per_rank = [[] for _ in range(world_size)]
    for i in range(num_models):
        rank = i % world_size
        models_per_rank[rank].append(i)
    return models_per_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_models', type=int, default=10,
                       help='Number of models to train')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for this process')
    args = parser.parse_args()
    
    # Initialize process group using environment variables set by srun
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Get local rank (GPU on this node)
    local_rank = args.local_rank
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print(f"Training {args.num_models} models across {world_size} processes")
    
    # Distribute models across all ranks
    models_to_train = distribute_models(args.num_models, world_size)
    
    if rank == 0:
        print(f"Model distribution: {models_to_train}")
    
    # Each rank trains its assigned models
    for model_id in models_to_train[rank]:
        model = ToyModel()
        trained_model = train_single_model(model, local_rank, model_id)
        
        # Save the trained model
        save_dir = os.path.join(os.environ.get('SCRATCH', '.'), 'trained_models')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            trained_model.state_dict(),
            os.path.join(save_dir, f'model_{model_id}_rank_{rank}.pth')
        )
    
    # Synchronize all processes before cleanup
    dist.barrier()
    
    if rank == 0:
        print("All models trained successfully!")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    main()