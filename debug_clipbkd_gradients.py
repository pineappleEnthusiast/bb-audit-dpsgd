import torch
import torch.nn as nn
from models import Models
from models.wideresnet import WSConv2d

def xavier_init_model(model):
    """Initialize model using Xavier initialization"""
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    model.apply(init_weights)

def init_wideresnet(model):
    """Initialize model using Kaiming initialization (He init) for ReLU"""
    for m in model.modules():
        if isinstance(m, WSConv2d):
            m._initialize_weights()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
from utils.data import load_data
import argparse
import numpy as np
import copy

def get_gradient(model, x, y, criterion):
    """Compute gradient for a single sample x, y"""
    model.eval()
    model.zero_grad()
    # x needs batch dim if not present
    if x.ndim == 3:
        x = x.unsqueeze(0)
    
    # Forward pass
    output = model(x)
    loss = criterion(output, y.view(-1))
    loss.backward()
    
    # Flatten gradient
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).clone())
    
    if not grads:
        return None
        
    return torch.cat(grads)

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two 1D vectors"""
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return torch.dot(vec1, vec2) / (norm1 * norm2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--canary_pt', type=str, required=True, help='Path to canary .pt file')
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--model_name', type=str, default='cnn')
    parser.add_argument('--epochs', type=int, default=100) # Run for enough epochs to see defense triggers
    parser.add_argument('--lr', type=float, default=3.0) # High LR to make updates visible
    parser.add_argument('--defense_k', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    X, y, out_dim = load_data(args.data_name, n_df=5000, split='train') # Use small subset for speed
    
    # 2. Load Canary
    print(f"Loading canary from {args.canary_pt}...")
    canary_payload = torch.load(args.canary_pt, map_location='cpu')
    if isinstance(canary_payload, dict):
        canary_X = canary_payload['canary']
        canary_y = canary_payload.get('audit_label', canary_payload.get('target_label', torch.tensor([0])))
        if not isinstance(canary_y, torch.Tensor):
            canary_y = torch.tensor([canary_y])
    else:
        # Fallback if just tensor
        canary_X = canary_payload
        canary_y = torch.tensor([0]) # Placeholder or infer?
    
    canary_X = canary_X.to(device)
    canary_y = canary_y.to(device).long()

    # Append canary to dataset
    # We'll just manually use it for gradient computation, 
    # but to simulate "being caught", we need to *include* it in training?
    # No, let's just track the gradient OF the canary on the CURRENT model state.
    # The user asked: "cosine sim btwn gradient on the epoch it gets caught vs on the epoch we do gradient ascent"
    # Defense mechanism: 
    #   1. Compute grads for batch
    #   2. Compute scores
    #   3. Top-k get flagged.
    #   4. Gradient Ascent: negate grads of flagged samples.
    #   5. Update model.
    
    # So we need to run a training loop where the canary IS in the data.
    
    # Merge canary into X, y
    X_train = torch.cat([X, canary_X.cpu().unsqueeze(0) if canary_X.ndim==3 else canary_X.cpu()], dim=0)
    y_train = torch.cat([y, canary_y.view(-1).cpu()], dim=0)
    
    canary_idx = len(X_train) - 1
    print(f"Canary index: {canary_idx}")

    # 3. Initialize Model
    print("Initializing model...")
    # Fix seed
    torch.manual_seed(0)
    model = Models[args.model_name](X.shape).to(device)
    if args.model_name == 'cnn':
        xavier_init_model(model)
    else:
        init_wideresnet(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) # Standard SGD
    criterion = nn.CrossEntropyLoss()

    # History
    canary_gradients = {} # epoch -> gradient vector
    
    print("\n--- Starting Training Loop with Monitoring ---")
    
    for epoch in range(args.epochs):
        model.train()
        
        # 1. Compute Gradient of Canary *before* any updates this epoch (on current weights)
        grad_curr = get_gradient(copy.deepcopy(model), canary_X, canary_y, criterion)
        canary_gradients[epoch] = grad_curr
        
        grad_norm = torch.norm(grad_curr).item()
        
        # 2. Simulate Defense Check (Simplified)
        # In real training, we process batches. Here we'll just check "Global" rank for simplicity
        # or just pretend we are checking against the batch.
        # But wait, audit `parallel_audit_model` computes per-sample grads for *entire* dataset?
        # Or batch? It does `clip_and_accum_grads` which processes in blocks.
        # Let's assume we are looking at the canary's gradient relative to a batch.
        # For simplicity, we just train on a batch containing the canary and some others.
        
        # Create a batch with canary and random samples
        batch_indices = torch.randperm(len(X))[:100] # Random 100 normal samples
        batch_X = X[batch_indices].to(device)
        batch_y = y[batch_indices].to(device)
        
        # Add canary to batch
        batch_X = torch.cat([batch_X, canary_X.unsqueeze(0) if canary_X.ndim==3 else canary_X])
        batch_y = torch.cat([batch_y, canary_y.view(-1)])
        
        # Compute per-sample gradients for this batch
        per_sample_grads = []
        for i in range(len(batch_X)):
            g = get_gradient(copy.deepcopy(model), batch_X[i], batch_y[i:i+1], criterion)
            per_sample_grads.append(g)
            
        # Compute Grad Norm Scores
        norms = [torch.norm(g).item() for g in per_sample_grads]
        canary_norm = norms[-1]
        
        # Determine Rank
        sorted_norms = sorted(norms, reverse=True)
        rank = sorted_norms.index(canary_norm) + 1
        is_caught = rank <= args.defense_k
        
        print(f"Epoch {epoch}: Canary Norm={canary_norm:.4f}, Rank={rank}/{len(batch_X)}, Caught={is_caught}")
        
        if is_caught:
            print(f"   >>> Canary CAUGHT! Defense would apply gradient ascent.")
            # Note: Gradient ascent in the defense simply negates the gradient (-g).
            # The user asked for "cosine sim btwn gradient on the epoch it gets caught vs on the epoch we do gradient ascent"
            # This phrasing is tricky. Maybe they mean "gradient at Epoch T (caught)" vs "gradient at Epoch T (after ascent)".
            # But "after ascent" implies the model CHANGED? NO, ascent changes the gradient used for the update.
            # So the model update becomes: theta_{t+1} = theta_t - lr * (-g_canary + sum(other_grads))
            # effectively theta_{t+1} = theta_t + lr * g_canary - lr * sum(other_grads)
            pass

        # 3. Compute Cosine Similarity with previous epochs
        if epoch > 0:
            cos_sim_prev = compute_cosine_similarity(grad_curr, canary_gradients[epoch-1])
            print(f"   Cosine Sim with Epoch {epoch-1}: {cos_sim_prev:.4f}")
            
            # Check with first epoch
            cos_sim_start = compute_cosine_similarity(grad_curr, canary_gradients[0])
            print(f"   Cosine Sim with Epoch 0: {cos_sim_start:.4f}")

        # 4. Actual Model Update (Standard SGD for this simple script, 
        #    ignoring complex defense application logic for the UPDATE itself unless we want to simulate it)
        #    To properly "debug", we should apply the defense logic to the update.
        
        final_grads = torch.zeros_like(per_sample_grads[0])
        for i, g in enumerate(per_sample_grads):
            if i == len(per_sample_grads) - 1 and is_caught:
                # Canary caught -> Gradient Ascent -> Negate gradient
                final_grads += (-g) 
            else:
                final_grads += g
        
        # Average
        final_grads /= len(batch_X)
        
        # Update weights
        with torch.no_grad():
            idx = 0
            for param in model.parameters():
                if param.grad is not None:
                    num_params = param.numel()
                    param_grad = final_grads[idx:idx+num_params].view(param.shape)
                    param.data -= args.lr * param_grad
                    idx += num_params

if __name__ == "__main__":
    main()
