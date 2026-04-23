"""
Print per-sample unclipped linf gradient norm distribution for a dataset/model combo.
Used to calibrate alpha for the gradient cancelling attack.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from models import Models
from utils.data import load_data


def xavier_init_model(model):
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    model.apply(init_weights)


def per_sample_linf_grad_norms(model, X, y, device, batch_size=64):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    norms = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size].to(device)
        yb = y[i:i+batch_size].to(device)
        for j in range(len(xb)):
            model.zero_grad()
            loss = criterion(model(xb[j:j+1]), yb[j:j+1])
            loss.backward()
            flat = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            norms.append(flat.abs().max().item())
        if len(norms) % 200 == 0:
            print(f'  computed {len(norms)} samples...', flush=True)
    return np.array(norms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--model_name', type=str, default='cnn')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of training samples to compute norms over')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'Loading {args.data_name}...')
    X, y, out_dim = load_data(args.data_name, n_df=args.n_samples)
    print(f'Loaded {len(X)} samples.')

    model = Models[args.model_name](X.shape, out_dim=out_dim).to(device)
    xavier_init_model(model)

    print(f'Computing per-sample linf gradient norms on {len(X)} samples...')
    norms = per_sample_linf_grad_norms(model, X, y, device)

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f'\nPer-sample unclipped linf gradient norm distribution ({args.data_name}, {args.model_name}):')
    print(f'  min   : {norms.min():.4f}')
    for p in pcts:
        print(f'  p{p:<3d}  : {np.percentile(norms, p):.4f}')
    print(f'  max   : {norms.max():.4f}')
    print(f'  mean  : {norms.mean():.4f}')
    print(f'  std   : {norms.std():.4f}')
    print(f'\nSuggested alpha (below p10): {np.percentile(norms, 10):.4f}')
    print(f'Suggested beta  (above p90): {np.percentile(norms, 90):.4f}')


if __name__ == '__main__':
    main()
