"""
Fairness experiment on Colored MNIST with blank canary.

Runs n_reps independent training runs (canary always included) and reports
averaged per-subgroup utility and defense removal distribution.

Subgroups:
  0 = class 0 (even digits), red   (majority for class 0)
  1 = class 0 (even digits), blue  (minority for class 0)
  2 = class 1 (odd digits),  red   (minority for class 1)
  3 = class 1 (odd digits),  blue  (majority for class 1)
"""
import os
import copy
import numpy as np
import torch
import torch.distributed as dist

from models import Models
from utils.data import load_colored_mnist
from utils.training import xavier_init_model
from utils.args import build_parser
from parallel_audit_model import train_model, distribute_reps

SUBGROUP_NAMES = [
    'class0_red  (majority)',
    'class0_blue (minority)',
    'class1_red  (minority)',
    'class1_blue (majority)',
]


def eval_per_subgroup(model, X_test, y_test, subgroups, device, batch_size=256):
    """Return {sg_id: accuracy} for each of the 4 subgroups."""
    model.eval()
    accs = {}
    with torch.no_grad():
        for sg in range(4):
            mask = subgroups == sg
            if not mask.any():
                accs[sg] = float('nan')
                continue
            X_sg = X_test[mask]
            y_sg = y_test[mask]
            correct = 0
            for i in range(0, len(X_sg), batch_size):
                xb = X_sg[i:i + batch_size].to(device)
                yb = y_sg[i:i + batch_size].to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
            accs[sg] = correct / int(mask.sum())
    model.train()
    return accs


def removal_counts(drop_mask, sg_labels):
    """
    Count removed samples (drop_mask==2) per subgroup.
    sg_labels: LongTensor aligned with drop_mask (canary slot already excluded).
    Returns {sg: (n_removed, n_total)}.
    """
    counts = {}
    sg_np = sg_labels.numpy()
    for sg in range(4):
        mask = sg_np == sg
        n_total = int(mask.sum())
        n_removed = int((drop_mask[mask] == 2).sum())
        counts[sg] = (n_removed, n_total)
    return counts


def main():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        print(f'[Rank {rank}] Using device: {torch.cuda.get_device_name(local_rank)}')
    else:
        local_rank = rank = 0
        world_size = 1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Single GPU mode: {device}')

    parser = build_parser()
    parser.add_argument('--majority_pct', type=float, default=0.995,
                        help='Fraction of each class assigned the majority color (e.g. 0.995)')
    args = parser.parse_args()
    if args.epsilon == -1:
        args.epsilon = None
    if args.max_grad_norm == -1:
        args.max_grad_norm = None

    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------ data
    if rank == 0:
        print(f'Loading Colored MNIST (binary: digits 0 and 1, majority_pct={args.majority_pct})...')
    X_train, y_train, sg_train, out_dim = load_colored_mnist(split='train', seed=args.seed, majority_pct=args.majority_pct)
    X_test,  y_test,  sg_test,  _       = load_colored_mnist(split='test',  seed=args.seed, majority_pct=args.majority_pct)

    if rank == 0:
        print(f'Train: {len(y_train)}  |  Test: {len(y_test)}')
        for sg in range(4):
            n_tr = int((sg_train == sg).sum())
            n_te = int((sg_test  == sg).sum())
            print(f'  {SUBGROUP_NAMES[sg]:26s}: train={n_tr:5d}, test={n_te:4d}')

    # ------------------------------------------------------------------ canary
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.target_type == 'minority':
        # Use the first class0_blue (sg==1) sample as the canary; remove from training.
        minority_idx = int((sg_train == 1).nonzero(as_tuple=True)[0][0])
        target_X = X_train[[minority_idx]]
        target_y = y_train[[minority_idx]]
        keep = torch.ones(len(y_train), dtype=torch.bool)
        keep[minority_idx] = False
        X_train  = X_train[keep]
        y_train  = y_train[keep]
        sg_train = sg_train[keep]
        if rank == 0:
            print(f'Minority canary: sg=1 (class0_blue), index {minority_idx}, label {target_y.item()}')
    else:
        # Blank canary: all-zeros, label 0.
        target_X = torch.zeros(1, *X_train.shape[1:])
        target_y = torch.zeros(1, dtype=torch.long)

    X_with_canary = torch.vstack((X_train, target_X))
    y_with_canary = torch.cat((y_train, target_y))
    # sg_train aligns with X_train; canary sits at [-1] and is excluded from subgroup stats

    # ------------------------------------------------------------------ fixed init
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    init_model = Models[args.model_name](X_train.shape, out_dim=out_dim)
    xavier_init_model(init_model)

    # ------------------------------------------------------------------ training
    reps_per_gpu = distribute_reps(args.n_reps, world_size)
    my_reps = reps_per_gpu[rank]

    all_accs     = []   # list of {sg: acc} dicts, one per rep
    all_removals = []   # list of {sg: (n_removed, n_total)} dicts, one per rep (defense only)

    for rep in my_reps:
        print(f'[Rank {rank}] rep={rep}')
        generator    = torch.Generator().manual_seed(args.seed + rep * 2)
        dl_generator = torch.Generator().manual_seed(args.seed + rep * 2 + 1)

        model, drop_mask, canary_dropped_epoch = train_model(
            args.model_name, X_with_canary, y_with_canary,
            target_X, target_y,
            args.epsilon, args.delta, args.max_grad_norm,
            args.n_epochs, args.lr, args.block_size, args.batch_size,
            init_model=copy.deepcopy(init_model),
            out_dim=out_dim,
            aug_mult=args.aug_mult,
            defense=args.defense,
            defense_k=int(args.defense_k),
            defense_apply_ascent=False,
            device=str(device),
            generator=generator,
            dl_generator=dl_generator,
            rank=local_rank,
            world_size=1,
            defense_score_fn=args.defense_score_fn,
            defense_score_norm=args.defense_score_norm,
            sampling=args.sampling,
            return_defense_state=True,
        )

        # Per-subgroup test accuracy
        accs = eval_per_subgroup(model, X_test, y_test, sg_test, device)
        all_accs.append(accs)
        print(f'  Per-subgroup test acc: ' +
              '  '.join(f'{SUBGROUP_NAMES[sg].split()[0]}={accs[sg]:.3f}' for sg in range(4)))

        # Per-subgroup removal distribution (real samples only — exclude canary at [-1])
        if args.defense:
            real_dm = drop_mask[:-1]    # drop_mask for X_train (no canary)
            rc = removal_counts(real_dm, sg_train)
            all_removals.append(rc)
            if canary_dropped_epoch is not None:
                print(f'  Canary dropped at epoch {canary_dropped_epoch}')
            else:
                print(f'  Canary NOT dropped')

    # ------------------------------------------------------------------ summary
    if rank == 0:
        print('\n' + '=' * 60)
        print(f'FAIRNESS SUMMARY  defense={args.defense}  epsilon={args.epsilon}  n_reps={args.n_reps}  majority_pct={args.majority_pct}')
        print('=' * 60)

        print('\nPer-subgroup test accuracy (mean ± std over reps):')
        for sg in range(4):
            vals = [a[sg] for a in all_accs if not np.isnan(a.get(sg, float('nan')))]
            if vals:
                print(f'  {SUBGROUP_NAMES[sg]:26s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

        if args.defense and all_removals:
            print('\nDefense removal distribution (mean ± std over reps):')
            for sg in range(4):
                pairs = [r[sg] for r in all_removals]
                ns = [p[0] for p in pairs]
                n_total = pairs[0][1]
                pct_mean = 100 * np.mean(ns) / max(n_total, 1)
                pct_std  = 100 * np.std(ns)  / max(n_total, 1)
                print(f'  {SUBGROUP_NAMES[sg]:26s}: {np.mean(ns):.1f} ± {np.std(ns):.1f} '
                      f'/ {n_total} ({pct_mean:.1f}% ± {pct_std:.1f}% removed)')

        print('=' * 60)

    np.save(f'{args.out}/fairness_results.npy', {
        'accs': all_accs,
        'removals': all_removals,
        'subgroup_names': SUBGROUP_NAMES,
    }, allow_pickle=True)
    if rank == 0:
        print(f'Saved to {args.out}/fairness_results.npy')


if __name__ == '__main__':
    main()
