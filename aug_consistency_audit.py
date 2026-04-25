"""
Augmentation-consistency audit for DP-SGD.

Identical to parallel_audit_model.py in every respect EXCEPT the scoring step:
instead of using negative cross-entropy loss as the MIA score, we score each
shadow model by the *augmentation-consistency* of its prediction on the canary:

    score = fraction of k augmented views on which the model predicts the
            modal (most-common) label.

Intuition: a model that memorized the canary will still predict consistently
across augmentations, whereas a model that never saw it will be uncertain.

Pipeline:
  1. Build the neighbouring dataset and canary exactly as parallel_audit_model.py.
  2. Train shadow models (in / out worlds) using the existing train_model().
  3. Score each model via aug_consistency_score() defined below.
  4. Estimate εlb via compute_eps_lower_from_mia() (same as everywhere else).
  5. Print and save results in the same format so they are directly comparable
     to Table 1 / Table 2 in the paper.

Entry-point: this script is designed for single-GPU interactive runs (idev).
All training hyper-parameters are shared with parallel_audit_model.py through
utils/args.py; three new flags are added:
  --aug_k       number of augmented views per scoring query  (default 50)
  --aug_seed    random seed for augmentation during scoring  (default 42)
"""

import os
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from utils.args import build_parser
from utils.data import load_data
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t
from utils.training import (
    AugmentationFunction,
    xavier_init_model,
    init_wideresnet,
    test_model,
)
from utils.checkpoint import save_checkpoint, init_run_state
from models import Models
from parallel_audit_model import train_model   # reuse existing training loop


# ---------------------------------------------------------------------------
# New scoring logic — everything else is imported from existing modules
# ---------------------------------------------------------------------------

def aug_consistency_score(
    model: nn.Module,
    x: torch.Tensor,
    canary_label: int,
    aug_fn: AugmentationFunction,
    k: int,
    device: torch.device,
) -> float:
    """
    Compute augmentation-consistency score for a single sample x.

    Query the model on k independently-augmented copies of x and return the
    fraction of views on which the model predicts the canary's assigned label.

    A model that memorized the canary predicts its label consistently → high
    score. A model that never saw it predicts a natural class → low score.

    Args:
        model:        trained shadow model (eval mode expected).
        x:            single image tensor, shape (1, C, H, W) or (C, H, W).
        canary_label: the label assigned to the canary during training.
        aug_fn:       AugmentationFunction instance (random crop + horizontal flip).
        k:            number of augmented views.
        device:       device to run inference on.

    Returns:
        Scalar float in [0.0, 1.0]. Higher = more consistent with canary label.
    """
    model.eval()

    # Ensure x has a batch dimension
    if x.ndim == 3:
        x = x.unsqueeze(0)   # (1, C, H, W)

    canary_label_count = 0
    with torch.no_grad():
        for _ in range(k):
            x_aug = aug_fn(x).to(device)
            logits = model(x_aug)            # (1, num_classes)
            pred = logits.argmax(dim=1).item()
            if pred == canary_label:
                canary_label_count += 1

    return canary_label_count / k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Parse arguments (shared base + aug-specific flags)
    # ------------------------------------------------------------------
    parser = build_parser()
    parser.add_argument(
        '--aug_k', type=int, default=50,
        help='Number of augmented views per canary query during scoring (default: 50)',
    )
    parser.add_argument(
        '--aug_seed', type=int, default=42,
        help='RNG seed for augmentation during the scoring phase (default: 42)',
    )
    args = parser.parse_args()

    if args.epsilon == -1:
        args.epsilon = None
    if args.max_grad_norm == -1:
        args.max_grad_norm = None

    # Only single-GPU (idev) mode is supported here
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    rank = 0

    print(f"Device: {device}")
    print(f"Augmentation consistency audit — k={args.aug_k}, aug_seed={args.aug_seed}")

    out_folder = os.path.join(
        args.out,
        f'{args.data_name}_{args.model_name}_eps{args.epsilon}_aug_consistency'
    )
    os.makedirs(out_folder, exist_ok=True)
    print(f"Output directory: {out_folder}")

    print("Loading data...")
    if args.n_df == 1:
        X_out, y_out, out_dim = load_data(args.data_name, 1)
    else:
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)

    X_test, y_test, _ = load_data(args.data_name, None, split='test')

    # DEBUG ONLY — remove this line to restore full dataset
    X_out, y_out = X_out[:500], y_out[:500]

    init_model = None
    if args.fixed_init is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim)
        if args.fixed_init == '':
            if args.model_name == 'cnn':
                xavier_init_model(init_model)
            elif args.model_name == 'wideresnet':
                init_wideresnet(init_model)
            else:
                xavier_init_model(init_model)
        else:
            init_model.load_state_dict(torch.load(args.fixed_init))
            X_out, y_out = X_out[len(X_out) // 2:], y_out[len(y_out) // 2:]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    print("Crafting canary...")
    if args.target_type == 'blank':
        target_X = torch.zeros_like(X_out[[0]])
        target_y = torch.tensor([9], dtype=torch.long)
    elif args.target_type == 'sanity_check':
        target_X = X_out[-1].unsqueeze(0)
        target_y = y_out[-1].unsqueeze(0)
    elif args.target_type == 'mislabeled':
        class_0_indices = (y_out == 0).nonzero(as_tuple=True)[0]
        if len(class_0_indices) == 0:
            raise ValueError("No class 0 samples found in dataset for mislabeled canary")
        target_idx = class_0_indices[0].item()
        target_X = X_out[target_idx].unsqueeze(0)
        target_y = torch.tensor([args.mislabeled_target_class], dtype=torch.long)
    else:
        raise NotImplementedError(
            f"target_type='{args.target_type}' is not implemented in this script. "
            "Use 'blank', 'sanity_check', or 'mislabeled'."
        )
    print(f"Canary: shape={tuple(target_X.shape)}, label={target_y.tolist()}")

    X_in  = torch.vstack((X_out[:-1], target_X))
    y_in  = torch.cat((y_out[:-1], target_y))

    if len(X_out.shape) > 2:
        aug_fn_score = AugmentationFunction(X_out.shape[2], X_out.shape[1])
    else:
        raise RuntimeError("Augmentation-consistency audit requires image data.")

    outputs, losses, all_losses, train_set_accs, test_set_accs = init_run_state(
        out_folder, args.fit_world_only, rank
    )

    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']

    n_reps_per_world = args.n_reps // 2
    for world in worlds:
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)

        for rep in range(n_reps_per_world):
            print(f"\n[{world.upper()}] Rep {rep + 1}/{n_reps_per_world}")

            generator    = torch.Generator().manual_seed(args.seed + rep * 2)
            dl_generator = torch.Generator().manual_seed(args.seed + rep * 2 + 1)

            model = train_model(
                args.model_name,
                curr_X,
                curr_y,
                target_X,
                target_y,
                args.epsilon,
                args.delta,
                args.max_grad_norm,
                args.n_epochs,
                args.lr,
                args.block_size,
                args.batch_size,
                init_model=init_model,
                out_dim=out_dim,
                defense=args.defense,
                defense_k=int(args.defense_k),
                defense_filter_every=int(args.defense_filter_every),
                aug_mult=args.aug_mult,
                gradient_space_audit=False,
                crafted_gradient=None,
                device=device,
                generator=generator,
                dl_generator=dl_generator,
                rank=rank,
                defense_score_norm=args.defense_score_norm,
                defense_score_fn=args.defense_score_fn,
                grad_norm_percentile_k=args.grad_norm_percentile_k,
                grad_dir_volatility_k=args.grad_dir_volatility_k,
                grad_dir_proj_dim=args.grad_dir_proj_dim,
                grad_dir_proj_seed=args.grad_dir_proj_seed,
                dir_unique_k=args.dir_unique_k,
                rand_proj_var_m=args.rand_proj_var_m,
                rand_proj_var_seed=args.rand_proj_var_seed,
                maxmin_proj_k=args.maxmin_proj_k,
                maxmin_proj_seed=args.maxmin_proj_seed,
                grad_rank_mode=args.grad_rank_mode,
                grad_rank_eps=args.grad_rank_eps,
                grad_accel_proj_dim=args.grad_accel_proj_dim,
                grad_accel_proj_seed=args.grad_accel_proj_seed,
                grad_jerk_proj_dim=args.grad_jerk_proj_dim,
                grad_jerk_proj_seed=args.grad_jerk_proj_seed,
                alignment_proj_k=args.alignment_proj_k,
                alignment_proj_seed=args.alignment_proj_seed,
                grad_scatter_k=args.grad_scatter_k,
                defense_apply_ascent=args.defense_apply_ascent,
                sampling=args.sampling,
            )

            # ----------------------------------------------------------
            # Score: augmentation consistency (replaces -loss)
            # Fix the augmentation seed so scoring is reproducible, but
            # different per rep to avoid seed correlation with training.
            # ----------------------------------------------------------
            torch.manual_seed(args.aug_seed + rep)
            np.random.seed(args.aug_seed + rep)

            model.eval()
            score = aug_consistency_score(
                model=model,
                x=target_X,
                canary_label=int(target_y.item()),
                aug_fn=aug_fn_score,
                k=args.aug_k,
                device=device,
            )
            print(f"  Aug-consistency score: {score:.4f}")

            # Store outputs (logits) unchanged for compatibility with
            # downstream loading scripts, even though they are not used
            # for the audit threshold.
            with torch.no_grad():
                target_X_dev = target_X.to(device)
                output = model(target_X_dev)
            outputs[world].append(output[0].cpu().numpy())

            # The KEY change: use consistency score instead of -loss
            losses[world].append(score)

            save_checkpoint(
                out_folder, outputs, losses, all_losses,
                train_set_accs, test_set_accs, args.fit_world_only, rank,
            )

            # Train / test accuracy from first 5 in-world reps
            if rep < 5 and world == 'in':
                train_set_accs.append(test_model(model, X_in, y_in))
                test_set_accs.append(test_model(model, X_test, y_test))
                print(f"  Train acc: {train_set_accs[-1]*100:.2f}%  "
                      f"Test acc:  {test_set_accs[-1]*100:.2f}%")
                save_checkpoint(
                    out_folder, outputs, losses, all_losses,
                    train_set_accs, test_set_accs, args.fit_world_only, rank,
                )

        outputs[world] = np.array(outputs[world])

    # ------------------------------------------------------------------
    # Save combined .npy files (same layout as parallel_audit_model.py)
    # ------------------------------------------------------------------
    if not args.fit_world_only:
        np.save(f'{out_folder}/outputs_in.npy',   outputs['in'])
        np.save(f'{out_folder}/outputs_out.npy',  outputs['out'])
        np.save(f'{out_folder}/losses_in.npy',    losses['in'])
        np.save(f'{out_folder}/losses_out.npy',   losses['out'])
        if train_set_accs:
            np.save(f'{out_folder}/train_set_accs.npy', train_set_accs)
        if test_set_accs:
            np.save(f'{out_folder}/test_set_accs.npy',  test_set_accs)

    # ------------------------------------------------------------------
    # εlb estimation — identical procedure to parallel_audit_model.py
    # ------------------------------------------------------------------
    if not args.fit_world_only:
        losses_in  = np.array(losses['in'])
        losses_out = np.array(losses['out'])
        n = len(losses_in)

        if args.holdout_audit:
            np.random.seed(args.seed)
            idx = np.random.permutation(n)
            t_idx, h_idx = idx[:n // 2], idx[n // 2:]

            t_scores = np.concatenate([losses_in[t_idx],  losses_out[t_idx]])
            t_labels = np.concatenate([np.ones(len(t_idx)), np.zeros(len(t_idx))])

            max_t, _ = compute_eps_lower_from_mia(
                t_scores, t_labels, args.alpha, args.delta, 'GDP', n_procs=1
            )

            h_scores = np.concatenate([losses_in[h_idx],  losses_out[h_idx]])
            h_labels = np.concatenate([np.ones(len(h_idx)), np.zeros(len(h_idx))])

            emp_eps = compute_eps_lower_from_mia_given_t(
                h_scores, h_labels, args.alpha, args.delta, max_t, 'GDP'
            )
        else:
            mia_scores = np.concatenate([losses_in,  losses_out])
            mia_labels = np.concatenate([np.ones(n), np.zeros(n)])

            max_t, emp_eps = compute_eps_lower_from_mia(
                mia_scores, mia_labels, args.alpha, args.delta, 'GDP', n_procs=1
            )
            # expose mia arrays for downstream use
            np.save(f'{out_folder}/mia_scores.npy', mia_scores)
            np.save(f'{out_folder}/mia_labels.npy', mia_labels)

        np.save(f'{out_folder}/emp_eps_loss.npy', [emp_eps])  # same key as main script

        # ------------------------------------------------------------------
        # Summary — same fields reported in Table 1 / Table 2
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("AUGMENTATION-CONSISTENCY AUDIT RESULTS")
        print("=" * 60)
        print(f"  Dataset          : {args.data_name}")
        print(f"  Model            : {args.model_name}")
        print(f"  Target type      : {args.target_type}")
        print(f"  Defense          : {'ON (k=' + str(args.defense_k) + ')' if args.defense else 'OFF'}")
        print(f"  ε (theoretical)  : {args.epsilon}")
        print(f"  δ                : {args.delta}")
        print(f"  Reps (per world) : {n_reps_per_world}")
        print(f"  Aug views (k)    : {args.aug_k}")
        print(f"  Score type       : augmentation-consistency")
        if train_set_accs:
            print(f"  Train accuracy   : {np.mean(train_set_accs)*100:.2f}%")
        if test_set_accs:
            print(f"  Test accuracy    : {np.mean(test_set_accs)*100:.2f}%")
        print(f"  εlb (empirical)  : {emp_eps:.4f}")
        print("=" * 60)
        print(f"\nResults saved to: {out_folder}/")


if __name__ == '__main__':
    main()
