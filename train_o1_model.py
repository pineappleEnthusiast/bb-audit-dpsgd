import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.audit import compute_eps_lower_from_mia
from audit_model import test_model, train_model as train_model_dp
from utils.data import load_data

def train_model(model_name, base_dataset, optimized_canaries, optimized_labels, init_model, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_base, y_base = base_dataset

    # split canaries
    m = len(optimized_canaries)
    perm = torch.randperm(m)
    half = m // 2
    C_in, y_inC = optimized_canaries[perm[:half]], optimized_labels[perm[:half]]
    C_out, y_outC = optimized_canaries[perm[half:]], optimized_labels[perm[half:]]

    # merge with base dataset
    X_train = torch.cat([X_base, C_in])
    y_train = torch.cat([y_base, y_inC])

    print(f"Training DP model with {half} canaries inserted ({m} total canaries).")

    # train one model
    model = train_model_dp(
        model_name=model_name,
        X=X_train,
        y=y_train,
        X_target=None,
        y_target=None,
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        lr=args.lr,
        block_size=args.block_size,
        batch_size=args.batch_size,
        init_model=init_model,
        out_dim=args.out_dim,
        defense=args.defense,
        aug_mult=args.aug_mult,
        rank=0,
        world_size=1,
        gradient_space_audit=False
    )

    model.eval()
    with torch.no_grad():
        train_acc = test_model(model, X_train, y_train)
        print(f"Train set accuracy: {train_acc * 100:.3f}%")

        # compute test set accuracy
        X_test, y_test, _ = base_dataset if len(base_dataset) == 3 else (None, None, None)

        if X_test is None or y_test is None:
            X_test, y_test, _ = load_data(args.data_name, None, split="test")

        test_acc = test_model(model, X_test, y_test)
        print(f"Test set accuracy: {test_acc * 100:.3f}%")

    # eval canaries for privacy leakage
    model.eval()
    with torch.no_grad():
        loss_in = F.cross_entropy(model(C_in.to(device)), y_inC.to(device), reduction="none")
        loss_out = F.cross_entropy(model(C_out.to(device)), y_outC.to(device), reduction="none")

    scores_in = (-loss_in).cpu().numpy()
    scores_out = (-loss_out).cpu().numpy()
    mia_scores = np.concatenate([scores_in, scores_out])
    mia_labels = np.concatenate([np.ones(len(scores_in)), np.zeros(len(scores_out))])

    # compute emp eps
    emp_eps = compute_eps_lower_from_mia(mia_scores, mia_labels, args.alpha, args.delta, "GDP")

    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/mia_scores.npy", mia_scores)
    np.save(f"{args.out}/mia_labels.npy", mia_labels)
    np.save(f"{args.out}/emp_eps_loss.npy", [emp_eps])
    np.save(f"{args.out}/train_set_accs.npy", [train_acc])
    np.save(f"{args.out}/test_set_accs.npy", [test_acc])

    return emp_eps
