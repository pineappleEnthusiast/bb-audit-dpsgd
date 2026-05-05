"""
Print comparison table for the three HAMP mini-experiment conditions.

Usage:
    python3 compare_hamp_results.py --out_root /tmp/hamp_mini
"""
import argparse
import os
import numpy as np


def load(out_root, defense):
    d = os.path.join(out_root, defense)
    emp_eps = float(np.load(os.path.join(d, 'emp_eps.npy')))
    bv_in   = np.load(os.path.join(d, 'binary_vectors_in.npy'))   # (N, 18)
    bv_out  = np.load(os.path.join(d, 'binary_vectors_out.npy'))  # (N, 18)
    return emp_eps, bv_in, bv_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root', default='/tmp/hamp_mini')
    args = parser.parse_args()

    conditions = ['none', 'hamp', 'filter']
    rows = []
    for defense in conditions:
        path = os.path.join(args.out_root, defense)
        if not os.path.exists(os.path.join(path, 'emp_eps.npy')):
            rows.append((defense, None, None, None, None))
            continue
        emp_eps, bv_in, bv_out = load(args.out_root, defense)
        mean_in  = bv_in.mean()
        mean_out = bv_out.mean()
        gap      = mean_in - mean_out
        rows.append((defense, emp_eps, mean_in, mean_out, gap))

    header = f"{'defense':<10} {'emp_eps':>9} {'mean_in':>9} {'mean_out':>10} {'gap':>8}"
    print(header)
    print('-' * len(header))
    for defense, emp_eps, mean_in, mean_out, gap in rows:
        if emp_eps is None:
            print(f"{defense:<10} {'NOT RUN':>9}")
        else:
            print(f"{defense:<10} {emp_eps:>9.4f} {mean_in:>9.3f} {mean_out:>10.3f} {gap:>8.3f}")

    print()

    # Interpret results
    data = {r[0]: r for r in rows if r[1] is not None}
    if 'none' in data and 'hamp' in data:
        _, eps_none, _, _, gap_none = data['none']
        _, eps_hamp, _, _, gap_hamp = data['hamp']
        # HAMP fails if it still has positive gap (attack still has signal)
        if gap_hamp > 0.01:
            print(f"CHECK 1 PASS  HAMP does NOT block the label-only attack "
                  f"(gap={gap_hamp:.3f} > 0, eps={eps_hamp:.4f})")
        else:
            print(f"CHECK 1 FAIL  HAMP appears to block the attack "
                  f"(gap={gap_hamp:.3f} ≈ 0) — need more reps or epochs")

    if 'none' in data and 'filter' in data:
        _, eps_none, _, _, gap_none = data['none']
        _, eps_filter, _, _, gap_filter = data['filter']
        if gap_filter < gap_none - 0.01:
            print(f"CHECK 2 PASS  Filter reduces the attack signal "
                  f"(gap: {gap_none:.3f} → {gap_filter:.3f}, "
                  f"eps: {eps_none:.4f} → {eps_filter:.4f})")
        else:
            print(f"CHECK 2 FAIL  Filter does not reduce the attack signal "
                  f"(gap: {gap_none:.3f} → {gap_filter:.3f}) — "
                  f"try larger defense_k or more epochs")


if __name__ == '__main__':
    main()
