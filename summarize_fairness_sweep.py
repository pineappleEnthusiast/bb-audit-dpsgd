"""
Aggregate and display results from the fairness sweep.
Run after (or during) run_fairness_sweep.sh:
    python summarize_fairness_sweep.py
"""
import os
import numpy as np

MAJORITY_PCTS = [0.995, 0.95, 0.90, 0.85, 0.80, 0.75]
PARENT_OUT = 'fairness_audits'
SG_SHORT = ['c0_red(maj)', 'c0_blue(min)', 'c1_red(min)', 'c1_blue(maj)']
N_SG = 4


def load(majority_pct, defense):
    tag = 'defense' if defense else 'no_defense'
    # Try both Python's str() representation (e.g. "0.9") and the 2-decimal
    # form bash uses for round values (e.g. "0.90").
    for pct_str in [str(majority_pct), f'{majority_pct:.2f}']:
        path = os.path.join(PARENT_OUT, f'colored_mnist_maj{pct_str}_{tag}', 'fairness_results.npy')
        if os.path.exists(path):
            return np.load(path, allow_pickle=True).item()
    return None


def acc_stats(data, sg):
    vals = [a[sg] for a in data['accs'] if not np.isnan(a.get(sg, float('nan')))]
    if not vals:
        return None, None
    return np.mean(vals), np.std(vals)


def removal_stats(data, sg):
    if not data.get('removals'):
        return None, None
    pairs = [r[sg] for r in data['removals']]
    ns = [p[0] for p in pairs]
    n_total = pairs[0][1]
    if n_total == 0:
        return None, None
    pct_mean = 100 * np.mean(ns) / n_total
    pct_std  = 100 * np.std(ns)  / n_total
    return pct_mean, pct_std


def fmt(mean, std, pct=False):
    if mean is None:
        return '  --   '
    unit = '%' if pct else ' '
    return f'{mean:5.1f}±{std:4.1f}{unit}'


# ------------------------------------------------------------------ accuracy
print('\n' + '=' * 100)
print('PER-SUBGROUP TEST ACCURACY (mean ± std over reps)')
print('=' * 100)

col_w = 16
header1 = f'{"majority_pct":>12} |'
header2 = f'{"":>12} |'
for sg in range(N_SG):
    header1 += f' {SG_SHORT[sg]:^{col_w*2+3}}'
    header2 += f' {"no_def":^{col_w}} {"def":^{col_w}}'
print(header1)
print(header2)
print('-' * len(header2))

for pct in MAJORITY_PCTS:
    d_nd = load(pct, defense=False)
    d_d  = load(pct, defense=True)
    row = f'{pct:>12.3f} |'
    for sg in range(N_SG):
        nd_mean, nd_std = acc_stats(d_nd, sg) if d_nd else (None, None)
        d_mean,  d_std  = acc_stats(d_d,  sg) if d_d  else (None, None)
        row += f' {fmt(nd_mean, nd_std):>{col_w}} {fmt(d_mean, d_std):>{col_w}}'
    print(row)

# ------------------------------------------------------------------ removal
print('\n' + '=' * 100)
print('DEFENSE REMOVAL RATE % (mean ± std over reps)  [defense condition only]')
print('=' * 100)

header1 = f'{"majority_pct":>12} |'
for sg in range(N_SG):
    header1 += f' {SG_SHORT[sg]:^{col_w}}'
print(header1)
print('-' * len(header1))

for pct in MAJORITY_PCTS:
    d_d = load(pct, defense=True)
    row = f'{pct:>12.3f} |'
    for sg in range(N_SG):
        mean, std = removal_stats(d_d, sg) if d_d else (None, None)
        row += f' {fmt(mean, std, pct=True):>{col_w}}'
    print(row)

print()
