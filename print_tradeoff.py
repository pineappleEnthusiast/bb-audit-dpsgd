import sys
import numpy as np
sys.path.insert(0, '.')
from utils.audit import compute_eps_lower_single, AttackResults

d = sys.argv[1] if len(sys.argv) > 1 else 'tradeoff_curves/mnist_no_defense_eps10/mnist_cnn_eps10.0'
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
delta = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5

scores_in  = np.load(f'{d}/losses_in.npy') if __import__('os').path.exists(f'{d}/losses_in.npy') else np.load(f'{d}/scores_in.npy')
scores_out = np.load(f'{d}/losses_out.npy') if __import__('os').path.exists(f'{d}/losses_out.npy') else np.load(f'{d}/scores_out.npy')

print(f"scores_in:  n={len(scores_in)}  mean={scores_in.mean():.4f}  std={scores_in.std():.4f}")
print(f"scores_out: n={len(scores_out)}  mean={scores_out.mean():.4f}  std={scores_out.std():.4f}")

if scores_in.mean() < scores_out.mean():
    print("(flipping sign so in > out)")
    scores_in  = -scores_in
    scores_out = -scores_out

scores = np.concatenate([scores_in, scores_out]).astype(np.float32)
labels = np.concatenate([np.ones(len(scores_in)), np.zeros(len(scores_out))]).astype(np.int64)

threshs = np.percentile(scores, np.arange(0, 101, 10))
n_in, n_out = len(scores_in), len(scores_out)

print(f"\n{'t':>10}  {'TP':>6}  {'FP':>6}  {'TN':>6}  {'FN':>6}  {'TPR':>6}  {'FPR':>6}  {'eps_lb':>8}")
print('-' * 70)
for t in threshs:
    tp = int(np.sum(scores[labels == 1] >= t))
    fp = int(np.sum(scores[labels == 0] >= t))
    fn = n_in - tp
    tn = n_out - fp
    r  = AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)
    eps = compute_eps_lower_single(r, alpha, delta, 'GDP')
    print(f"{t:10.4f}  {tp:6d}  {fp:6d}  {tn:6d}  {fn:6d}  {tp/n_in:6.3f}  {fp/n_out:6.3f}  {eps:8.4f}")
