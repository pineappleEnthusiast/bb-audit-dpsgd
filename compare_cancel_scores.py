import numpy as np
import os

for tag in ['no_defense', 'defense']:
    d = 'grad_cancel_test/audit_' + tag
    scores_in  = np.load(os.path.join(d, 'scores_in.npy'))
    scores_out = np.load(os.path.join(d, 'scores_out.npy'))
    mean_in  = scores_in.mean()
    mean_out = scores_out.mean()
    gap = mean_in - mean_out
    print(f'{tag:12s}: in={mean_in:.4f}  out={mean_out:.4f}  gap={gap:.4f}')
