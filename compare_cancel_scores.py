import numpy as np
import os
import sys
import glob

output_dir = sys.argv[1] if len(sys.argv) > 1 else 'grad_cancel_test'

for tag in ['no_defense', 'defense']:
    base = os.path.join(output_dir, 'audit_' + tag)
    # find the subdirectory (e.g. mnist_cnn_epsNone)
    subdirs = glob.glob(os.path.join(base, '*'))
    if not subdirs:
        print(f'{tag:12s}: no results found in {base}')
        continue
    d = subdirs[0]
    scores_in  = np.load(os.path.join(d, 'scores_in.npy'))
    scores_out = np.load(os.path.join(d, 'scores_out.npy'))
    mean_in  = scores_in.mean()
    mean_out = scores_out.mean()
    gap = mean_in - mean_out
    print(f'{tag:12s}: in={mean_in:.4f}  out={mean_out:.4f}  gap={gap:.4f}')
