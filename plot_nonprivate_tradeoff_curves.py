"""
plot_nonprivate_tradeoff.py

Plots Figure 2 (Non-Private Empirical Epsilon: Defense vs No Defense)
with new numbers.

Usage:
    python plot_nonprivate_tradeoff.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

datasets   = ["MNIST (CNN)", "Purchase100\n(MLP)", "CIFAR-10\n(CNN)", "CIFAR-10\n(WideResNet)", "CIFAR-10\n(AugMult4)"]
no_defense = [4.974814,      5.748285,              19.747136,          12.675705,                  8.308668            ]
defense    = [0.452629,      0.777528,               7.502931,           0.07851,                   5.304449            ]

# ---------------------------------------------------------------------------
# Colors — matching Figure 2 style: solid purple ND, hatched lighter purple D
# ---------------------------------------------------------------------------

ND_COLOR = "#7B68EE"   # medium slate blue
D_COLOR  = "#C3B1E1"   # light purple

x     = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.bar(x - width/2, no_defense, width, color=ND_COLOR, alpha=0.88,
       edgecolor="none", label="No Defense")
ax.bar(x + width/2, defense,    width, color=D_COLOR,  alpha=0.88,
       edgecolor=ND_COLOR, linewidth=0.8, hatch="///", label="Defense")

ax.set_title("Non-Private Empirical Epsilon: Defense vs No Defense", fontsize=11, pad=8)
ax.set_ylabel("Empirical Epsilon", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=9)
ax.set_ylim(bottom=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
ax.set_axisbelow(True)

ax.legend(fontsize=9, frameon=True, loc="upper right")

plt.tight_layout()
plt.show()