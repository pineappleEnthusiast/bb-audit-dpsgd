import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

epsilons = [2, 4, 6, 8, 10]

no_defense = {
    "MNIST":            [0.125165, 1.306895, 2.466728, 3.287312, 3.708773],
    "Purchase100":      [0.756123, 1.851655, 1.357458, 2.284626, 2.382529],
    "CIFAR-10 (CNN)":   [0.336209, 0.451714, 1.374738, 1.818479, 1.528651],
    "CIFAR-10 (WRN-16)":[0,        0.70017,  1.178806, 1.842942, 1.798339],
    "CIFAR-10 (CNN + AugMult4)": [0, 0.025169, 0.305612, 1.851655, 1.702905],
}

defense = {
    "MNIST":            [0.270209, 0,        0,        0,        0       ],
    "Purchase100":      [0.156302, 0.371062, 1.455932, 0.775874, 0.305612],
    "CIFAR-10 (CNN)":   [0,        0,        0.828466, 0.996661, 0.890761],
    "CIFAR-10 (WRN-16)":[0.072228, 0,        0,        0,        0.630157],
    "CIFAR-10 (CNN + AugMult4)": [0, 0, 0, 1.561189, 0.570582],
}

# replace all 0's in defense and no defense with 0.05
for dataset in no_defense:
    no_defense[dataset] = [0.03 if x == 0 else x for x in no_defense[dataset]]
for dataset in defense:
    defense[dataset] = [0.03 if x == 0 else x for x in defense[dataset]]

# ---------------------------------------------------------------------------
# Colors: No Defense solid, Defense is a lighter version of the same color
# Matching the paper's palette
# ---------------------------------------------------------------------------

DATASETS = ["MNIST", "Purchase100", "CIFAR-10 (CNN)", "CIFAR-10 (WRN-16)", "CIFAR-10 (CNN + AugMult4)"]

ND_COLORS = ["#C0504D", "#4472C4", "#70AD47", "#ED7D31", "#7030A0"]
D_COLORS  = ["#F4AAAA", "#9DC3E6", "#C6E0B4", "#F8CBAD", "#C5A3D8"]

# ---------------------------------------------------------------------------
# Layout: all ND bars left block, all D bars right block, small gap between
# ---------------------------------------------------------------------------

n_datasets = len(DATASETS)
n_eps      = len(epsilons)
width      = 0.07
block_gap  = 0.05
group_gap  = 0.25

group_width = n_datasets * width * 2 + block_gap
x_centers   = np.arange(n_eps) * (group_width + group_gap)

fig, ax = plt.subplots(figsize=(13, 4.5))

for ei, xc in enumerate(x_centers):
    total_block = n_datasets * width
    nd_start = xc - total_block - block_gap / 2
    d_start  = xc + block_gap / 2

    for di, dataset in enumerate(DATASETS):
        nd_x = nd_start + di * width
        d_x  = d_start  + di * width

        ax.bar(nd_x, no_defense[dataset][ei], width,
               color=ND_COLORS[di], alpha=0.88, edgecolor="none")
        ax.bar(d_x,  defense[dataset][ei],    width,
               color=D_COLORS[di],  alpha=0.88, edgecolor="none")

# Add dotted vertical separators between epsilon groups (like Figure 1)
for i in range(n_eps - 1):
    sep_x = (x_centers[i] + x_centers[i+1]) / 2
    ax.axvline(sep_x, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

# ---------------------------------------------------------------------------
# Legend: ND on left column, Defense on right column
# ---------------------------------------------------------------------------

nd_handles = [mpatches.Patch(color=ND_COLORS[i], label=f"{DATASETS[i]} (No Defense)")
              for i in range(n_datasets)]
d_handles  = [mpatches.Patch(color=D_COLORS[i],  label=f"{DATASETS[i]} (Defense)")
              for i in range(n_datasets)]

# Interleave so ncol=2 puts ND on left, Defense on right
handles = []
for nd, d in zip(nd_handles, d_handles):
    handles.append(nd)
    handles.append(d)

ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False,
          ncol=2, bbox_to_anchor=(0.0, 1.0))

ax.set_title("Privacy Tradeoff Curves: Defense vs No Defense", fontsize=12, pad=8)
ax.set_xlabel("Privacy Budget ($\\varepsilon$)", fontsize=10)
ax.set_ylabel("Empirical $\\varepsilon$ ($\\varepsilon_{\\mathrm{lb}}$)", fontsize=10)
ax.set_xticks(x_centers)
ax.set_xticklabels([str(e) for e in epsilons], fontsize=9)
ax.set_ylim(bottom=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()