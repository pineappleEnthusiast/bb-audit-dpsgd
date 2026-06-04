import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── New data from spreadsheet ──────────────────────────────────────────────────
sampling_methods = ["Poisson Sampling", "Shuffled Sampling"]

no_defense = [3.708773, 3.321261]   # rows 78 & 76  (no defense)
defense    = [0.05,      0.05     ]   # rows 79 & 77  (defense)

# ── Layout ────────────────────────────────────────────────────────────────────
x          = np.arange(len(sampling_methods))
bar_width  = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

# Match the original colour scheme: sky-blue for No Defense, salmon for Defense
color_no_def = "#87CEEB"   # light sky blue
color_def    = "#FA8072"   # salmon / coral-red

bars_no_def = ax.bar(x - bar_width / 2, no_defense, bar_width,
                     color=color_no_def, label="No Defense")
bars_def    = ax.bar(x + bar_width / 2, defense,    bar_width,
                     color=color_def,    label="With Defense")

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_title("MNIST CNN: Empirical Epsilon by Sampling Method", fontsize=12)
ax.set_xlabel("Sampling Method", fontsize=11)
ax.set_ylabel("Empirical Epsilon", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(sampling_methods, fontsize=11)

# y-axis: match original range (0 – 4.0) with 0.5 step
ax.set_ylim(0, 4.25)
ax.set_yticks(np.arange(0, 4.5, 0.5))
ax.yaxis.grid(True, linewidth=0.5, color="lightgrey")
ax.set_axisbelow(True)

# Remove top / right spines to keep the clean look of the original
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
patch_no_def = mpatches.Patch(color=color_no_def, label="No Defense")
patch_def    = mpatches.Patch(color=color_def,    label="With Defense")
ax.legend(handles=[patch_no_def, patch_def], loc="upper right", frameon=True,
          fontsize=10)

plt.tight_layout()
plt.show()