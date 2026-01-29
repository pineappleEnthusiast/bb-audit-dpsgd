# FINAL LINE PLOT

# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# experiments = {
#     'MNIST': {
#         'no_defense': [1.1894, 1.7118, 1.9863],
#         'defense': [0, 0, 0]
#     },
#     'Purchase100': {
#         'no_defense': [1.6011, 1.9243, 2.4953],
#         'defense': [0.02, 0.02, 0.02]
#     },
#     'CIFAR-10 (CNN)': {
#         'no_defense': [0, 0.8009, 1.0651],
#         'defense': [0.04, 0.04, 0.04]
#     },
#     'CIFAR-10 (WideResnet-16)': {
#         'no_defense': [0, 0.6045, 0.7172],
#         'defense': [0, 0.1789, 0]
#     },
#     'CIFAR-10 (CNN + AugMult4)': {
#         'no_defense': [0.4058, 1.4891, 1.0727],
#         'defense': [0.06, 0.06, 0.06]
#     }
# }

# epsilon_values = [6, 8, 10]

# # Color palette
# colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# # Create single plot
# fig, ax = plt.subplots(figsize=(12, 7))

# # Plot each experiment
# for idx, (exp_name, data) in enumerate(experiments.items()):
#     color = colors[idx]
    
#     # Plot no defense line (solid)
#     ax.plot(epsilon_values, data['no_defense'], 
#             marker='o', linewidth=2.5, markersize=8, 
#             label=f'{exp_name} (No Defense)', 
#             color=color, linestyle='-')
    
#     # Plot defense line (dashed)
#     ax.plot(epsilon_values, data['defense'], 
#             marker='s', linewidth=2.5, markersize=8, 
#             label=f'{exp_name} (Defense)', 
#             color=color, linestyle='--', alpha=0.7)

# # Styling
# ax.set_xlabel('Privacy Budget (ε)', fontsize=13, fontweight='bold')
# ax.set_ylabel('Empirical ε', fontsize=13, fontweight='bold')
# ax.set_title('Privacy Tradeoff Curves: Defense vs No Defense', 
#              fontsize=14, fontweight='bold', pad=20)
# ax.grid(True, alpha=0.3, linestyle='--')
# ax.legend(loc='upper left', fontsize=9, ncol=2)
# ax.set_xticks(epsilon_values)
# ax.set_ylim(-0.1, 2.7)

# plt.tight_layout()
# plt.savefig('plots_results/tradeoff_curves/privacy_tradeoff_curves_line.png', dpi=300, bbox_inches='tight')






# FINAL BAR GRAPH

import matplotlib.pyplot as plt
import numpy as np

# Data
experiments = {
    'MNIST': {
        'no_defense': [1.1894, 1.7118, 1.9863],
        'defense': [0.02, 0.02, 0.02]
    },
    'Purchase100': {
        'no_defense': [1.6011, 1.9243, 2.4953],
        'defense': [0.02, 0.02, 0.02]
    },
    'CIFAR-10 (CNN)': {
        'no_defense': [0.02, 0.8009, 1.0651],
        'defense': [0.02, 0.02, 0.02]
    },
    'CIFAR-10 (WideResnet-16)': {
        'no_defense': [0.02, 0.6045, 0.7172],
        'defense': [0.02, 0.1789, 0.02]
    },
    'CIFAR-10 (CNN + AugMult4)': {
        'no_defense': [0.4058, 1.4891, 1.0727],
        'defense': [0.02, 0.02, 0.02]
    }
}

epsilon_values = [6, 8, 10]

# Color palette
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# Create single plot
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(epsilon_values))
n_experiments = len(experiments)
total_width = 0.8
bar_width = total_width / (n_experiments * 2)

# Plot each experiment
for idx, (exp_name, data) in enumerate(experiments.items()):
    color = colors[idx]
    
    # Calculate offsets
    # We want all No Defense bars first, then all Defense bars
    # Order: [ND_Exp1, ND_Exp2, ..., ND_ExpN, D_Exp1, D_Exp2, ..., D_ExpN]
    
    # Global index for No Defense bar
    pos_nd = idx
    offset_nd = pos_nd * bar_width - (total_width / 2) + (bar_width / 2)
    
    # Global index for Defense bar
    pos_d = n_experiments + idx
    offset_d = pos_d * bar_width - (total_width / 2) + (bar_width / 2)
    
    # Plot no defense bar
    ax.bar(x + offset_nd, data['no_defense'], 
            width=bar_width, 
            label=f'{exp_name}', 
            color=color, alpha=0.9, edgecolor='white')
    
    # Plot defense bar (hatched to distinguish, or slightly lighter/different shade?)
    # Using hatching '//' for defense
    ax.bar(x + offset_d, data['defense'], 
            width=bar_width, 
            label=f'{exp_name}', 
            color=color, alpha=0.5, hatch='//', edgecolor=color)

# Styling
# Add vertical dotted line to separate Defense and No Defense groups
for val in x:
    ax.axvline(x=val, color='black', linestyle=':', linewidth=2, alpha=0.5)

ax.set_xlabel('Privacy Budget (ε)', fontsize=14, fontweight='bold')
ax.set_ylabel('Empirical ε', fontsize=14, fontweight='bold')
ax.set_title('Privacy Tradeoff Curves: Defense vs No Defense', 
             fontsize=18, fontweight='bold', pad=20)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(epsilon_values, fontsize=12)

# handles, labels = ax.get_legend_handles_labels()
# sorted_handles = handles[::2] + handles[1::2]
# sorted_labels = labels[::2] + labels[1::2]
# ax.legend(handles=sorted_handles, labels=sorted_labels, loc='upper left', fontsize=13.5, ncol=2)


# Reorder legend: All No Defense (evens) then All Defense (odds)
handles, labels = ax.get_legend_handles_labels()
sorted_handles = handles[::2] + handles[1::2]
sorted_labels = labels[::2] + labels[1::2]

# Create invisible handle for headers
from matplotlib.patches import Patch
header_handle = Patch(facecolor='none', edgecolor='none')

# Insert headers at the beginning of each column
n_experiments = len(sorted_handles) // 2
sorted_handles.insert(0, header_handle)  # Before No Defense experiments
sorted_handles.insert(n_experiments + 1, header_handle)  # Before Defense experiments
sorted_labels.insert(0, 'No Defense')
sorted_labels.insert(n_experiments + 1, 'Defense')

ax.legend(handles=sorted_handles, labels=sorted_labels, loc='upper left', fontsize=16, ncol=2)




ax.set_ylim(0, 3.0)

plt.tight_layout()
plt.savefig('plots_results/tradeoff_curves/privacy_tradeoff_curves_bar.png', dpi=300, bbox_inches='tight')
# plt.show()