import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Each list must match epsilon_values length (e.g., [eps6, eps8, eps10])
experiments_acc = {
    "MNIST (CNN)": {
        "no_defense": {"train": [0.9854633333, 0.98623, 0.9866133333],
                       "test":  [0.98438,      0.98474, 0.9854]},
        "defense":    {"train": [0.9784466667, 0.97899, 0.97928],
                       "test":  [0.97994,      0.9807,  0.98096]},
    },
    "Purchase100 (MLP)": {
        "no_defense": {"train": [0.8917834316, 0.8991358851, 0.9013075508],
                       "test":  [0.8590548587, 0.8699100469, 0.8719624984]},
        "defense":    {"train": [0.8636662137, 0.8756545881, 0.8829524249],
                       "test":  [0.8297326745, 0.8427619410, 0.8512707462]},
    },
    "CIFAR-10 (CNN)": {
        "no_defense": {"train": [0.699876, 0.741224, 0.763408],
                       "test":  [0.66688,  0.69842,  0.71274]},
        "defense":    {"train": [0.687828, 0.7341,   0.756652],
                       "test":  [0.65744,  0.69202,  0.7083]},
    },
    "CIFAR-10 (WideResnet-16)": {
        "no_defense": {"train": [0.72254,  0.760196, 0.774808],
                       "test":  [0.6953,   0.72502,  0.73216]},
        "defense":    {"train": [0.706712, 0.741124, 0.77136],
                       "test":  [0.68166,  0.70818,  0.72952]},
    },
    "CIFAR-10 (CNN + AugMult4)": {
        "no_defense": {"train": [0.659096, 0.694988, 0.71462],
                       "test":  [0.65284,  0.68624,  0.70448]},
        "defense":    {"train": [0.65262,  0.686908, 0.7092],
                       "test":  [0.64568,  0.67832,  0.70008]},
    },
}

epsilon_values = [6, 8, 10]

# Color palette (match your style)
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

import numpy as np
import os

# Plot
for idx, (exp_name, data) in enumerate(experiments_acc.items()):
    clean_name = exp_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    output_dir = f"plots_results/tradeoff_curves"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar settings
    x = np.arange(len(epsilon_values))  # label locations
    width = 0.2  # width of bars
    
    # Plot bars
    # Order: No Def Train, No Def Test, Def Train, Def Test
    rects1 = ax.bar(x - 1.5*width, data["no_defense"]["train"], width, label='No Defense Train', color='lightskyblue')
    rects2 = ax.bar(x - 0.5*width, data["no_defense"]["test"], width, label='No Defense Test', color='royalblue')
    rects3 = ax.bar(x + 0.5*width, data["defense"]["train"], width, label='Defense Train', color='lightgreen')
    rects4 = ax.bar(x + 1.5*width, data["defense"]["test"], width, label='Defense Test', color='forestgreen')

    # Labels + styling
    ax.set_xlabel('Privacy Budget (ε)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title(f'Accuracy Tradeoff: {exp_name}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(epsilon_values)
    ax.set_ylim(0.0, 1.05)

    # Legend
    ax.legend(loc='lower right', fontsize=10, title_fontsize=11)

    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_dir, f"{clean_name}_accs.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")
