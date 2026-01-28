import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    output_dir = "plots_results/non_private_tradeoffs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Data - REPLACE these with your actual numbers
    datasets = [
        "MNIST (CNN)", 
        "Purchase100\n(MLP)", 
        "CIFAR-10\n(CNN)", 
        "CIFAR-10\n(WideResNet)", 
        "CIFAR-10\n(AugMult4)"
    ]
    
    # Structure: [MNIST, Purchase, CIFAR-CNN, CIFAR-WRN, CIFAR-AugMult]
    data = {
        "no_defense": {
            "train": [0.9862066667, 0.8886104591, 0.931764, 1.0, 0.796672],
            "test":  [0.98442,      0.8441099709, 0.74334,  0.7044, 0.7715],
            "eps":   [14.6179,      0.1,          15.8092,  7.9041, 7.8194]
        },
        "defense": {
            "train": [0.9692833333, 0.9397239254, 0.89844, 0.86166, 0.788116],
            "test":  [0.97174,      0.8891777524, 0.74086, 0.7599,  0.76556],
            "eps":   [0.1,          0.1,          0.1,     0.1,     0.5241]
        }
    }


    # ==========================================
    # Plot 1: Accuracies (Bar Graph)
    # ==========================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(datasets))
    width = 0.2  # width of each bar
    
    # Order: No Def Train, No Def Test, Def Train, Def Test
    rects1 = ax.bar(x - 1.5*width, data["no_defense"]["train"], width, label='No Defense Train', color='lightskyblue')
    rects2 = ax.bar(x - 0.5*width, data["no_defense"]["test"], width, label='No Defense Test', color='royalblue')
    rects3 = ax.bar(x + 0.5*width, data["defense"]["train"], width, label='Defense Train', color='lightgreen')
    rects4 = ax.bar(x + 1.5*width, data["defense"]["test"], width, label='Defense Test', color='forestgreen')

    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Non-Private Model Accuracies: Defense vs No Defense', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # autolabel removed

    plt.tight_layout()
    save_path_acc = os.path.join(output_dir, "non_private_accuracies.png")
    plt.savefig(save_path_acc, dpi=300)
    print(f"Saved accuracy plot to {save_path_acc}")
    plt.close()

    # ==========================================
    # Plot 2: Empirical Epsilon (Bar Graph)
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    width = 0.35
    
    rects1 = ax.bar(x - width/2, data["no_defense"]["eps"], width, label='No Defense', color='mediumpurple')
    rects2 = ax.bar(x + width/2, data["defense"]["eps"], width, label='Defense', color='purple', hatch='//', alpha=0.7)

    ax.set_ylabel('Empirical Epsilon', fontsize=14, fontweight='bold')
    ax.set_title('Non-Private Empirical Epsilon: Defense vs No Defense', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # autolabel_eps removed

    plt.tight_layout()
    save_path_eps = os.path.join(output_dir, "non_private_epsilon.png")
    plt.savefig(save_path_eps, dpi=300)
    print(f"Saved epsilon plot to {save_path_eps}")
    plt.close()

if __name__ == "__main__":
    main()
