import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os

def main():
    # Output directory
    output_base = "plots_results/blank_alpha_ablations"
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    # Data
    # Alpha values for x-axis
    alpha_values = [0, 0.25, 0.5, 0.75, 1]
    
    # Structure:
    # "Experiment Name": {
    #     "no_defense": {"train": [...], "test": [...]},
    #     "defense":    {"train": [...], "test": [...]}
    # }
    experiments_acc = {
        "MNIST (CNN)": {
            "no_defense": {"train": [0.9866, 0.9866, 0.9866, 0.9866, 0.9867],
                           "test":  [0.9854, 0.9854, 0.9855, 0.9855, 0.9854]},
            "defense":    {"train": [0.9793, 0.9794, 0.9793, 0.9794, 0.9794],
                           "test":  [0.9810, 0.9811, 0.9810, 0.9810, 0.9810]},
        },
        "CIFAR-10 (CNN)": {
            "no_defense": {"train": [0.7634, 0.7635, 0.7659, 0.7643, 0.7651],
                           "test":  [0.7127, 0.7114, 0.7143, 0.7115, 0.7134]},
            "defense":    {"train": [0.7567, 0.7565, 0.7542, 0.7544, 0.7563],
                           "test":  [0.7083, 0.7088, 0.7057, 0.7082, 0.7089]},
        }
    }

    # Plot
    for exp_name, data in experiments_acc.items():
        clean_name = exp_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Bar settings
        x = np.arange(len(alpha_values))  # label locations
        width = 0.2  # width of bars
        
        # Plot bars
        # Order: No Def Train, No Def Test, Def Train, Def Test
        # Colors: lighter for train, darker for test
        # No Defense: Blueish
        # Defense: Greenish
        
        rects1 = ax.bar(x - 1.5*width, data["no_defense"]["train"], width, label='No Defense Train', color='lightskyblue')
        rects2 = ax.bar(x - 0.5*width, data["no_defense"]["test"], width, label='No Defense Test', color='royalblue')
        rects3 = ax.bar(x + 0.5*width, data["defense"]["train"], width, label='Defense Train', color='lightgreen')
        rects4 = ax.bar(x + 1.5*width, data["defense"]["test"], width, label='Defense Test', color='forestgreen')

        # Labels + styling
        ax.set_xlabel('Alpha', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'Blank Alpha Ablations: {exp_name}', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(alpha_values, fontsize=12)
        ax.set_ylim(0, 1.05)

        # Legend
        ax.legend(loc='lower right', fontsize=12)

        plt.tight_layout()
        
        # Save
        filename = f"{clean_name}_accs.png"
        save_path = os.path.join(output_base, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
