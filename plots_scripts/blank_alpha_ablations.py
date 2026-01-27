import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    output_directory = "plots_results/blank_alpha_ablations"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Data
    alpha_values = [0, 0.25, 0.5, 0.75, 1]
    
    # Format: "Label": [values_for_each_alpha]
    data = {
        "MNIST (Defense)":    [0.02, 0.02, 0.02, 0.02, 0.02],
        "MNIST (No Defense)": [1.9863, 0.02, 0.02, 0.02, 0.02],
        "CIFAR-10 (Defense)": [0.02, 0.02, 0.02, 0.02, 0.02],
        "CIFAR-10 (No Defense)": [1.0651, 0.02, 0.02, 0.02, 0.02]
    }

    # Plot settings
    x = np.arange(len(alpha_values))  # label locations
    width = 0.2  # width of bars
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors suitable for 4 groups
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    
    # Plotting bars
    # We have 4 bars per group. Center them around x.
    # Offsets: -1.5*w, -0.5*w, 0.5*w, 1.5*w
    
    rects1 = ax.bar(x - 1.5*width, data["MNIST (Defense)"], width, label='MNIST (Defense)', color=colors[0])
    rects2 = ax.bar(x - 0.5*width, data["MNIST (No Defense)"], width, label='MNIST (No Defense)', color=colors[1])
    rects3 = ax.bar(x + 0.5*width, data["CIFAR-10 (Defense)"], width, label='CIFAR-10 (Defense)', color=colors[2])
    rects4 = ax.bar(x + 1.5*width, data["CIFAR-10 (No Defense)"], width, label='CIFAR-10 (No Defense)', color=colors[3])

    # Styling
    ax.set_xlabel('Alpha', fontsize=14, fontweight='bold')
    ax.set_ylabel('Empirical Epsilon', fontsize=14, fontweight='bold')
    ax.set_title('Blank Alpha Ablations', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_values, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Set y-limit slightly higher than max value to look good
    max_val = max([max(v) for v in data.values()])
    ax.set_ylim(0, max_val * 1.2 if max_val > 0 else 1.0)
    
    plt.tight_layout()
    
    # Save
    filename = "blank_alpha_ablations.png"
    save_path = os.path.join(output_directory, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
