import matplotlib.pyplot as plt
import numpy as np
import os

def plot_grouped_epsilon(x_groups, no_defense_eps, defense_eps, experiment_name, output_dir):
    """
    Plots Empirical Epsilon as a grouped bar chart.
    """
    x = np.arange(len(x_groups))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(10, 6))
    # No Defense bars - using skyblue from reference script
    rects1 = plt.bar(x - width/2, no_defense_eps, width, label='No Defense', color='skyblue')
    # Defense bars - using lightcoral from reference script
    rects2 = plt.bar(x + width/2, defense_eps, width, label='With Defense', color='lightcoral')
    
    plt.xlabel('Sampling Method', fontsize=14)
    plt.ylabel('Empirical Epsilon', fontsize=14)
    plt.title(f'{experiment_name}', fontsize=16)
    


    # Set y-limit with some headroom
    max_val = max(max(no_defense_eps), max(defense_eps))
    plt.ylim(0, max_val * 1.2 if max_val > 0 else 1.0)
    
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(x, x_groups, rotation=0, fontsize=12)
    plt.tight_layout()
    
    filename = "poisson_vs_shuffled_epsilon.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    experiment_name = "MNIST CNN: Empirical Epsilon by Sampling Method"
    output_directory = "plots_results/sampling_comparison/"

    # Data
    x_groups = ["Poisson Sampling", "Shuffled Sampling"]
    
    no_defense_eps = [3.5973, 1.9863]
    defense_eps = [0.05, 0.05]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate Plot
    plot_grouped_epsilon(x_groups, no_defense_eps, defense_eps, experiment_name, output_directory)

if __name__ == "__main__":
    main()
