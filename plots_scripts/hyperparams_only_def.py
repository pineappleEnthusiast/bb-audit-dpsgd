
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracies(x_values, train_acc, test_acc, experiment_name, output_dir, x_label="Parameter"):
    """
    Plots Training and Testing Accuracy as a grouped bar chart.
    """
    x = np.arange(len(x_values))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_acc, width, label='Train Acc', color='skyblue')
    plt.bar(x + width/2, test_acc, width, label='Test Acc', color='lightcoral')
    
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'{experiment_name}: Training and Testing Accuracy', fontsize=16)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(x, x_values, rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"{experiment_name.lower().replace(' ', '_')}_accuracy.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved accuracy plot to {filename}")

 
def plot_epsilon(x_values, epsilons, experiment_name, output_dir, x_label="Parameter"):
    """
    Plots Empirical Epsilon (CP) explicitly labeled as 'Empirical Epsilon LB' using a bar chart.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, epsilons, color='orange', label='Empirical Epsilon LB')
    
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Epsilon', fontsize=14)
    plt.title(f'{experiment_name}: Empirical Epsilon', fontsize=16)
    
    max_eps = max(epsilons) if epsilons else 0
    if max_eps <= 10:
        plt.ylim(0, 10)
    else:
        plt.ylim(0, max_eps * 1.2)

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"{experiment_name.lower().replace(' ', '_')}_epsilon.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved epsilon plot to {filename}")


def main():
    experiment_name = "Varying Filter Frequency (CIFAR-10, CNN)"
    output_directory = "plots_results/varying_filter_frequency"

    x_values = ["1", "5", "10", "20"]
    x_axis_label = "Filtering every k Epochs"

    train_accuracies = [0.759704, 0.763896, 0.762992, 0.766488]
    test_accuracies = [0.71426, 0.71308, 0.71154, 0.71504]
    cp_emp_epsilons = [1.0161, 0.4174, 0.05, 1.1553]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate Plots
    plot_accuracies(x_values, train_accuracies, test_accuracies, experiment_name, output_directory, x_label=x_axis_label)
    plot_epsilon(x_values, cp_emp_epsilons, experiment_name, output_directory, x_label=x_axis_label)

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()
