import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    # ========== EDIT THESE VARIABLES ==========
    per_sample_eps_file = 'mnist_do_we_expose_anyone_else/mnist_cnn_eps10.0/per_sample_epsilons.npy'
    canary_eps = 1.9863  # Empirical epsilon of canary from no defense run
    # ===========================================
    
    # Load per-sample epsilons
    per_sample_eps = np.load(per_sample_eps_file)
    
    # Get top 10 most vulnerable samples
    top_k = 10
    top_indices = np.argsort(per_sample_eps)[-top_k:][::-1]
    top_epsilons = per_sample_eps[top_indices]
    
    # Create labels
    labels = [f'Sample {idx}' for idx in top_indices] + ['Canary\n(No Defense)']
    values = np.concatenate([top_epsilons, [canary_eps]])
    
    # Create colors: blue for regular samples, red for canary
    colors = ['#3498db'] * top_k + ['#e74c3c']
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize plot
    plt.xlabel('Sample', fontsize=12, fontweight='bold')
    plt.ylabel('Empirical Epsilon (ε)', fontsize=12, fontweight='bold')
    plt.title('Top 10 Most Vulnerable Samples After Defense vs Canary (No Defense)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Regular Samples'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Canary (No Defense)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    # Show plot
    # create directory first
    os.makedirs('plots_results/do_we_expose_anyone_else', exist_ok=True)
    plt.savefig('plots_results/do_we_expose_anyone_else/top_vulnerable_samples.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
