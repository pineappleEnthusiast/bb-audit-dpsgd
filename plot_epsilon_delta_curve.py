import numpy as np
import matplotlib.pyplot as plt
import os
from curves import convex_lower_bound, compute_empirical_fdp_curve, compute_epsilon_delta_curve

def plot_epsilon_delta_curve(losses_in, losses_out, title='ε-δ Curve'):
    losses_D = losses_in
    losses_D_prime = losses_out
    
    alpha_emp, beta_emp = compute_empirical_fdp_curve(losses_D, losses_D_prime, use_holdout=True, seed=0)
    
    valid_mask = (alpha_emp > 0) & (beta_emp < 1)
    alpha_valid = alpha_emp[valid_mask]
    beta_valid = beta_emp[valid_mask]
    
    if len(alpha_valid) == 0:
        print(f"Error: No valid points for epsilon-delta analysis in {title}")
        return
    
    alpha_bound, beta_bound = convex_lower_bound(alpha_valid, beta_valid)
    
    if len(alpha_bound) < 2:
        print(f"Error: Not enough points for epsilon-delta curve in {title}")
        return
    
    epsilon_vals, delta_vals = compute_epsilon_delta_curve(
        alpha_bound, beta_bound, 
        delta_min=1e-10, delta_max=0.99,
        num_delta_points=500, 
        epsilon_search_max=50.0)
    
    if len(epsilon_vals) == 0:
        print(f"Error: Could not compute epsilon-delta curve for {title}")
        return
    
    plt.figure(figsize=(10, 6))
    
    plt.semilogx(delta_vals, epsilon_vals, 'b-', linewidth=2, 
              label='ε(δ) - Minimum ε for target δ')
    
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='ε = 1')
    plt.axvline(x=1e-5, color='gray', linestyle='--', alpha=0.5, label='δ = 10⁻⁵')
    
    plt.xlabel('δ (Privacy Parameter)', fontsize=12)
    plt.ylabel('ε(δ) (Minimum Privacy Budget)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    # plt.savefig('epsilon_delta_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    file_paths = [
        ("tradeoff_curves/mnist_no_defense_eps10/mnist_cnn_eps10.0/", "MNIST, CNN, Eps=10, No Defense"),
        ("tradeoff_curves/mnist_defense_eps10/mnist_cnn_eps10.0/", "MNIST, CNN, Eps=10, With Defense"),
    ]
    
    for dir_path, title in file_paths:
        losses_in = np.load(os.path.join(dir_path, 'losses_in.npy'))
        losses_out = np.load(os.path.join(dir_path, 'losses_out.npy'))
        plot_epsilon_delta_curve(losses_in, losses_out, title)

if __name__ == '__main__':
    main()
