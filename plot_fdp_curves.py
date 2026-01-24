import numpy as np
import matplotlib.pyplot as plt
import os
from curves import convex_lower_bound, compute_empirical_fdp_curve

def plot_fdp_curve(losses_in, losses_out, title='f-DP Curve'):
    losses_D = losses_in
    losses_D_prime = losses_out
    
    alpha_emp, beta_emp = compute_empirical_fdp_curve(losses_D, losses_D_prime, use_holdout=True, seed=0)
    
    valid_mask = (alpha_emp > 0) & (beta_emp < 1)
    alpha_valid = alpha_emp[valid_mask]
    beta_valid = beta_emp[valid_mask]
    
    if len(alpha_valid) == 0:
        print(f"Error: No valid points for f-DP analysis in {title}")
        return
    
    alpha_bound, beta_bound = convex_lower_bound(alpha_valid, beta_valid)
    
    plt.figure(figsize=(8, 6))
    
    plt.scatter(alpha_emp, beta_emp, alpha=0.4, s=20, color='lightblue', label='All empirical points')
    plt.scatter(alpha_valid, beta_valid, alpha=0.7, s=30, color='blue')
    
    if len(alpha_bound) > 1:
        plt.plot(alpha_bound, beta_bound, 'r-', linewidth=3, label='Convex lower bound')
        plt.fill_between(alpha_bound, beta_bound, 1, alpha=0.2, color='red',
                       label='Upper bound region on true f-DP')
    
    plt.axline((0, 1), slope=-1, color='gray', linestyle='--', alpha=0.7, 
              label='β = 1 - α (perfect privacy)')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('α (False Positive Rate)', fontsize=12)
    plt.ylabel('β (False Negative Rate)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # plt.savefig('fdp_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    file_paths = [
        ("tradeoff_curves/mnist_no_defense_eps10/mnist_cnn_eps10.0/", "MNIST, CNN, Eps=10, No Defense"),
        ("tradeoff_curves/mnist_defense_eps10/mnist_cnn_eps10.0/", "MNIST, CNN, Eps=10, With Defense"),
    ]
    
    for dir_path, title in file_paths:
        losses_in = np.load(os.path.join(dir_path, 'losses_in.npy'))
        losses_out = np.load(os.path.join(dir_path, 'losses_out.npy'))
        plot_fdp_curve(losses_in, losses_out, title)

if __name__ == '__main__':
    main()
