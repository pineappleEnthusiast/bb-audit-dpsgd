import numpy as np
import matplotlib.pyplot as plt
import os

def plot_loss_distribution(losses_in, losses_out, title='Loss Distribution'):
    # print(f"\n{title}:")
    # print(f"  losses_in shape: {losses_in.shape}, values: {losses_in}")
    # print(f"  losses_out shape: {losses_out.shape}, values: {losses_out}")
    
    plt.figure(figsize=(8, 5))
    plt.hist(losses_in, bins=50, alpha=0.6, label='Member (In)', color='#2E86AB', density=True)
    plt.hist(losses_out, bins=50, alpha=0.6, label='Non-Member (Out)', color='#A23B72', density=True)
    plt.xlabel('Loss', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # plt.savefig('loss_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    file_paths = [
        ("tradeoff_curves/mnist_no_defense_eps10/mnist_cnn_eps10.0/", "MNIST, CNN, Eps=10, No Defense"),
        ("tradeoff_curves/mnist_defense_eps10/mnist_cnn_eps10.0/", "MNIST, CNN, Eps=10, With Defense"),
    ]
    
    for dir_path, title in file_paths:
        losses_in = np.load(os.path.join(dir_path, 'losses_in.npy'))
        losses_out = np.load(os.path.join(dir_path, 'losses_out.npy'))
        plot_loss_distribution(losses_in, losses_out, title)

if __name__ == '__main__':
    main()
