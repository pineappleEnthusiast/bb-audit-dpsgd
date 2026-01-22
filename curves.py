import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

def convex_lower_bound(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the convex lower bound (lower convex hull) of points (x, y).
    This gives an upper bound on the true f-DP curve.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    # Sort points by x-coordinate
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Remove duplicate x values, keeping the minimum y value
    unique_x = []
    unique_y = []
    i = 0
    while i < len(x_sorted):
        current_x = x_sorted[i]
        min_y = y_sorted[i]
        
        # Find all points with the same x value and take the minimum y
        while i < len(x_sorted) and x_sorted[i] == current_x:
            min_y = min(min_y, y_sorted[i])
            i += 1
        
        unique_x.append(current_x)
        unique_y.append(min_y)
    
    x_unique = np.array(unique_x)
    y_unique = np.array(unique_y)
    
    if len(x_unique) < 2:
        return x_unique, y_unique
    
    # Compute lower convex hull using monotone chain algorithm
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    points = list(zip(x_unique, y_unique))
    hull = []
    
    # Build lower hull
    for p in points:
        while len(hull) >= 2 and cross_product(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    
    if hull:
        x_hull, y_hull = zip(*hull)
        return np.array(x_hull), np.array(y_hull)
    else:
        return np.array([]), np.array([])

def compute_empirical_fdp_curve(losses_D: np.ndarray, losses_D_prime: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical f-DP curve from loss data.
    
    Parameters:
    losses_D: Losses on canary for models trained on D
    losses_D_prime: Losses on canary for models trained on D'
    
    Returns:
    alpha_values: False positive rates
    beta_values: 1 - True positive rates
    """
    all_losses = np.concatenate([losses_D, losses_D_prime])
    
    # True labels: 0 for D, 1 for D'
    true_labels = np.concatenate([
        np.zeros(len(losses_D)),      # Models trained on D
        np.ones(len(losses_D_prime))  # Models trained on D'
    ])
    
    # Use all unique loss values as thresholds
    thresholds = np.unique(all_losses)
    alpha_values = []
    beta_values = []
    
    for threshold in thresholds:
        # Predict D' (label 1) if loss <= threshold
        predictions = (all_losses <= threshold).astype(int)
        
        tp = np.sum((predictions == 1) & (true_labels == 1))  # True positives
        fp = np.sum((predictions == 1) & (true_labels == 0))  # False positives
        
        total_D = len(losses_D)        # Total models trained on D
        total_D_prime = len(losses_D_prime)  # Total models trained on D'
        
        alpha = fp / total_D if total_D > 0 else 0        # False positive rate
        tpr = tp / total_D_prime if total_D_prime > 0 else 0  # True positive rate
        beta = 1 - tpr                                   # 1 - true positive rate
        
        alpha_values.append(alpha)
        beta_values.append(beta)
    
    return np.array(alpha_values), np.array(beta_values)

def compute_delta_for_epsilon(eps: float, alpha_points: np.ndarray, 
                              beta_points: np.ndarray, f_interp) -> float:
    """
    Compute δ(ε) for a given ε using the convex conjugate.
    
    Parameters:
    eps: Privacy parameter epsilon
    alpha_points: Discrete alpha values from f-DP curve
    beta_points: Discrete beta values from f-DP curve
    f_interp: Interpolation function for the f-DP curve
    
    Returns:
    delta_eps: The value of δ(ε)
    """
    t = -np.exp(eps)
    
    # Define the function to maximize: α*t - f(α)
    def objective(alpha_val):
        return -(alpha_val * t - f_interp(alpha_val))  # Negative for minimization
    
    # Find the maximum over [0, 1]
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    f_star_t = -result.fun  # Convert back from minimization
    
    delta_eps = 1 + f_star_t
    return delta_eps

def compute_epsilon_delta_curve(alpha: np.ndarray, beta: np.ndarray,
                               delta_min: float = 1e-10, delta_max: float = 1.0,
                               num_delta_points: int = 1000,
                               epsilon_search_max: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the epsilon(delta) curve by inverting the delta(epsilon) relationship.
    
    For each target δ, we find the minimum ε such that δ(ε) ≤ target_δ using binary search.
    
    Parameters:
    alpha: False positive rates (α) from the f-DP curve
    beta: 1 - True positive rates (β) from the f-DP curve
    delta_min: Minimum delta value to compute
    delta_max: Maximum delta value to compute
    num_delta_points: Number of delta points to compute
    epsilon_search_max: Maximum epsilon value to search
    
    Returns:
    epsilon_values: Values of ε(δ) (privacy parameter for each delta)
    delta_values: Corresponding δ values
    """
    # Ensure we have valid data
    if len(alpha) < 2 or len(delta) < 2:
        return np.array([]), np.array([])
    
    # Sort by alpha and remove duplicates
    unique_alpha, unique_idx = np.unique(alpha, return_index=True)
    unique_beta = beta[unique_idx]
    
    # Sort by alpha
    sort_idx = np.argsort(unique_alpha)
    unique_alpha = unique_alpha[sort_idx]
    unique_beta = unique_beta[sort_idx]
    
    # Ensure we start at (0,1) and end at (1,0) for proper interpolation
    if unique_alpha[0] > 0:
        unique_alpha = np.concatenate([[0], unique_alpha])
        unique_beta = np.concatenate([[1], unique_beta])
    
    if unique_alpha[-1] < 1:
        unique_alpha = np.concatenate([unique_alpha, [1]])
        unique_beta = np.concatenate([unique_beta, [0]])
    
    # Create interpolation function
    f_interp = interp1d(unique_alpha, unique_beta, kind='linear', 
                        bounds_error=False, fill_value=(1.0, 0.0))
    
    # Create delta values - use log spacing for better coverage
    delta_targets = np.logspace(np.log10(delta_min), np.log10(delta_max), num_delta_points)
    epsilon_values = []
    valid_delta_values = []
    
    print("Computing ε(δ) curve using binary search...")
    print(f"Target: δ range [{delta_min:.1e}, {delta_max:.1e}]")
    print(f"Epsilon search range: [0, {epsilon_search_max}]")
    
    for i, target_delta in enumerate(delta_targets):
        # Binary search for minimum epsilon such that δ(ε) ≤ target_delta
        eps_low = 0.0
        eps_high = epsilon_search_max
        tolerance = 1e-6
        max_iterations = 100
        
        found_valid = False
        best_eps = None
        
        for iteration in range(max_iterations):
            eps_mid = (eps_low + eps_high) / 2
            delta_mid = compute_delta_for_epsilon(eps_mid, unique_alpha, unique_beta, f_interp)
            
            if delta_mid <= target_delta:
                # Found a valid epsilon, try to find smaller one
                found_valid = True
                best_eps = eps_mid
                eps_high = eps_mid
            else:
                # Need larger epsilon
                eps_low = eps_mid
            
            # Check convergence
            if eps_high - eps_low < tolerance:
                break
        
        if found_valid and best_eps is not None:
            epsilon_values.append(best_eps)
            valid_delta_values.append(target_delta)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            if found_valid:
                print(f"  Progress: {i+1}/{len(delta_targets)}, δ = {target_delta:.2e}, ε(δ) = {best_eps:.2f}")
            else:
                print(f"  Progress: {i+1}/{len(delta_targets)}, δ = {target_delta:.2e}, ε(δ) not found")
    
    epsilon_array = np.array(epsilon_values)
    delta_array = np.array(valid_delta_values)
    
    print(f"\nSuccessfully computed {len(epsilon_array)} points")
    if len(epsilon_array) > 0:
        print(f"ε(δ) range: [{np.min(epsilon_array):.2f}, {np.max(epsilon_array):.2f}]")
    
    return epsilon_array, delta_array

def plot_fdp_curves(losses_in_file: str = "losses_in.npy", 
                   losses_out_file: str = "losses_out.npy",
                   plot_epsilon_delta: bool = True):
    """
    Load data and plot empirical f-DP curve with convex lower bound.
    
    Parameters:
    losses_in_file: Path to losses for models trained on D
    losses_out_file: Path to losses for models trained on D'
    plot_epsilon_delta: Whether to plot ε(δ) curve
    """
    try:
        # Load data
        print(f"Loading {losses_in_file}...")
        losses_D = np.load(losses_in_file)
        print(f"Loaded {len(losses_D)} loss values for models trained on D")
        
        print(f"Loading {losses_out_file}...")
        losses_D_prime = np.load(losses_out_file)
        print(f"Loaded {len(losses_D_prime)} loss values for models trained on D'")
        
        # Basic statistics
        print(f"\n=== Data Statistics ===")
        print(f"losses_in (D): mean={np.mean(losses_D):.4f}, std={np.std(losses_D):.4f}")
        print(f"losses_out (D'): mean={np.mean(losses_D_prime):.4f}, std={np.std(losses_D_prime):.4f}")
        
        # Compute empirical f-DP curve
        alpha_emp, beta_emp = compute_empirical_fdp_curve(losses_D, losses_D_prime)
        
        # Remove trivial points (alpha = 0 or beta = 1)
        valid_mask = (alpha_emp > 0) & (beta_emp < 1)
        alpha_valid = alpha_emp[valid_mask]
        beta_valid = beta_emp[valid_mask]
        
        if len(alpha_valid) == 0:
            print("Error: No valid points for f-DP analysis")
            return
        
        # Compute convex lower bound
        alpha_bound, beta_bound = convex_lower_bound(alpha_valid, beta_valid)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot all empirical points
        plt.scatter(alpha_emp, beta_emp, alpha=0.4, s=20, color='lightblue', 
                   label='All empirical points')
        
        # Plot valid empirical points
        plt.scatter(alpha_valid, beta_valid, alpha=0.7, s=30, color='blue')
        
        # Plot convex lower bound
        if len(alpha_bound) > 1:
            plt.plot(alpha_bound, beta_bound, 'r-', linewidth=3, 
                    label='Convex lower bound')
            plt.fill_between(alpha_bound, beta_bound, 1, alpha=0.2, color='red',
                           label='Upper bound region on true f-DP')
        
        # Add reference lines
        plt.axline((0, 1), slope=-1, color='gray', linestyle='--', alpha=0.7, 
                  label='β = 1 - α (perfect privacy)')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        # Formatting
        plt.xlabel('α (False Positive Rate)', fontsize=14)
        plt.ylabel('β (False Negative Rate)', fontsize=14)
        plt.title('Empirical f-DP Curve and Convex Lower Bound', fontsize=16)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add summary text
        summary_text = (f'Models on D: {len(losses_D)}\n'
                       f'Models on D\': {len(losses_D_prime)}\n'
                       f'Convex bound points: {len(alpha_bound)}')
        
        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print some key points from the convex bound
        print(f"\n=== Convex Lower Bound Key Points ===")
        for i, (a, d) in enumerate(zip(alpha_bound, beta_bound)):
            print(f"Point {i+1}: α = {a:.4f}, β = {d:.4f}")
            
        # Plot epsilon(delta) curve if requested
        if plot_epsilon_delta and len(alpha_bound) > 1:
            print("\nComputing ε(δ) curve...")
            epsilon_vals, delta_vals = compute_epsilon_delta_curve(
                alpha_bound, beta_bound, 
                delta_min=1e-10, delta_max=0.99,
                num_delta_points=500, 
                epsilon_search_max=50.0)
            
            if len(epsilon_vals) > 0:
                plt.figure(figsize=(12, 6))
                
                # Plot epsilon(delta) curve
                plt.semilogx(delta_vals, epsilon_vals, 'b-', linewidth=2, 
                          label='ε(δ) - Minimum ε for target δ')
                
                # Add reference lines
                plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='ε = 1')
                plt.axvline(x=1e-5, color='gray', linestyle='--', alpha=0.5, label='δ = 10⁻⁵')
                
                plt.xlabel('δ (Privacy Parameter)', fontsize=14)
                plt.ylabel('ε(δ) (Minimum Privacy Budget)', fontsize=14)
                plt.title('ε(δ) Curve: Minimum ε Required for Target δ', fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3, which='both')
                plt.tight_layout()
                
                # Print key points from the epsilon(delta) curve
                print("\n=== Epsilon(Delta) Key Points ===")
                
                # Show epsilon values for specific delta targets
                delta_targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
                for delta_target in delta_targets:
                    # Find the epsilon value for this delta
                    if delta_target >= delta_vals[0] and delta_target <= delta_vals[-1]:
                        idx = np.argmin(np.abs(delta_vals - delta_target))
                        actual_delta = delta_vals[idx]
                        eps_val = epsilon_vals[idx]
                        print(f"δ = {actual_delta:.2e}, ε(δ) = {eps_val:.2f}")
                
                plt.show()
            else:
                print("\nCould not compute epsilon(delta) curve - not enough valid points")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        print("Please make sure both files are in the current directory.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    # Try to load and plot your actual data
    
    print("Loading and plotting...")
    # plot_fdp_curves(r"results\test_mnist_fgsm\mnist_cnn_eps10.0\losses_in.npy", 
    #                r"results\test_mnist_fgsm\mnist_cnn_eps10.0\losses_out.npy")
    # plot_fdp_curves(r"local\results\no_defense\test_mnist_fgsm\mnist_cnn_eps10.0\losses_in.npy", 
    #             r"local\results\no_defense\test_mnist_fgsm\mnist_cnn_eps10.0\losses_out.npy")

    plot_fdp_curves("cifar10_fgsm_canary_no_defense/cifar10_cnn_eps10.0/losses_in.npy", 
                "cifar10_fgsm_canary_no_defense/cifar10_cnn_eps10.0/losses_out.npy")