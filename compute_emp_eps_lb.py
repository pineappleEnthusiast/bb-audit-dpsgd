import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from curves import convex_lower_bound, compute_empirical_fdp_curve, compute_epsilon_delta_curve

def plot_fdp_curves(losses_in_file: str = "losses_in.npy", 
                   losses_out_file: str = "losses_out.npy",
                   use_holdout: bool = True):

    # Simple direct loading
    losses_D = np.load(losses_in_file)
    losses_D_prime = np.load(losses_out_file)
    
    # Compute empirical f-DP curve
    alpha_emp, beta_emp = compute_empirical_fdp_curve(losses_D, losses_D_prime, use_holdout=use_holdout, seed=0)
    
    # Remove trivial points (alpha = 0 or beta = 1)
    valid_mask = (alpha_emp > 0) & (beta_emp < 1)
    alpha_valid = alpha_emp[valid_mask]
    beta_valid = beta_emp[valid_mask]
    
    if len(alpha_valid) == 0:
        print("Error: No valid points for f-DP analysis")
        return
    
    # Compute convex lower bound
    alpha_bound, beta_bound = convex_lower_bound(alpha_valid, beta_valid)
    
    # Plot epsilon(delta) curve if requested
    if len(alpha_bound) > 1:
        epsilon_vals, delta_vals = compute_epsilon_delta_curve(
            alpha_bound, beta_bound, 
            delta_min=1e-10, delta_max=0.99,
            num_delta_points=500, 
            epsilon_search_max=50.0)
        
        if len(epsilon_vals) > 0:
            delta_target = 1e-5
            idx = np.argmin(np.abs(delta_vals - delta_target))
            actual_delta = delta_vals[idx]
            eps_val = epsilon_vals[idx]
            print(f"\n*** EPSILON AT DELTA=1e-5: ε = {eps_val:.4f} ***\n")
            
        else:
            print("\nCould not compute epsilon(delta) curve - not enough valid points")

if __name__ == "__main__":
    experiments = [
        "non_private_audits_rerun/purchase_defense_non_private/purchase_mlp_epsNone",
        "non_private_audits_rerun/purchase_no_defense_non_private/purchase_mlp_epsNone"
    ]

    for experiment in experiments:
        plot_fdp_curves(f"{experiment}/losses_in.npy", 
                        f"{experiment}/losses_out.npy",
                        use_holdout=True)