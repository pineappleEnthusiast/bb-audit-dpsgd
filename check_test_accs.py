#!/usr/bin/env python
"""
Script to load and display test set accuracies from saved checkpoints.
Usage:
    python check_test_accs.py <checkpoint_folder>
    python check_test_accs.py  # scans common folders
"""

import os
import sys
import numpy as np
from pathlib import Path


def load_test_accs(folder):
    """Load test set accuracies from a checkpoint folder."""
    test_accs_path = os.path.join(folder, 'test_set_accs.npy')
    train_accs_path = os.path.join(folder, 'train_set_accs.npy')
    
    results = {}
    
    if os.path.exists(test_accs_path):
        results['test_accs'] = np.load(test_accs_path)
    
    if os.path.exists(train_accs_path):
        results['train_accs'] = np.load(train_accs_path)
    
    # Also check for outputs to see how many reps completed
    for world in ['in', 'out']:
        outputs_path = os.path.join(folder, f'outputs_{world}.npy')
        if os.path.exists(outputs_path):
            outputs = np.load(outputs_path)
            results[f'n_reps_{world}'] = len(outputs)
    
    return results


def print_accs(folder, results):
    """Pretty print the accuracies."""
    print(f"\n{'='*60}")
    print(f"Checkpoint: {folder}")
    print('='*60)
    
    if 'test_accs' in results:
        test_accs = results['test_accs']
        print(f"\nTest Set Accuracies ({len(test_accs)} reps):")
        for i, acc in enumerate(test_accs):
            print(f"  Rep {i}: {acc:.4f} ({acc*100:.2f}%)")
        if len(test_accs) > 0:
            print(f"\n  Mean: {np.mean(test_accs):.4f} ({np.mean(test_accs)*100:.2f}%)")
            print(f"  Std:  {np.std(test_accs):.4f}")
            print(f"  Min:  {np.min(test_accs):.4f}")
            print(f"  Max:  {np.max(test_accs):.4f}")
    else:
        print("\nNo test_set_accs.npy found")
    
    if 'train_accs' in results:
        train_accs = results['train_accs']
        print(f"\nTrain Set Accuracies ({len(train_accs)} reps):")
        for i, acc in enumerate(train_accs):
            print(f"  Rep {i}: {acc:.4f} ({acc*100:.2f}%)")
        if len(train_accs) > 0:
            print(f"\n  Mean: {np.mean(train_accs):.4f} ({np.mean(train_accs)*100:.2f}%)")
            print(f"  Std:  {np.std(train_accs):.4f}")
    
    # Print progress info
    for world in ['in', 'out']:
        if f'n_reps_{world}' in results:
            print(f"\nCompleted reps ({world}): {results[f'n_reps_{world}']}")


def find_checkpoint_folders(base_dir='.'):
    """Find all folders containing checkpoint files."""
    folders = []
    for root, dirs, files in os.walk(base_dir):
        if 'test_set_accs.npy' in files or 'outputs_in.npy' in files or 'outputs_out.npy' in files:
            folders.append(root)
    return folders


def main():
    if len(sys.argv) > 1:
        # Specific folder provided
        folder = sys.argv[1]
        if os.path.isdir(folder):
            results = load_test_accs(folder)
            print_accs(folder, results)
        else:
            print(f"Error: {folder} is not a valid directory")
            sys.exit(1)
    else:
        # Scan for checkpoint folders
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Common checkpoint locations
        search_dirs = [
            base_dir,
            os.path.join(base_dir, 'results'),
            os.path.join(base_dir, 'defense'),
            os.path.join(base_dir, 'no_defense'),
            os.path.join(base_dir, 'debug'),
            os.path.join(base_dir, 'local'),
        ]
        
        found_folders = set()
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for folder in find_checkpoint_folders(search_dir):
                    found_folders.add(folder)
        
        if not found_folders:
            print("No checkpoint folders found. Specify a folder path as argument.")
            print("Usage: python check_test_accs.py <checkpoint_folder>")
            sys.exit(1)
        
        print(f"Found {len(found_folders)} checkpoint folder(s)")
        
        for folder in sorted(found_folders):
            results = load_test_accs(folder)
            if results:
                print_accs(folder, results)


if __name__ == '__main__':
    main()
