"""Quick test to demonstrate soft label generation"""
import torch
import torch.nn.functional as F

def create_soft_labels(y, num_classes, gamma=0.0):
    """Convert hard labels to soft labels with controlled entropy."""
    if gamma == 0.0:
        return F.one_hot(y, num_classes=num_classes).float()
    
    batch_size = y.size(0)
    soft_labels = torch.ones(batch_size, num_classes, device=y.device) * (gamma / num_classes)
    
    ground_truth_prob = 1.0 - gamma + (gamma / num_classes)
    soft_labels.scatter_(1, y.unsqueeze(1), ground_truth_prob)
    
    return soft_labels

# Test with different gamma values
y = torch.tensor([0, 1, 2])  # 3 samples with classes 0, 1, 2
num_classes = 10

print("=" * 60)
print("Soft Label Examples (num_classes=10)")
print("=" * 60)

for gamma in [0.0, 0.5, 0.8, 1.0]:
    soft = create_soft_labels(y, num_classes, gamma)
    ground_truth_prob = 1.0 - gamma + (gamma / num_classes)
    other_prob = gamma / num_classes
    
    print(f"\ngamma = {gamma:.1f}")
    print(f"  Ground-truth prob: {ground_truth_prob:.4f} ({ground_truth_prob*100:.1f}%)")
    print(f"  Other class prob:  {other_prob:.4f} ({other_prob*100:.1f}%)")
    print(f"  Sample 0 (class 0): {soft[0].numpy()}")
    print(f"  Sum check: {soft[0].sum():.6f} (should be 1.0)")
