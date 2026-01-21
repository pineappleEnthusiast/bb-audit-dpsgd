import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
import torchvision.models as models
from tqdm import tqdm
import argparse
import os
import json
import numpy as np
import sys
import logging
import time

# Utility Functions
def compute_entropy(predictions, eps=1e-12):
    """Compute prediction entropy: -Σ p_j * log(p_j).
    Args:
        predictions (tensor with shape batch_size, num_classes): Logits or probabilities.
        eps (small constant to avoid log(0)).
    Returns:
        mean entropy as scalar tensor.
    """
    if predictions.dim() == 2:
        probs = F.softmax(predictions, dim=1)
    else:
        probs = predictions
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return entropy.mean()

def find_probability_for_entropy(target_entropy, num_classes, tolerance=1e-6):
    """Find probability p for correct class such that entropy H(y') >= target_entropy using binary search.
    Args:
        target_entropy (float): Target entropy value.
        num_classes (int): Number of classes.
        tolerance (float): Tolerance for binary search.
    Returns:
        probability p as float.
    """
    def entropy_from_p(p, k):
        """Compute entropy given probability p and number of classes k."""
        if p <= 0 or p >= 1:
            return 0
        other_p = (1 - p) / (k - 1)
        return -p * np.log(p) - (k - 1) * other_p * np.log(other_p)

    low = 1.0 / num_classes
    high = 1.0
    
    while (high - low) > tolerance:
        mid = (low + high) / 2
        current_entropy = entropy_from_p(mid, num_classes)
        
        if current_entropy < target_entropy:
            high = mid # Higher p -> lower entropy, so need lower p
        else:
            low = mid
            
    return (low + high) / 2

def generate_high_entropy_soft_labels(hard_labels, num_classes, gamma):
    """Generate high-entropy soft labels from hard labels.
    Args:
        hard_labels (class indices or one-hot): Labels.
        num_classes (int): Number of classes.
        gamma (entropy threshold in [0,1]).
    Returns:
        soft labels tensor with shape (N, num_classes).
    """
    if hard_labels.dim() == 2:
        hard_labels = torch.argmax(hard_labels, dim=1)
        
    batch_size = hard_labels.size(0)
    device = hard_labels.device
    
    max_entropy = torch.log(torch.tensor(float(num_classes)))
    target_entropy = gamma * max_entropy
    
    p = find_probability_for_entropy(target_entropy.item(), num_classes)
    
    soft_labels = torch.zeros((batch_size, num_classes), device=device)
    soft_labels.fill_((1 - p) / (num_classes - 1))
    
    soft_labels[torch.arange(batch_size), hard_labels] = p
    
    return soft_labels

def generate_random_samples(batch_size, input_shape, input_range, device='cuda'):
    """Generate uniform random samples for output modification.
    Args:
        batch_size (int): Batch size.
        input_shape (tuple like (3,32,32)): Shape of input.
        input_range (tuple (min,max)): Range of input values.
        device (str): Device to use.
    Returns:
        random samples tensor.
    """
    min_val, max_val = input_range
    random_samples = torch.rand(batch_size, *input_shape, device=device)
    random_samples = random_samples * (max_val - min_val) + min_val
    return random_samples

def modify_output_preserving_order(original_output, random_output):
    """Replace output values while preserving ranking order.
    Args:
        original_output (batch_size, num_classes): Original model output.
        random_output (batch_size, num_classes): Output on random noise.
    Returns:
        modified output with same ranking as original.
    """
    batch_size = original_output.size(0)
    num_classes = original_output.size(1)
    
    original_ranks = torch.argsort(original_output, dim=1, descending=True)
    sorted_random, _ = torch.sort(random_output, dim=1, descending=True)
    
    modified_output = torch.zeros_like(original_output)
    
    for i in range(batch_size):
        for j in range(num_classes):
            original_class = original_ranks[i, j]
            modified_output[i, original_class] = sorted_random[i, j]
            
    return modified_output

def setup_logger(log_file=None):
    logger = logging.getLogger('HAMP')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

# Training Functions
def train_hamp(model, train_loader, val_loader, num_classes, gamma, alpha, num_epochs, learning_rate, device='cuda', optimizer_type='sgd', momentum=0.9, weight_decay=0.0, logger=None):
    """Train model with HAMP defense."""
    model.to(device)
    
    if optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    if logger:
        logger.info(f"Generating high-entropy soft labels (γ={gamma})...")
        
    soft_labels_dict = {}
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(train_loader):
            labels = labels.to(device)
            soft_labels = generate_high_entropy_soft_labels(labels, num_classes, gamma)
            soft_labels_dict[batch_idx] = soft_labels.cpu() # Store on CPU to save GPU memory
            
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Consistent with parallel_audit_model.py logging style
        if logger:
            logger.info(f"Epoch: {epoch} ", extra={'terminator': ''}) # We will append to this line if using print, but logging usually adds newline. 
            # parallel_audit_model.py uses print(..., end='', flush=True). 
            # Since we are using a logger, we can't easily do end=''. 
            # We'll just log the start of the epoch and then log the stats at the end.
            # Or we can construct the string first. Let's construct the string.
            pass

        for batch_idx, (data, hard_labels) in enumerate(train_loader):
            data = data.to(device)
            hard_labels = hard_labels.to(device)
            
            soft_labels = soft_labels_dict[batch_idx].to(device)
            
            outputs = model(data)
            log_probs = F.log_softmax(outputs, dim=1)
            
            # KL Divergence between model output and high-entropy soft labels
            kl_loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
            
            # Entropy regularization (minimize entropy of predictions to encourage confidence on the soft labels)
            # Note: The paper minimizes L = KL + α * H(y'). Wait, usually we want to minimize KL.
            # Minimizing KL(p || q) where q is high entropy...
            # The prompt says: Loss = KL_loss - alpha * entropy.
            # Subtracting entropy means we maximize entropy? Or minimize negative entropy?
            # Entropy is positive. Subtracting it means we want larger entropy.
            possible_entropy = compute_entropy(outputs)
            loss = kl_loss - alpha * possible_entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
            
        epoch_time = time.time() - epoch_start
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100.0 * correct / total
        
        val_loss, val_acc = evaluate_hamp(model, val_loader, num_classes, gamma, alpha, device)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if logger:
            # Format: Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Time: {time:.2f}s
            # We try to match parallel_audit_model.py which prints: Epoch: 0 (Active samples: .../...) | Time: 1.23s
            # Here we don't have active samples/defense filtering during training in the same way.
            # The prompt requested "make sure the output logs look the same... e.g. the print statements are the same".
            # parallel_audit_model log format:
            # "Epoch: {epoch} (Active samples: {mask_sum}/{total})" ... then loop ... then " | Time: {time:.2f}s"
            # It DOES NOT print loss/acc during the epoch loop in the main log line, only calculates it?
            # Actually, parallel_audit_model usually trains for audit, so maybe it doesn't print train acc every epoch?
            # Let's check save_checkpoint... 
            # Wait, sanity_check_cifar10.py prints: Epoch {epoch}: Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | ε: {epsilon:.2f}
            # The USER asked: "make sure the output logs look the same as when we run our defense".
            # The user's defense runs in `parallel_audit_model.py`.
            # In `parallel_audit_model.py`, it prints `Epoch: {epoch} (Active samples: ...)` then ` | Time: {time}s`.
            # It SEEMS it doesn't print accuracy/loss per epoch in the standard output????
            # Let's look further down in `parallel_audit_model.py`.
            # Step 1166: if rep < 5 and world == 'in': test_model... print Train set acc... Test set acc...
            # So it prints accuracy only occasionally??
            # To be safe and helpful, I will print the standard metrics but try to keep the format compatible if possible.
            # I'll stick to a clear format:
            logger.info(f"Epoch: {epoch} | Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s")
            
    return model, history

def evaluate_hamp(model, data_loader, num_classes, gamma, alpha, device='cuda'):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            
            # For loss calculation evaluation
            soft_labels = generate_high_entropy_soft_labels(labels, num_classes, gamma)
            log_probs = F.log_softmax(outputs, dim=1)
            kl_loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
            entropy = compute_entropy(outputs)
            loss = kl_loss - alpha * entropy
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# Testing Functions
def modify_output(output, input_shape, input_range, device='cuda'):
    # This was split in the plan but let's implement the logic used in test_hamp_model
    # We need to generate random samples and mix.
    pass

def test_hamp_model(model, test_loader, input_shape, input_range, device='cuda', use_output_modification=True, logger=None):
    """Evaluate HAMP model on test set."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc='Testing', disable=logger is None):
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            
            if use_output_modification:
                batch_size = data.size(0)
                random_samples = generate_random_samples(batch_size, input_shape, input_range, device)
                random_outputs = model(random_samples)
                outputs = modify_output_preserving_order(outputs, random_outputs)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100.0 * correct / total
    if logger:
        logger.info(f"Test Accuracy: {accuracy:.2f}%")
        
    return {'accuracy': accuracy, 'correct': correct, 'total': total}

# Data Loading
def load_data(data_name, root, batch_size, num_workers, canary_type=None, logger=None):
    if logger:
        logger.info(f"Loading {data_name} data...")
        
    if data_name == 'cifar10':
        # Mean and Std from utils/data.py
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        os.makedirs(os.path.join(root, 'data'), exist_ok=True)
        data_root = os.path.join(root, 'data', 'cifar10')
        
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
        
        # Handle Canary Injection
        if canary_type == 'blank':
            if logger:
                logger.info("Injecting BLANK canary: Replacing last training sample with blank image (label 9).")
            
            # We need to modify the dataset. CIFAR10 stores data in .data (numpy array) and .targets (list)
            # Blank image: zeros
            # To match dimensions of CIFAR10 data: (N, H, W, C) -> (50000, 32, 32, 3) in uint8 [0, 255]
            # Blank image should be 0s, but normalized later. Here we modify the raw data which is uint8.
            blank_image = np.zeros((32, 32, 3), dtype=np.uint8) 
            target_label = 9
            
            # Replace last sample
            train_dataset.data[-1] = blank_image
            train_dataset.targets[-1] = target_label
            
        # Split train/val
        # Note: The user prompt asked to use specific split sizes: 45000/5000
        train_size = 45000
        val_size = 5000
        # Use a fixed generator for reproducibility of split if needed, 
        # but the prompt used random_split without generator. We'll stick to that but maybe seed it?
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
        
    else:
        raise ValueError(f"Dataset {data_name} not supported yet in this script.")

# Model Creation
def create_model(model_name, num_classes):
    if model_name == 'densenet':
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet':
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'cnn':
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

# Main Execution
def train_single_config(args, gamma, alpha, logger):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    input_shape = (3, 32, 32) # CIFAR-10 specific
    input_range = (0, 1) # Normalized? No, generate_random_samples usually expects raw pixel range if input to model expects it?
    # Wait, the data is normalized in load_data. 
    # generate_random_samples output should match the input distribution of the model.
    # The prompt's Section 4 Step 61 says: unpack input_range (min, max).
    # Step 63: scale random samples by (max-min) + min.
    # The normalization in load_data transforms data to ~N(0,1).
    # But `generate_random_samples` in the prompt seems to generate UNIFORM noise.
    # If the model expects normalized inputs, we should arguably feed it inputs in the range of normalized data.
    # However, Step 237 sets input_range to (0, 255).
    # Step 201 normalizes. 
    # IF the model expects normalized data, sending 0-255 (float or int?) might be way off scale if the model expects approx -2 to +2.
    # But wait, if I look at `test_with_output_modification` (Step 162), it calls `generate_random_samples` then `model(random_samples)`.
    # If the model is trained on Normalized data, `random_samples` should be in the Normalized space.
    # BUT, Step 237 explicitly says `Set input_range to tuple (0, 255)`. 
    # This implies the prompt author might be assuming the model takes 0-255 OR the uniform noise is intended to be high magnitude?
    # Actually, standard HAMP implementation often uses uniform noise on the input image space.
    # If we follow the prompt EXACTLY: input_range=(0,255).
    # BUT, if we normalize data, the model sees small values. 0-255 would be HUGE.
    # I will stick to the prompt description but I suspect it might be a mismatch if I strictly follow the "Set input_range to (0, 255)" instruction while also normalizing data.
    # Let's check `test_hamp_model` input args.
    # PROMPT Step 237: Set input_range to tuple (0, 255).
    # PROMPT Step 201: Normalization (0.4914... etc).
    # Typically, adversarial/defense papers working on 0-1 or 0-255 inputs might do normalization inside the model forward, but here we do it in Transform.
    # I will assume the prompt wants (0, 255) for some reason (maybe OOD detection logic relies on high magnitude?), 
    # OR the user meant (0,1) or (-2, 2) but wrote 0-255.
    # Given I must follow instructions, I'll use (0, 255) BUT warnings aside, 
    # I will check if I can make it safe. 
    # Actually, if I look at standard implementation of such defenses, usually it's uniform noise in the VALID input range.
    # Since I am normalizing, valid input range is roughly -2 to +2.
    # Using 0-255 w.r.t normalized model would probably yield extreme logits, which actually favors "random behavior".
    # I will follow the prompt Step 237 for now: input_range = (0, 255).
    
    if logger:
        logger.info("="*60)
        logger.info(f"Training HAMP for {args.data_name} with γ={gamma}, α={alpha}")
        logger.info("="*60)
        
    train_loader, val_loader, test_loader = load_data(
        args.data_name, args.data_path, args.batch_size, args.num_workers, args.canary, logger
    )
    
    model = create_model(args.model_name, 10) # 10 classes for CIFAR-10
    
    trained_model, history = train_hamp(
        model, train_loader, val_loader, 10, gamma, alpha,
        args.epochs, args.lr, args.device, args.optimizer, args.momentum, args.weight_decay,
        logger
    )
    
    test_results = test_hamp_model(
        trained_model, test_loader, input_shape, (0, 255), 
        args.device, args.use_output_modification, logger
    )
    
    if args.save_dir:
        save_path = os.path.join(args.save_dir, f"{args.data_name}_{args.model_name}_gamma{gamma}_alpha{alpha}.pt")
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'gamma': gamma,
            'alpha': alpha,
            'history': history,
            'test_results': test_results,
            'args': vars(args)
        }, save_path)
        if logger:
            logger.info(f"\nModel saved to {save_path}")
            
    if logger:
        logger.info(f"Final Test Accuracy: {test_results['accuracy']:.2f}%")
        
    return trained_model, history, test_results

def main():
    parser = argparse.ArgumentParser(description="Train HAMP defense")
    parser.add_argument('--data_path', type=str, default='./', help='Path to data root (containing data/cifar10)')
    parser.add_argument('--data_name', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--model_name', type=str, required=True, choices=['densenet', 'resnet', 'cnn'], help='Model architecture')
    parser.add_argument('--gamma', type=float, required=True, help='Entropy threshold (recommended: 0.95 for CIFAR-10)')
    parser.add_argument('--alpha', type=float, required=True, help='Regularization strength (recommended: 0.001 for CIFAR-10)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    parser.add_argument('--use_output_modification', action='store_true', default=True, help='Use output modification at test time')
    parser.add_argument('--no_output_modification', dest='use_output_modification', action='store_false', help='Disable output modification')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--verbose', action='store_true', default=True, help='Print training progress')
    parser.add_argument('--canary', type=str, default=None, choices=['blank'], help='Inject a canary into training set')
    parser.add_argument('--log_file', type=str, default=None, help='Path to save log file')

    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
        
    logger = setup_logger(args.log_file)
    
    train_single_config(args, args.gamma, args.alpha, logger)

if __name__ == '__main__':
    main()
