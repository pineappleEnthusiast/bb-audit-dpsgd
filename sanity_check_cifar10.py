import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 3125
EPOCHS = 100
LR = 3.0
MAX_GRAD_NORM = 1.0
EPSILON = 10.0
DELTA = 1e-5

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        act_func = nn.Tanh
        feature_layer_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']
        feature_layers = []

        c = 3
        for v in feature_layer_config:
            if v == 'M':
                feature_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)
                feature_layers += [conv2d, act_func()]
                c = v
        self.features = nn.Sequential(*feature_layers)

        self.dropout = nn.Dropout(0.0)
        self.classifier = nn.Sequential(
            nn.Linear(c * 4 * 4, 128), act_func(), nn.Linear(128, 10)
        )
        self.embeddings = None
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        self.embeddings = x.clone().detach()
        x = self.classifier(x)
        return x

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,  # Shuffle order mini-batching
    num_workers=2
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2
)

# Calculate noise multiplier for target epsilon
sample_rate = BATCH_SIZE / len(train_dataset)
noise_multiplier = get_noise_multiplier(
    target_epsilon=EPSILON,
    target_delta=DELTA,
    sample_rate=sample_rate,
    epochs=EPOCHS,
    accountant="rdp"
)

print(f"\nPrivacy Parameters:")
print(f"Target ε: {EPSILON}")
print(f"Target δ: {DELTA}")
print(f"Calculated noise multiplier σ: {noise_multiplier:.4f}")
print(f"Sample rate: {sample_rate:.4f}")
print(f"Steps per epoch: {len(train_loader)}")
print(f"Total steps: {len(train_loader) * EPOCHS}\n")

# Initialize model
model = SimpleCNN().to(device)

# Make model compatible with Opacus
model = ModuleValidator.fix(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# Attach privacy engine
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=noise_multiplier,
    max_grad_norm=MAX_GRAD_NORM,
)

print("Privacy engine attached successfully!\n")

# Training function
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    
    # Get current privacy spent
    epsilon = privacy_engine.get_epsilon(DELTA)
    
    print(f"Epoch {epoch}: Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | ε: {epsilon:.2f}")
    
    return accuracy, epsilon

# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f"Test: Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%\n")
    
    return accuracy

# Training loop
print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    train_acc, current_epsilon = train(epoch)
    
    # Test every 10 epochs
    if epoch % 10 == 0:
        test_acc = test()

# Final test
print("\nFinal Evaluation:")
final_test_acc = test()
final_epsilon = privacy_engine.get_epsilon(DELTA)
print(f"Final ε spent: {final_epsilon:.2f} (target was {EPSILON})")
print(f"Final test accuracy: {final_test_acc:.2f}%")