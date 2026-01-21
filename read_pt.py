python - << 'EOF'
import torch

path = "checkpoints/cifar10_cnn_gamma0.5_alpha0.001.pt"
ckpt = torch.load(path, map_location="cpu")

print("Loaded:", path)
print("Type:", type(ckpt))

if isinstance(ckpt, dict):
    print("Top-level keys:", list(ckpt.keys()))
EOF
