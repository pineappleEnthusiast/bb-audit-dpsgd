# read_pt.py
import torch
from pprint import pprint

path = "checkpoints/cifar10_cnn_gamma0.5_alpha0.001.pt"
ckpt = torch.load(path, map_location="cpu")

print("=== TOP LEVEL ===")
print("Type:", type(ckpt))
print("Keys:", list(ckpt.keys()))

print("\n=== FULL CHECKPOINT (EXCEPT model_state_dict) ===")
for k, v in ckpt.items():
    if k == "model_state_dict":
        continue
    print(f"\n--- {k} ---")
    pprint(v)

print("\n=== model_state_dict SUMMARY ===")
sd = ckpt["model_state_dict"]
print("num tensors:", len(sd))
for name, t in sd.items():
    print(f"{name:60s} shape={tuple(t.shape)} dtype={t.dtype}")
