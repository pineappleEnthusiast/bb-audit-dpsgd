import torch

if torch.cuda.is_available():
    print("torch cuda gpu available")
    print("torch cuda gpu number:", torch.cuda.device_count())
else:
    print("torch cuda gpu not available")
