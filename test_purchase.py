from utils.data import load_data

X, y, out_dim = load_data('purchase', None, split='train')

print("X shape:", X.shape)
print("y shape:", y.shape)
print("num classes (out_dim):", out_dim)
print("First 10 labels:", y[:10])
print(y[-1])