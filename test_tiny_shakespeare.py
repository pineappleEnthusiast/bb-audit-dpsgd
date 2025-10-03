from utils.data import load_data

X, y, out_dim = load_data('tiny_shakespeare', None, split='train')

print("X shape:", X.shape)
print("y shape:", y.shape)
print("num classes (out_dim):", out_dim)

print("\nFirst input sequence (as IDs):")
print(X[0])

print("\nFirst output sequence (labels, as IDs):")
print(y[0])

print("\nSecond input sequence (as IDs):")
print(X[1])

print("\nSecond output sequence (labels, as IDs):")

print("\nLast input sequence:")
print(X[-1])
print("\nLast output sequence:")
print(y[-1])
