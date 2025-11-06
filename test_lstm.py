import torch
import torch.nn as nn
import torch.optim as optim
from utils.data import load_data
from models.lstm import LSTMModel

def main():
    X, y, out_dim = load_data('tiny_shakespeare', None, split='train')
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("num classes (out_dim):", out_dim)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = LSTMModel(vocab_size=out_dim, hidden_dim=128, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            logits = model(batch_X)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch_y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()