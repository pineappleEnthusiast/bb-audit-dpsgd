import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, out_dim=None, embed_dim=128, hidden_dim=256, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.embeddings = None

        if out_dim is None:
            out_dim = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)

        # use last hidden state
        h_last = h_n[-1]
        h_last = self.dropout(h_last)
        self.embeddings = h_last.clone().detach()

        out = self.fc(h_last)
        return out