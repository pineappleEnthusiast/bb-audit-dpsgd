# import torch
# import torch.nn as nn

# class LSTM(nn.Module):
#     def __init__(self, vocab_size, out_dim=None, embed_dim=128, hidden_dim=256, num_layers=1, dropout_rate=0.1):
#         super().__init__()

#         if out_dim is None:
#             out_dim = vocab_size

#         self.embedding = nn.Embedding(vocab_size, embed_dim)

#         self.lstm = nn.LSTM(
#             embed_dim, hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout_rate if num_layers > 1 else 0
#         )

#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(hidden_dim, out_dim)

#     def forward(self, x):
#         x = self.embedding(x)
#         output, (h_n, c_n) = self.lstm(x)

#         # output: [batch_size, seq_len, hidden_dim]
#         # Take the last time step's output for each sequence
#         last_output = output[:, -1, :]  # [batch_size, hidden_dim]

#         last_output = self.dropout(last_output)
#         out = self.fc(last_output)      # [batch_size, out_dim]
#         return out


# lstm.py
import torch
import torch.nn as nn
from opacus.layers import DPLSTM   # <-- add this

class LSTM(nn.Module):
    def __init__(self, vocab_size, out_dim=None, embed_dim=128, hidden_dim=256, num_layers=1, dropout_rate=0.1):
        super().__init__()

        if out_dim is None:
            out_dim = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Use DPLSTM instead of nn.LSTM
        # (Opacus DPLSTM currently supports a single layer; keep num_layers=1)
        self.lstm = DPLSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        out = self.fc(last_output)
        return out
