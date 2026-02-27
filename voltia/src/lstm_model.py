import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        output_size=1,
        dropout=0.2,
        bidirectional=True,
    ):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)

        self.norm = nn.LayerNorm(lstm_out_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_out_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)              # (batch, seq_len, lstm_out_size)
        out = out[:, -1, :]                # último paso temporal
        out = self.norm(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out