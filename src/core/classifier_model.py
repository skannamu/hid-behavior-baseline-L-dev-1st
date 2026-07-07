import torch
import torch.nn as nn


class LSTMSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        logit = self.classifier(h).squeeze(1)
        return logit
