import torch
import torch.nn as nn


class ReConHID(nn.Module):
    """
    ReCon-HID v1
    Reconstruction + Classification Hybrid Detector

    Input:
        x: [batch, seq_len, input_dim]

    Output:
        reconstructed: [batch, seq_len, input_dim]
        logit: [batch]
        latent: [batch, hidden_dim]
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        latent_dim=32,
        num_layers=1,
        dropout=0.2,
    ):find data/attack -maxdepth 4 -type f | sort | sed -n '1,200p'

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)

        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        _, (h_n, _) = self.encoder(x)

        h = h_n[-1]
        latent = self.latent_proj(h)

        decoder_seed = self.decoder_input(latent)
        decoder_input = decoder_seed.unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder(decoder_input)
        reconstructed = self.reconstruction_head(decoded)

        logit = self.classifier_head(latent).squeeze(1)

        return reconstructed, logit, latent
