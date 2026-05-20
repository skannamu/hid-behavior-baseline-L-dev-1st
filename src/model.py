import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        _, (hidden, _) = self.encoder_lstm(x)

        last_hidden = hidden[-1]

        latent = self.to_latent(last_hidden)

        decoder_init = self.from_latent(latent)

        decoder_input = decoder_init.unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder_lstm(decoder_input)

        reconstructed = self.output_layer(decoded)

        return reconstructed