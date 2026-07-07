import torch
import torch.nn as nn
import torch.nn.functional as F


class ReConHIDv2(nn.Module):
    """
    ReCon-HID v2
    Reconstruction + Classification + Contrastive + Prototype Memory

    Input:
        x: [batch, seq_len, input_dim]

    Output:
        reconstructed: [batch, seq_len, input_dim]
        class_logit: [batch]
        latent: [batch, latent_dim]
        latent_norm: [batch, latent_dim]
        proto_logits: [batch, 2]
        proto_distances: [batch, 2]
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        latent_dim=32,
        num_layers=1,
        dropout=0.2,
        proto_temperature=0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.proto_temperature = proto_temperature

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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

        # Prototype Memory
        # index 0: normal prototype
        # index 1: attack prototype
        self.prototypes = nn.Parameter(
            torch.randn(2, latent_dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]

        latent = self.latent_proj(h)
        latent_norm = F.normalize(latent, dim=1)

        decoder_seed = self.decoder_input(latent)
        decoder_input = decoder_seed.unsqueeze(1).repeat(1, seq_len, 1)

        decoded, _ = self.decoder(decoder_input)
        reconstructed = self.reconstruction_head(decoded)

        class_logit = self.classifier_head(latent).squeeze(1)

        proto_norm = F.normalize(self.prototypes, dim=1)

        proto_distances = torch.cdist(
            latent_norm,
            proto_norm,
            p=2,
        ).pow(2)

        proto_logits = -proto_distances / self.proto_temperature

        return (
            reconstructed,
            class_logit,
            latent,
            latent_norm,
            proto_logits,
            proto_distances,
        )
