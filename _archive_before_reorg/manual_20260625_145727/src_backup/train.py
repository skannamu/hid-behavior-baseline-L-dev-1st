import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import HIDDataset, SELECTED_FEATURES
from src.model import LSTMAutoencoder


TRAIN_DATA_PATH = "data/processed"

CHECKPOINT_DIR = "checkpoints"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "lstm_autoencoder.pt")
SCALER_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001

HIDDEN_DIM = 64
LATENT_DIM = 16
NUM_LAYERS = 1


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = HIDDataset(
        TRAIN_DATA_PATH,
        scaler_path=SCALER_SAVE_PATH,
        fit_scaler=True,
        save_scaler=True,
        label="normal",
    )

    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_dim = len(SELECTED_FEATURES)

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\n===== TRAINING START =====")
    print(f"Train data    : {TRAIN_DATA_PATH}")
    print(f"Train samples : {train_size}")
    print(f"Valid samples : {valid_size}")
    print(f"Input dim     : {input_dim}")
    print(f"Hidden dim    : {HIDDEN_DIM}")
    print(f"Latent dim    : {LATENT_DIM}")
    print(f"Scaler path   : {SCALER_SAVE_PATH}")
    print("==========================\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)

                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)

                valid_loss += loss.item() * batch.size(0)

        valid_loss /= len(valid_loader.dataset)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Valid Loss: {valid_loss:.6f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "num_layers": NUM_LAYERS,
            "selected_features": SELECTED_FEATURES,
            "scaler_path": SCALER_SAVE_PATH,
            "train_data_path": TRAIN_DATA_PATH,
        },
        MODEL_SAVE_PATH,
    )

    print("\n===== TRAINING DONE =====")
    print(f"Saved model  to: {MODEL_SAVE_PATH}")
    print(f"Saved scaler to: {SCALER_SAVE_PATH}")


if __name__ == "__main__":
    main()