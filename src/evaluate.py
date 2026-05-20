import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import HIDDataset
from src.model import LSTMAutoencoder


CSV_PATH = "data/processed/window.csv"
MODEL_PATH = "checkpoints/lstm_autoencoder.pt"

RESULT_DIR = "results"
RESULT_CSV = os.path.join(RESULT_DIR, "reconstruction_errors.csv")

BATCH_SIZE = 32


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    dataset = HIDDataset(CSV_PATH)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = LSTMAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mse = nn.MSELoss(reduction="none")

    all_errors = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            reconstructed = model(batch)

            loss = mse(reconstructed, batch)

            # loss shape: (batch, 50, 15)
            # 각 window마다 평균 reconstruction error 계산
            sample_errors = loss.mean(dim=(1, 2))

            all_errors.extend(
                sample_errors.cpu().numpy().tolist()
            )

    result_df = pd.DataFrame({
        "window_index": list(range(len(all_errors))),
        "reconstruction_error": all_errors,
    })

    result_df.to_csv(
        RESULT_CSV,
        index=False,
        encoding="utf-8",
    )

    print("\n===== RECONSTRUCTION ERROR =====")
    print(f"Samples : {len(all_errors)}")
    print(f"Mean    : {result_df['reconstruction_error'].mean():.6f}")
    print(f"Std     : {result_df['reconstruction_error'].std():.6f}")
    print(f"Min     : {result_df['reconstruction_error'].min():.6f}")
    print(f"Max     : {result_df['reconstruction_error'].max():.6f}")
    print(f"Saved   : {RESULT_CSV}")
    print("================================\n")


if __name__ == "__main__":
    main()