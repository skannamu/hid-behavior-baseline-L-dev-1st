import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import HIDDataset
from src.model import LSTMAutoencoder


NORMAL_DATA_PATH = "data/processed"
ATTACK_DATA_PATH = "data/attack/FeatureGuideHumanMimic"

MODEL_PATH = "checkpoints/lstm_autoencoder.pt"
SCALER_PATH = "checkpoints/scaler.pkl"

RESULT_DIR = "results"
RESULT_CSV = os.path.join(RESULT_DIR, "reconstruction_errors_all.csv")
SUMMARY_CSV = os.path.join(RESULT_DIR, "reconstruction_summary.csv")

BATCH_SIZE = 256


def load_model(device):
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model = LSTMAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def evaluate_dataset(data_path, label, model, device):
    dataset = HIDDataset(
        data_path,
        scaler_path=SCALER_PATH,
        fit_scaler=False,
        label=label,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    mse = nn.MSELoss(reduction="none")

    all_errors = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            reconstructed = model(batch)
            loss = mse(reconstructed, batch)

            sample_errors = loss.mean(dim=(1, 2))

            all_errors.extend(
                sample_errors.cpu().numpy().tolist()
            )

    result_df = pd.DataFrame({
        "global_index": list(range(len(all_errors))),
        "label": label,
        "source_file": dataset.df["source_file"].tolist(),
        "reconstruction_error": all_errors,
    })

    return result_df


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, checkpoint = load_model(device)

    print("\n===== MODEL INFO =====")
    print(f"Model path : {MODEL_PATH}")
    print(f"Scaler path: {SCALER_PATH}")
    print(f"Train data : {checkpoint.get('train_data_path', checkpoint.get('train_csv_path', 'unknown'))}")
    print("======================\n")

    normal_df = evaluate_dataset(
        NORMAL_DATA_PATH,
        label="normal",
        model=model,
        device=device,
    )

    attack_df = evaluate_dataset(
        ATTACK_DATA_PATH,
        label="attack",
        model=model,
        device=device,
    )

    result_df = pd.concat(
        [normal_df, attack_df],
        ignore_index=True,
    )

    result_df.to_csv(
        RESULT_CSV,
        index=False,
        encoding="utf-8",
    )

    summary_df = (
        result_df
        .groupby("label")["reconstruction_error"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8",
    )

    print("\n===== RECONSTRUCTION ERROR SUMMARY =====")
    print(summary_df)
    print("========================================\n")

    print(f"Saved errors  to: {RESULT_CSV}")
    print(f"Saved summary to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()