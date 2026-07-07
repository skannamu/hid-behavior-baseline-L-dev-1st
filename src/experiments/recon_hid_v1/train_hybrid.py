import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import SELECTED_FEATURES
from src.classifier_dataset import HIDClassifierDataset
from src.hybrid_model import ReConHID


NORMAL_DATA_PATH = "data/processed"
HUMAN_MIMIC_DATA_PATH = "data/attack/HumanMimic"
STRONG_HUMAN_MIMIC_DATA_PATH = "data/attack/StrongHumanMimic"

SCALER_PATH = "checkpoints/scaler.pkl"

CHECKPOINT_DIR = "checkpoints"
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "recon_hid_v1.pt")

BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 0.001

HIDDEN_DIM = 64
LATENT_DIM = 32
NUM_LAYERS = 1
DROPOUT = 0.2

CLASSIFICATION_LOSS_WEIGHT = 1.0
RECONSTRUCTION_LOSS_WEIGHT = 1.0


def evaluate(model, loader, recon_loss_fn, cls_loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_cls_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            reconstructed, logits, _ = model(batch_x)

            recon_loss = recon_loss_fn(reconstructed, batch_x)
            cls_loss = cls_loss_fn(logits, batch_y)

            loss = (
                RECONSTRUCTION_LOSS_WEIGHT * recon_loss
                + CLASSIFICATION_LOSS_WEIGHT * cls_loss
            )

            total_loss += loss.item() * batch_x.size(0)
            total_recon_loss += recon_loss.item() * batch_x.size(0)
            total_cls_loss += cls_loss.item() * batch_x.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return {
        "loss": total_loss / total,
        "recon_loss": total_recon_loss / total,
        "cls_loss": total_cls_loss / total,
        "acc": correct / total,
    }


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = [
        {
            "group": "normal",
            "path": NORMAL_DATA_PATH,
            "label": 0,
        },
        {
            "group": "HumanMimic",
            "path": HUMAN_MIMIC_DATA_PATH,
            "label": 1,
        },
        {
            "group": "StrongHumanMimic",
            "path": STRONG_HUMAN_MIMIC_DATA_PATH,
            "label": 1,
        },
    ]

    dataset = HIDClassifierDataset(
        samples=samples,
        scaler_path=SCALER_PATH,
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

    model = ReConHID(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    y_all = dataset.y
    normal_count = (y_all == 0).sum().item()
    attack_count = (y_all == 1).sum().item()

    pos_weight = torch.tensor(
        [normal_count / max(attack_count, 1)],
        dtype=torch.float32,
        device=device,
    )

    recon_loss_fn = nn.MSELoss()
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n===== TRAIN ReCon-HID v1 START =====")
    print(f"Train samples : {train_size}")
    print(f"Valid samples : {valid_size}")
    print(f"Input dim     : {input_dim}")
    print(f"Hidden dim    : {HIDDEN_DIM}")
    print(f"Latent dim    : {LATENT_DIM}")
    print(f"Normal count  : {normal_count}")
    print(f"Attack count  : {attack_count}")
    print(f"Pos weight    : {pos_weight.item():.4f}")
    print(f"Recon weight  : {RECONSTRUCTION_LOSS_WEIGHT}")
    print(f"Cls weight    : {CLASSIFICATION_LOSS_WEIGHT}")
    print("====================================\n")

    best_valid_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_cls_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            reconstructed, logits, _ = model(batch_x)

            recon_loss = recon_loss_fn(reconstructed, batch_x)
            cls_loss = cls_loss_fn(logits, batch_y)

            loss = (
                RECONSTRUCTION_LOSS_WEIGHT * recon_loss
                + CLASSIFICATION_LOSS_WEIGHT * cls_loss
            )

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item() * batch_x.size(0)
            train_recon_loss += recon_loss.item() * batch_x.size(0)
            train_cls_loss += cls_loss.item() * batch_x.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_metrics = {
            "loss": train_total_loss / train_total,
            "recon_loss": train_recon_loss / train_total,
            "cls_loss": train_cls_loss / train_total,
            "acc": train_correct / train_total,
        }

        valid_metrics = evaluate(
            model,
            valid_loader,
            recon_loss_fn,
            cls_loss_fn,
            device,
        )

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_metrics['loss']:.6f} "
            f"Train Recon: {train_metrics['recon_loss']:.6f} "
            f"Train Cls: {train_metrics['cls_loss']:.6f} "
            f"Train Acc: {train_metrics['acc']:.4f} "
            f"Valid Loss: {valid_metrics['loss']:.6f} "
            f"Valid Recon: {valid_metrics['recon_loss']:.6f} "
            f"Valid Cls: {valid_metrics['cls_loss']:.6f} "
            f"Valid Acc: {valid_metrics['acc']:.4f}"
        )

        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "latent_dim": LATENT_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "selected_features": SELECTED_FEATURES,
                    "scaler_path": SCALER_PATH,
                    "train_groups": [
                        "normal",
                        "HumanMimic",
                        "StrongHumanMimic",
                    ],
                    "excluded_from_train": [
                        "FeatureGuideHumanMimic",
                    ],
                    "reconstruction_loss_weight": RECONSTRUCTION_LOSS_WEIGHT,
                    "classification_loss_weight": CLASSIFICATION_LOSS_WEIGHT,
                },
                SAVE_PATH,
            )

    print("\n===== TRAIN ReCon-HID v1 DONE =====")
    print(f"Best valid loss: {best_valid_loss:.6f}")
    print(f"Saved model to : {SAVE_PATH}")


if __name__ == "__main__":
    main()
