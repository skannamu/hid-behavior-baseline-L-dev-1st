import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import SELECTED_FEATURES
from src.classifier_dataset import HIDClassifierDataset
from src.classifier_model import LSTMSequenceClassifier


NORMAL_DATA_PATH = "data/processed"
HUMAN_MIMIC_DATA_PATH = "data/attack/HumanMimic"
STRONG_HUMAN_MIMIC_DATA_PATH = "data/attack/StrongHumanMimic"

SCALER_PATH = "checkpoints/scaler.pkl"
CHECKPOINT_DIR = "checkpoints"
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "lstm_sequence_classifier.pt")

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.2


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
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    input_dim = len(SELECTED_FEATURES)

    model = LSTMSequenceClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
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

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n===== TRAIN CLASSIFIER START =====")
    print(f"Train samples : {train_size}")
    print(f"Valid samples : {valid_size}")
    print(f"Input dim     : {input_dim}")
    print(f"Hidden dim    : {HIDDEN_DIM}")
    print(f"Normal count  : {normal_count}")
    print(f"Attack count  : {attack_count}")
    print(f"Pos weight    : {pos_weight.item():.4f}")
    print("==================================\n")

    best_valid_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for batch_x, batch_y, _, _ in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                valid_loss += loss.item() * batch_x.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                valid_correct += (preds == batch_y).sum().item()
                valid_total += batch_y.size(0)

        valid_loss /= len(valid_loader.dataset)
        valid_acc = valid_correct / valid_total

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Train Acc: {train_acc:.4f} "
            f"Valid Loss: {valid_loss:.6f} "
            f"Valid Acc: {valid_acc:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": HIDDEN_DIM,
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
                },
                SAVE_PATH,
            )

    print("\n===== TRAIN CLASSIFIER DONE =====")
    print(f"Best valid loss: {best_valid_loss:.6f}")
    print(f"Saved model to : {SAVE_PATH}")


if __name__ == "__main__":
    main()
