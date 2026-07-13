import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.dataset import SELECTED_FEATURES
from src.classifier_dataset import HIDClassifierDataset
from src.hybrid_v2_model import ReConHIDv2

NORMAL_DATA_PATH = "data/processed"
HUMAN_MIMIC_DATA_PATH = "data/attack/HumanMimic"
STRONG_HUMAN_MIMIC_DATA_PATH = "data/attack/StrongHumanMimic"

SCALER_PATH = "checkpoints/scaler.pkl"

CHECKPOINT_DIR = "checkpoints"
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "recon_hid_v2.pt")

RESULT_DIR = "results/recon_hid_v2"
TRAIN_HISTORY_CSV = os.path.join(RESULT_DIR, "train_history_recon_hid_v2.csv")

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001

HIDDEN_DIM = 64
LATENT_DIM = 32
NUM_LAYERS = 1
DROPOUT = 0.2
PROTO_TEMPERATURE = 0.2
CONTRASTIVE_TEMPERATURE = 0.1

# v2 핵심:
# reconstruction은 normal behavior manifold를 잡기 위한 보조 역할
# classification / prototype / contrastive를 더 강하게 둔다.
RECONSTRUCTION_LOSS_WEIGHT = 0.3
CLASSIFICATION_LOSS_WEIGHT = 1.0
CONTRASTIVE_LOSS_WEIGHT = 0.2
PROTOTYPE_LOSS_WEIGHT = 0.5


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """
    Supervised contrastive loss.
    같은 label끼리는 가깝게, 다른 label끼리는 멀게 만든다.
    labels: 0 normal, 1 attack
    """
    device = features.device
    batch_size = features.size(0)

    features = F.normalize(features, dim=1)
    labels = labels.view(-1, 1)

    mask = torch.eq(labels, labels.T).float().to(device)

    logits = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = mask.sum(dim=1)
    valid = positive_count > 0

    if valid.sum() == 0:
        return features.sum() * 0.0

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_count + 1e-12)
    loss = -mean_log_prob_pos[valid].mean()

    return loss


def normal_only_reconstruction_loss(reconstructed, target, labels):
    """
    reconstruction loss는 normal sample에만 적용한다.
    attack까지 잘 복원하게 만들면 anomaly detector로서 의미가 약해진다.
    """
    mse = F.mse_loss(
        reconstructed,
        target,
        reduction="none",
    ).mean(dim=(1, 2))

    normal_mask = labels == 0

    if normal_mask.sum() == 0:
        return mse.mean() * 0.0

    return mse[normal_mask].mean()


def evaluate(model, loader, cls_loss_fn, proto_loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_cls_loss = 0.0
    total_contrastive_loss = 0.0
    total_proto_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_long = batch_y.long()

            (
                reconstructed,
                class_logits,
                latent,
                latent_norm,
                proto_logits,
                _,
            ) = model(batch_x)

            recon_loss = normal_only_reconstruction_loss(
                reconstructed,
                batch_x,
                batch_y,
            )

            cls_loss = cls_loss_fn(class_logits, batch_y)
            contrastive_loss = supervised_contrastive_loss(
                latent_norm,
                batch_y_long,
                temperature=CONTRASTIVE_TEMPERATURE,
            )
            proto_loss = proto_loss_fn(proto_logits, batch_y_long)

            loss = (
                RECONSTRUCTION_LOSS_WEIGHT * recon_loss
                + CLASSIFICATION_LOSS_WEIGHT * cls_loss
                + CONTRASTIVE_LOSS_WEIGHT * contrastive_loss
                + PROTOTYPE_LOSS_WEIGHT * proto_loss
            )

            total_loss += loss.item() * batch_x.size(0)
            total_recon_loss += recon_loss.item() * batch_x.size(0)
            total_cls_loss += cls_loss.item() * batch_x.size(0)
            total_contrastive_loss += contrastive_loss.item() * batch_x.size(0)
            total_proto_loss += proto_loss.item() * batch_x.size(0)

            probs = torch.sigmoid(class_logits)
            preds = (probs >= 0.5).float()

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return {
        "loss": total_loss / total,
        "recon_loss": total_recon_loss / total,
        "cls_loss": total_cls_loss / total,
        "contrastive_loss": total_contrastive_loss / total,
        "proto_loss": total_proto_loss / total,
        "acc": correct / total,
    }


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

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

    model = ReConHIDv2(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proto_temperature=PROTO_TEMPERATURE,
    ).to(device)

    y_all = dataset.y
    normal_count = (y_all == 0).sum().item()
    attack_count = (y_all == 1).sum().item()

    pos_weight = torch.tensor(
        [normal_count / max(attack_count, 1)],
        dtype=torch.float32,
        device=device,
    )

    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    proto_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n===== TRAIN ReCon-HID v2 START =====")
    print(f"Train samples       : {train_size}")
    print(f"Valid samples       : {valid_size}")
    print(f"Input dim           : {input_dim}")
    print(f"Hidden dim          : {HIDDEN_DIM}")
    print(f"Latent dim          : {LATENT_DIM}")
    print(f"Normal count        : {normal_count}")
    print(f"Attack count        : {attack_count}")
    print(f"Pos weight          : {pos_weight.item():.4f}")
    print(f"Recon weight        : {RECONSTRUCTION_LOSS_WEIGHT}")
    print(f"Cls weight          : {CLASSIFICATION_LOSS_WEIGHT}")
    print(f"Contrastive weight  : {CONTRASTIVE_LOSS_WEIGHT}")
    print(f"Prototype weight    : {PROTOTYPE_LOSS_WEIGHT}")
    print("====================================\n")

    best_valid_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_cls_loss = 0.0
        train_contrastive_loss = 0.0
        train_proto_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_long = batch_y.long()

            optimizer.zero_grad()

            (
                reconstructed,
                class_logits,
                latent,
                latent_norm,
                proto_logits,
                _,
            ) = model(batch_x)

            recon_loss = normal_only_reconstruction_loss(
                reconstructed,
                batch_x,
                batch_y,
            )

            cls_loss = cls_loss_fn(class_logits, batch_y)

            contrastive_loss = supervised_contrastive_loss(
                latent_norm,
                batch_y_long,
                temperature=CONTRASTIVE_TEMPERATURE,
            )

            proto_loss = proto_loss_fn(proto_logits, batch_y_long)

            loss = (
                RECONSTRUCTION_LOSS_WEIGHT * recon_loss
                + CLASSIFICATION_LOSS_WEIGHT * cls_loss
                + CONTRASTIVE_LOSS_WEIGHT * contrastive_loss
                + PROTOTYPE_LOSS_WEIGHT * proto_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_total_loss += loss.item() * batch_x.size(0)
            train_recon_loss += recon_loss.item() * batch_x.size(0)
            train_cls_loss += cls_loss.item() * batch_x.size(0)
            train_contrastive_loss += contrastive_loss.item() * batch_x.size(0)
            train_proto_loss += proto_loss.item() * batch_x.size(0)

            probs = torch.sigmoid(class_logits)
            preds = (probs >= 0.5).float()

            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_metrics = {
            "loss": train_total_loss / train_total,
            "recon_loss": train_recon_loss / train_total,
            "cls_loss": train_cls_loss / train_total,
            "contrastive_loss": train_contrastive_loss / train_total,
            "proto_loss": train_proto_loss / train_total,
            "acc": train_correct / train_total,
        }

        valid_metrics = evaluate(
            model,
            valid_loader,
            cls_loss_fn,
            proto_loss_fn,
            device,
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_metrics['loss']:.6f} "
            f"Recon: {train_metrics['recon_loss']:.6f} "
            f"Cls: {train_metrics['cls_loss']:.6f} "
            f"Con: {train_metrics['contrastive_loss']:.6f} "
            f"Proto: {train_metrics['proto_loss']:.6f} "
            f"Acc: {train_metrics['acc']:.4f} | "
            f"Valid Loss: {valid_metrics['loss']:.6f} "
            f"Recon: {valid_metrics['recon_loss']:.6f} "
            f"Cls: {valid_metrics['cls_loss']:.6f} "
            f"Con: {valid_metrics['contrastive_loss']:.6f} "
            f"Proto: {valid_metrics['proto_loss']:.6f} "
            f"Acc: {valid_metrics['acc']:.4f}"
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
                    "proto_temperature": PROTO_TEMPERATURE,
                    "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
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
                    "loss_weights": {
                        "reconstruction": RECONSTRUCTION_LOSS_WEIGHT,
                        "classification": CLASSIFICATION_LOSS_WEIGHT,
                        "contrastive": CONTRASTIVE_LOSS_WEIGHT,
                        "prototype": PROTOTYPE_LOSS_WEIGHT,
                    },
                },
                SAVE_PATH,
            )

    pd.DataFrame(history).to_csv(
        TRAIN_HISTORY_CSV,
        index=False,
        encoding="utf-8",
    )

    print("\n===== TRAIN ReCon-HID v2 DONE =====")
    print(f"Best valid loss: {best_valid_loss:.6f}")
    print(f"Saved model to : {SAVE_PATH}")
    print(f"Saved history  : {TRAIN_HISTORY_CSV}")


if __name__ == "__main__":
    main()
