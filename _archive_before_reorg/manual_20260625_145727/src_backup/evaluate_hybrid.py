import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classifier_dataset import HIDClassifierDataset
from src.hybrid_model import ReConHID


NORMAL_DATA_PATH = "data/processed"
FEATURE_GUIDE_DATA_PATH = "data/attack/FeatureGuideHumanMimic"

MODEL_PATH = "checkpoints/recon_hid_v1.pt"
SCALER_PATH = "checkpoints/scaler.pkl"

RESULT_DIR = "results/recon_hid_v1"
RESULT_CSV = os.path.join(RESULT_DIR, "recon_hid_v1_predictions_featureguide_v4.csv")
SUMMARY_CSV = os.path.join(RESULT_DIR, "recon_hid_v1_summary_featureguide_v4.csv")
SOURCE_SUMMARY_CSV = os.path.join(RESULT_DIR, "recon_hid_v1_source_summary_featureguide_v4.csv")

BATCH_SIZE = 256
CLASSIFIER_THRESHOLD = 0.5


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def compute_summary(df, pred_col, threshold_name):
    y_true = df["true_label"].to_numpy()
    y_pred = df[pred_col].to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)

    return {
        "method": threshold_name,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall_attack_detection_rate": recall,
        "f1": f1,
        "normal_false_positive_rate": fpr,
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = ReConHID(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = [
        {
            "group": "normal",
            "path": NORMAL_DATA_PATH,
            "label": 0,
        },
        {
            "group": "FeatureGuideHumanMimic_v4",
            "path": FEATURE_GUIDE_DATA_PATH,
            "label": 1,
            "exclude_filenames": {"window5.csv"},
        },
    ]

    dataset = HIDClassifierDataset(
        samples=samples,
        scaler_path=SCALER_PATH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    mse = nn.MSELoss(reduction="none")

    rows = []
    global_index = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_group, batch_source in loader:
            batch_x = batch_x.to(device)

            reconstructed, logits, latent = model(batch_x)

            loss = mse(reconstructed, batch_x)
            sample_recon_error = loss.mean(dim=(1, 2)).cpu().numpy()

            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch_y.numpy()

            latent_norm = torch.norm(latent, dim=1).cpu().numpy()

            for i in range(len(probs)):
                prob = float(probs[i])
                recon_error = float(sample_recon_error[i])
                true = int(labels[i])

                rows.append({
                    "global_index": global_index,
                    "group": batch_group[i],
                    "source_file": batch_source[i],
                    "true_label": true,
                    "reconstruction_error": recon_error,
                    "attack_probability": prob,
                    "latent_norm": float(latent_norm[i]),
                    "pred_classifier": int(prob >= CLASSIFIER_THRESHOLD),
                })

                global_index += 1

    result_df = pd.DataFrame(rows)

    normal = result_df[result_df["true_label"] == 0]
    attack = result_df[result_df["true_label"] == 1]

    recon_90 = normal["reconstruction_error"].quantile(0.90)
    recon_95 = normal["reconstruction_error"].quantile(0.95)
    recon_99 = normal["reconstruction_error"].quantile(0.99)

    result_df["pred_recon_90"] = (
        result_df["reconstruction_error"] > recon_90
    ).astype(int)
    result_df["pred_recon_95"] = (
        result_df["reconstruction_error"] > recon_95
    ).astype(int)
    result_df["pred_recon_99"] = (
        result_df["reconstruction_error"] > recon_99
    ).astype(int)

    # 단순 hybrid rule:
    # classifier가 attack이라고 하거나 reconstruction error가 normal 95% threshold를 넘으면 attack
    result_df["pred_hybrid_or_95"] = (
        (result_df["pred_classifier"] == 1)
        | (result_df["pred_recon_95"] == 1)
    ).astype(int)

    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8")

    summaries = [
        compute_summary(result_df, "pred_classifier", "classifier_prob_0.5"),
        compute_summary(result_df, "pred_recon_90", f"recon_90_{recon_90:.6f}"),
        compute_summary(result_df, "pred_recon_95", f"recon_95_{recon_95:.6f}"),
        compute_summary(result_df, "pred_recon_99", f"recon_99_{recon_99:.6f}"),
        compute_summary(result_df, "pred_hybrid_or_95", "hybrid_classifier_or_recon95"),
    ]

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")

    source_summary_df = (
        result_df
        .groupby(["group", "source_file"])
        .agg(
            count=("true_label", "count"),
            mean_reconstruction_error=("reconstruction_error", "mean"),
            mean_attack_probability=("attack_probability", "mean"),
            classifier_detection_rate=("pred_classifier", "mean"),
            recon95_detection_rate=("pred_recon_95", "mean"),
            hybrid_or95_detection_rate=("pred_hybrid_or_95", "mean"),
        )
        .reset_index()
    )

    source_summary_df.to_csv(SOURCE_SUMMARY_CSV, index=False, encoding="utf-8")

    print("\n===== ReCon-HID v1 EVALUATION =====")
    print(summary_df.to_string(index=False))
    print("===================================\n")

    print("\n===== SOURCE SUMMARY =====")
    print(source_summary_df.to_string(index=False))
    print("==========================\n")

    print(f"Normal recon 90 threshold: {recon_90:.6f}")
    print(f"Normal recon 95 threshold: {recon_95:.6f}")
    print(f"Normal recon 99 threshold: {recon_99:.6f}")

    print(f"\nSaved predictions    : {RESULT_CSV}")
    print(f"Saved summary        : {SUMMARY_CSV}")
    print(f"Saved source summary : {SOURCE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
