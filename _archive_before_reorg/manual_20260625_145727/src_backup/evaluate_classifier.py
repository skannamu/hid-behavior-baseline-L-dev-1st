import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.classifier_dataset import HIDClassifierDataset
from src.classifier_model import LSTMSequenceClassifier


NORMAL_DATA_PATH = "data/processed"
FEATURE_GUIDE_DATA_PATH = "data/attack/FeatureGuideHumanMimic"

MODEL_PATH = "checkpoints/lstm_sequence_classifier.pt"
SCALER_PATH = "checkpoints/scaler.pkl"

RESULT_DIR = "results/classifier_lstm"
RESULT_CSV = os.path.join(RESULT_DIR, "classifier_predictions_featureguide_v4.csv")
SUMMARY_CSV = os.path.join(RESULT_DIR, "classifier_summary_featureguide_v4.csv")
SOURCE_SUMMARY_CSV = os.path.join(RESULT_DIR, "classifier_source_summary_featureguide_v4.csv")

BATCH_SIZE = 256
THRESHOLD = 0.5


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = LSTMSequenceClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
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
        drop_last=False,
    )

    rows = []
    global_index = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_group, batch_source in loader:
            batch_x = batch_x.to(device)

            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()

            labels = batch_y.numpy()

            for i in range(len(probs)):
                prob = float(probs[i])
                pred = 1 if prob >= THRESHOLD else 0
                true = int(labels[i])

                rows.append({
                    "global_index": global_index,
                    "group": batch_group[i],
                    "source_file": batch_source[i],
                    "true_label": true,
                    "attack_probability": prob,
                    "pred_label": pred,
                    "correct": int(pred == true),
                })

                global_index += 1

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8")

    y_true = result_df["true_label"].to_numpy()
    y_pred = result_df["pred_label"].to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)

    summary_df = pd.DataFrame([
        {
            "threshold": THRESHOLD,
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
    ])

    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")

    source_summary_df = (
        result_df
        .groupby(["group", "source_file"])
        .agg(
            count=("pred_label", "count"),
            mean_attack_probability=("attack_probability", "mean"),
            detection_rate=("pred_label", "mean"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    source_summary_df.to_csv(SOURCE_SUMMARY_CSV, index=False, encoding="utf-8")

    print("\n===== LSTM CLASSIFIER EVALUATION =====")
    print(summary_df.to_string(index=False))
    print("======================================\n")

    print("\n===== SOURCE SUMMARY =====")
    print(source_summary_df.to_string(index=False))
    print("==========================\n")

    print(f"Saved predictions    : {RESULT_CSV}")
    print(f"Saved summary        : {SUMMARY_CSV}")
    print(f"Saved source summary : {SOURCE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
