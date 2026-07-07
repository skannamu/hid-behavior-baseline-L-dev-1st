import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.classifier_dataset import HIDClassifierDataset
from src.hybrid_v2_model import ReConHIDv2


NORMAL_DATA_PATH = "data/processed"
FEATURE_GUIDE_DATA_PATH = "data/attack/FeatureGuideHumanMimic"

MODEL_PATH = "checkpoints/recon_hid_v2.pt"
SCALER_PATH = "checkpoints/scaler.pkl"

RESULT_DIR = "results/recon_hid_v2"
RESULT_CSV = os.path.join(RESULT_DIR, "recon_hid_v2_predictions_featureguide_v4.csv")
SUMMARY_CSV = os.path.join(RESULT_DIR, "recon_hid_v2_summary_featureguide_v4.csv")
SOURCE_SUMMARY_CSV = os.path.join(RESULT_DIR, "recon_hid_v2_source_summary_featureguide_v4.csv")

BATCH_SIZE = 256
CLASSIFIER_THRESHOLD = 0.5
PROTOTYPE_THRESHOLD = 0.5


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def compute_summary(df, pred_col, method_name):
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
        "method": method_name,
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

    model = ReConHIDv2(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        proto_temperature=checkpoint["proto_temperature"],
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

            (
                reconstructed,
                class_logits,
                latent,
                latent_norm,
                proto_logits,
                proto_distances,
            ) = model(batch_x)

            reconstruction_error = mse(
                reconstructed,
                batch_x,
            ).mean(dim=(1, 2)).cpu().numpy()

            classifier_prob = torch.sigmoid(
                class_logits
            ).cpu().numpy()

            proto_prob = F.softmax(
                proto_logits,
                dim=1,
            )[:, 1].cpu().numpy()

            proto_distances_np = proto_distances.cpu().numpy()

            distance_to_normal = proto_distances_np[:, 0]
            distance_to_attack = proto_distances_np[:, 1]
            latent_norm_value = torch.norm(latent, dim=1).cpu().numpy()

            labels = batch_y.numpy()

            for i in range(len(classifier_prob)):
                rows.append({
                    "global_index": global_index,
                    "group": batch_group[i],
                    "source_file": batch_source[i],
                    "true_label": int(labels[i]),
                    "reconstruction_error": float(reconstruction_error[i]),
                    "classifier_attack_probability": float(classifier_prob[i]),
                    "prototype_attack_probability": float(proto_prob[i]),
                    "distance_to_normal_prototype": float(distance_to_normal[i]),
                    "distance_to_attack_prototype": float(distance_to_attack[i]),
                    "latent_norm": float(latent_norm_value[i]),
                    "pred_classifier": int(classifier_prob[i] >= CLASSIFIER_THRESHOLD),
                    "pred_prototype": int(proto_prob[i] >= PROTOTYPE_THRESHOLD),
                })

                global_index += 1

    result_df = pd.DataFrame(rows)

    normal = result_df[result_df["true_label"] == 0]

    recon_90 = normal["reconstruction_error"].quantile(0.90)
    recon_95 = normal["reconstruction_error"].quantile(0.95)
    recon_99 = normal["reconstruction_error"].quantile(0.99)

    latent_dist_90 = normal["distance_to_normal_prototype"].quantile(0.90)
    latent_dist_95 = normal["distance_to_normal_prototype"].quantile(0.95)
    latent_dist_99 = normal["distance_to_normal_prototype"].quantile(0.99)

    result_df["pred_recon_90"] = (
        result_df["reconstruction_error"] > recon_90
    ).astype(int)
    result_df["pred_recon_95"] = (
        result_df["reconstruction_error"] > recon_95
    ).astype(int)
    result_df["pred_recon_99"] = (
        result_df["reconstruction_error"] > recon_99
    ).astype(int)

    result_df["pred_latent_dist_90"] = (
        result_df["distance_to_normal_prototype"] > latent_dist_90
    ).astype(int)
    result_df["pred_latent_dist_95"] = (
        result_df["distance_to_normal_prototype"] > latent_dist_95
    ).astype(int)
    result_df["pred_latent_dist_99"] = (
        result_df["distance_to_normal_prototype"] > latent_dist_99
    ).astype(int)

    # Fusion 1:
    # classifier 또는 prototype이 attack이면 attack
    result_df["pred_fusion_cls_or_proto"] = (
        (result_df["pred_classifier"] == 1)
        | (result_df["pred_prototype"] == 1)
    ).astype(int)

    # Fusion 2:
    # classifier/prototype/latent distance 중 하나라도 attack이면 attack
    result_df["pred_fusion_cls_proto_or_latent95"] = (
        (result_df["pred_classifier"] == 1)
        | (result_df["pred_prototype"] == 1)
        | (result_df["pred_latent_dist_95"] == 1)
    ).astype(int)

    result_df.to_csv(
        RESULT_CSV,
        index=False,
        encoding="utf-8",
    )

    summaries = [
        compute_summary(result_df, "pred_classifier", "classifier_prob_0.5"),
        compute_summary(result_df, "pred_prototype", "prototype_prob_0.5"),
        compute_summary(result_df, "pred_recon_90", f"recon_90_{recon_90:.6f}"),
        compute_summary(result_df, "pred_recon_95", f"recon_95_{recon_95:.6f}"),
        compute_summary(result_df, "pred_recon_99", f"recon_99_{recon_99:.6f}"),
        compute_summary(result_df, "pred_latent_dist_90", f"latent_dist_90_{latent_dist_90:.6f}"),
        compute_summary(result_df, "pred_latent_dist_95", f"latent_dist_95_{latent_dist_95:.6f}"),
        compute_summary(result_df, "pred_latent_dist_99", f"latent_dist_99_{latent_dist_99:.6f}"),
        compute_summary(result_df, "pred_fusion_cls_or_proto", "fusion_classifier_or_prototype"),
        compute_summary(result_df, "pred_fusion_cls_proto_or_latent95", "fusion_classifier_prototype_or_latent95"),
    ]

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8",
    )

    source_summary_df = (
        result_df
        .groupby(["group", "source_file"])
        .agg(
            count=("true_label", "count"),
            mean_reconstruction_error=("reconstruction_error", "mean"),
            mean_classifier_attack_probability=("classifier_attack_probability", "mean"),
            mean_prototype_attack_probability=("prototype_attack_probability", "mean"),
            mean_distance_to_normal_prototype=("distance_to_normal_prototype", "mean"),
            classifier_detection_rate=("pred_classifier", "mean"),
            prototype_detection_rate=("pred_prototype", "mean"),
            latent95_detection_rate=("pred_latent_dist_95", "mean"),
            fusion_cls_proto_detection_rate=("pred_fusion_cls_or_proto", "mean"),
            fusion_all_detection_rate=("pred_fusion_cls_proto_or_latent95", "mean"),
        )
        .reset_index()
    )

    source_summary_df.to_csv(
        SOURCE_SUMMARY_CSV,
        index=False,
        encoding="utf-8",
    )

    print("\n===== ReCon-HID v2 EVALUATION =====")
    print(summary_df.to_string(index=False))
    print("===================================\n")

    print("\n===== SOURCE SUMMARY =====")
    print(source_summary_df.to_string(index=False))
    print("==========================\n")

    print(f"Normal recon 90 threshold       : {recon_90:.6f}")
    print(f"Normal recon 95 threshold       : {recon_95:.6f}")
    print(f"Normal recon 99 threshold       : {recon_99:.6f}")
    print(f"Normal latent dist 90 threshold : {latent_dist_90:.6f}")
    print(f"Normal latent dist 95 threshold : {latent_dist_95:.6f}")
    print(f"Normal latent dist 99 threshold : {latent_dist_99:.6f}")

    print(f"\nSaved predictions    : {RESULT_CSV}")
    print(f"Saved summary        : {SUMMARY_CSV}")
    print(f"Saved source summary : {SOURCE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
