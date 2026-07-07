import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.classifier_dataset import HIDClassifierDataset
from src.hybrid_v2_model import ReConHIDv2


MODEL_PATH = "checkpoints/recon_hid_v2.pt"
SCALER_PATH = "checkpoints/scaler.pkl"

NORMAL_DATA_PATH = "data/processed"
HUMAN_MIMIC_DATA_PATH = "data/attack/HumanMimic"
STRONG_HUMAN_MIMIC_DATA_PATH = "data/attack/StrongHumanMimic"
FEATURE_GUIDE_DATA_PATH = "data/attack/FeatureGuideHumanMimic"

OUT_DIR = "results/recon_hid_v2/latent_visualization"

BATCH_SIZE = 256
MAX_SAMPLES_PER_GROUP = 800
RANDOM_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_per_group(df, max_samples):
    sampled = []

    for group, sub in df.groupby("group"):
        if len(sub) > max_samples:
            sampled.append(
                sub.sample(
                    n=max_samples,
                    random_state=RANDOM_SEED,
                )
            )
        else:
            sampled.append(sub)

    return pd.concat(sampled, axis=0).reset_index(drop=True)


def run_tsne(features):
    """
    sklearn 버전에 따라 TSNE 인자가 조금 다를 수 있어서 fallback 처리함.
    """
    pca_dim = min(10, features.shape[1])
    pca_features = PCA(n_components=pca_dim, random_state=RANDOM_SEED).fit_transform(features)

    try:
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        return tsne.fit_transform(pca_features)
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            n_iter=1000,
            random_state=RANDOM_SEED,
        )
        return tsne.fit_transform(pca_features)


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

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
            "group": "HumanMimic",
            "path": HUMAN_MIMIC_DATA_PATH,
            "label": 1,
        },
        {
            "group": "StrongHumanMimic",
            "path": STRONG_HUMAN_MIMIC_DATA_PATH,
            "label": 1,
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
    latent_rows = []

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

            prototype_prob = F.softmax(
                proto_logits,
                dim=1,
            )[:, 1].cpu().numpy()

            proto_distances_np = proto_distances.cpu().numpy()
            latent_np = latent_norm.cpu().numpy()

            labels = batch_y.numpy()

            for i in range(len(labels)):
                row = {
                    "global_index": global_index,
                    "group": batch_group[i],
                    "source_file": batch_source[i],
                    "true_label": int(labels[i]),
                    "reconstruction_error": float(reconstruction_error[i]),
                    "classifier_attack_probability": float(classifier_prob[i]),
                    "prototype_attack_probability": float(prototype_prob[i]),
                    "distance_to_normal_prototype": float(proto_distances_np[i, 0]),
                    "distance_to_attack_prototype": float(proto_distances_np[i, 1]),
                }

                rows.append(row)

                latent_row = {
                    "global_index": global_index,
                    "group": batch_group[i],
                }

                for j in range(latent_np.shape[1]):
                    latent_row[f"z{j}"] = float(latent_np[i, j])

                latent_rows.append(latent_row)

                global_index += 1

    score_df = pd.DataFrame(rows)
    latent_df = pd.DataFrame(latent_rows)

    full_df = score_df.merge(
        latent_df,
        on=["global_index", "group"],
        how="inner",
    )

    full_csv = os.path.join(OUT_DIR, "recon_hid_v2_latent_scores_full.csv")
    full_df.to_csv(full_csv, index=False, encoding="utf-8")

    print(f"Saved full latent score CSV: {full_csv}")

    sampled_df = sample_per_group(
        full_df,
        MAX_SAMPLES_PER_GROUP,
    )

    latent_cols = [c for c in sampled_df.columns if c.startswith("z")]
    latent_features = sampled_df[latent_cols].to_numpy()

    print("\nRunning t-SNE...")
    tsne_xy = run_tsne(latent_features)

    sampled_df["tsne_x"] = tsne_xy[:, 0]
    sampled_df["tsne_y"] = tsne_xy[:, 1]

    sampled_csv = os.path.join(OUT_DIR, "recon_hid_v2_latent_tsne_sampled.csv")
    sampled_df.to_csv(sampled_csv, index=False, encoding="utf-8")

    print(f"Saved sampled t-SNE CSV: {sampled_csv}")

    # 1. t-SNE scatter
    plt.figure(figsize=(10, 8))

    for group, sub in sampled_df.groupby("group"):
        plt.scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            s=12,
            alpha=0.65,
            label=group,
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("ReCon-HID v2 Latent Space t-SNE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "01_latent_tsne_by_group.png"),
        dpi=300,
    )
    plt.close()

    # 2. mean reconstruction error by group
    group_summary = (
        full_df
        .groupby("group")
        .agg(
            count=("global_index", "count"),
            mean_reconstruction_error=("reconstruction_error", "mean"),
            mean_classifier_attack_probability=("classifier_attack_probability", "mean"),
            mean_prototype_attack_probability=("prototype_attack_probability", "mean"),
            mean_distance_to_normal_prototype=("distance_to_normal_prototype", "mean"),
            mean_distance_to_attack_prototype=("distance_to_attack_prototype", "mean"),
        )
        .reset_index()
    )

    group_summary_csv = os.path.join(OUT_DIR, "group_score_summary.csv")
    group_summary.to_csv(group_summary_csv, index=False, encoding="utf-8")

    print("\n===== GROUP SCORE SUMMARY =====")
    print(group_summary.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.bar(
        group_summary["group"],
        group_summary["mean_reconstruction_error"],
    )
    plt.ylabel("Mean reconstruction error")
    plt.title("Mean Reconstruction Error by Group")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "02_mean_reconstruction_error_by_group.png"),
        dpi=300,
    )
    plt.close()

    # 3. mean classifier probability by group
    plt.figure(figsize=(10, 6))
    plt.bar(
        group_summary["group"],
        group_summary["mean_classifier_attack_probability"],
    )
    plt.ylabel("Mean classifier attack probability")
    plt.title("Mean Classifier Attack Probability by Group")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "03_mean_classifier_probability_by_group.png"),
        dpi=300,
    )
    plt.close()

    # 4. mean prototype probability by group
    plt.figure(figsize=(10, 6))
    plt.bar(
        group_summary["group"],
        group_summary["mean_prototype_attack_probability"],
    )
    plt.ylabel("Mean prototype attack probability")
    plt.title("Mean Prototype Attack Probability by Group")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "04_mean_prototype_probability_by_group.png"),
        dpi=300,
    )
    plt.close()

    # 5. distance to normal prototype
    plt.figure(figsize=(10, 6))
    plt.bar(
        group_summary["group"],
        group_summary["mean_distance_to_normal_prototype"],
    )
    plt.ylabel("Mean distance to normal prototype")
    plt.title("Distance to Normal Prototype by Group")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "05_distance_to_normal_prototype_by_group.png"),
        dpi=300,
    )
    plt.close()

    # 6. source-level summary for FeatureGuideHumanMimic v4
    fg = full_df[full_df["group"] == "FeatureGuideHumanMimic_v4"].copy()

    source_summary = (
        fg
        .groupby("source_file")
        .agg(
            count=("global_index", "count"),
            mean_reconstruction_error=("reconstruction_error", "mean"),
            mean_classifier_attack_probability=("classifier_attack_probability", "mean"),
            mean_prototype_attack_probability=("prototype_attack_probability", "mean"),
            mean_distance_to_normal_prototype=("distance_to_normal_prototype", "mean"),
            mean_distance_to_attack_prototype=("distance_to_attack_prototype", "mean"),
        )
        .reset_index()
    )

    source_summary_csv = os.path.join(OUT_DIR, "featureguide_source_score_summary.csv")
    source_summary.to_csv(source_summary_csv, index=False, encoding="utf-8")

    print("\n===== FEATUREGUIDE SOURCE SUMMARY =====")
    print(source_summary.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.bar(
        source_summary["source_file"],
        source_summary["mean_reconstruction_error"],
    )
    plt.ylabel("Mean reconstruction error")
    plt.title("FeatureGuideHumanMimic v4 Reconstruction Error by Source")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "06_featureguide_reconstruction_error_by_source.png"),
        dpi=300,
    )
    plt.close()

    print("\n===== ReCon-HID v2 Latent Visualization Done =====")
    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
