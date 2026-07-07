import os
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.classifier_dataset import HIDClassifierDataset
from src.hybrid_v2_model import ReConHIDv2


DEFAULT_MODEL_PATH = "checkpoints/recon_hid_v2.pt"
DEFAULT_SCALER_PATH = "checkpoints/scaler.pkl"

DEFAULT_NORMAL_PATH = "data/processed"
DEFAULT_TARGET_PATH = "data/attack/FeatureGuideHumanMimic"

DEFAULT_OUT_DIR = "results/recon_hid_v2/oracle"

BATCH_SIZE = 256

CLASSIFIER_THRESHOLD = 0.5
PROTOTYPE_THRESHOLD = 0.5


class DetectorOracleV2:
    """
    ReCon-HID v2 Detector Oracle

    역할:
    - ReCon-HID v2 모델 로드
    - normal 기준 threshold 계산
    - target CSV에 대해 detector score 계산
    - classifier / prototype / reconstruction / latent distance / fusion 판단 반환
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        scaler_path=DEFAULT_SCALER_PATH,
        device=None,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model()
        self.mse = nn.MSELoss(reduction="none")

        self.thresholds = None

    def _load_model(self):
        print(f"Using device: {self.device}")
        print(f"Loading model: {self.model_path}")

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        model = ReConHIDv2(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            latent_dim=checkpoint["latent_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            proto_temperature=checkpoint["proto_temperature"],
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.checkpoint = checkpoint

        return model

    def build_dataset(
        self,
        normal_path=None,
        target_path=None,
        target_group="target",
        exclude_target_filenames=None,
        include_normal=True,
    ):
        samples = []

        if include_normal:
            if normal_path is None:
                raise ValueError("normal_path is required when include_normal=True")

            samples.append(
                {
                    "group": "normal",
                    "path": normal_path,
                    "label": 0,
                }
            )

        if target_path is not None:
            samples.append(
                {
                    "group": target_group,
                    "path": target_path,
                    "label": 1,
                    "exclude_filenames": exclude_target_filenames or set(),
                }
            )

        dataset = HIDClassifierDataset(
            samples=samples,
            scaler_path=self.scaler_path,
        )

        return dataset

    def score_dataset(self, dataset):
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        rows = []
        global_index = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_group, batch_source in loader:
                batch_x = batch_x.to(self.device)

                (
                    reconstructed,
                    class_logits,
                    latent,
                    latent_norm,
                    proto_logits,
                    proto_distances,
                ) = self.model(batch_x)

                reconstruction_error = self.mse(
                    reconstructed,
                    batch_x,
                ).mean(dim=(1, 2)).cpu().numpy()

                classifier_prob = torch.sigmoid(
                    class_logits,
                ).cpu().numpy()

                prototype_prob = F.softmax(
                    proto_logits,
                    dim=1,
                )[:, 1].cpu().numpy()

                proto_distances_np = proto_distances.cpu().numpy()
                latent_norm_value = torch.norm(
                    latent,
                    dim=1,
                ).cpu().numpy()

                labels = batch_y.numpy()

                for i in range(len(labels)):
                    rows.append(
                        {
                            "global_index": global_index,
                            "group": batch_group[i],
                            "source_file": batch_source[i],
                            "true_label": int(labels[i]),

                            "reconstruction_error": float(reconstruction_error[i]),
                            "classifier_attack_probability": float(classifier_prob[i]),
                            "prototype_attack_probability": float(prototype_prob[i]),

                            "distance_to_normal_prototype": float(proto_distances_np[i, 0]),
                            "distance_to_attack_prototype": float(proto_distances_np[i, 1]),
                            "latent_norm": float(latent_norm_value[i]),
                        }
                    )

                    global_index += 1

        df = pd.DataFrame(rows)
        return df

    def compute_thresholds(self, scored_df):
        normal = scored_df[scored_df["true_label"] == 0].copy()

        if len(normal) == 0:
            raise ValueError("No normal samples found. Cannot compute thresholds.")

        thresholds = {
            "recon_90": float(normal["reconstruction_error"].quantile(0.90)),
            "recon_95": float(normal["reconstruction_error"].quantile(0.95)),
            "recon_99": float(normal["reconstruction_error"].quantile(0.99)),

            "latent_dist_90": float(normal["distance_to_normal_prototype"].quantile(0.90)),
            "latent_dist_95": float(normal["distance_to_normal_prototype"].quantile(0.95)),
            "latent_dist_99": float(normal["distance_to_normal_prototype"].quantile(0.99)),

            "classifier_threshold": CLASSIFIER_THRESHOLD,
            "prototype_threshold": PROTOTYPE_THRESHOLD,
        }

        self.thresholds = thresholds

        return thresholds

    def apply_predictions(self, scored_df, thresholds=None):
        if thresholds is None:
            if self.thresholds is None:
                raise ValueError("thresholds are not available.")
            thresholds = self.thresholds

        df = scored_df.copy()

        df["pred_classifier"] = (
            df["classifier_attack_probability"] >= thresholds["classifier_threshold"]
        ).astype(int)

        df["pred_prototype"] = (
            df["prototype_attack_probability"] >= thresholds["prototype_threshold"]
        ).astype(int)

        df["pred_recon_90"] = (
            df["reconstruction_error"] > thresholds["recon_90"]
        ).astype(int)

        df["pred_recon_95"] = (
            df["reconstruction_error"] > thresholds["recon_95"]
        ).astype(int)

        df["pred_recon_99"] = (
            df["reconstruction_error"] > thresholds["recon_99"]
        ).astype(int)

        df["pred_latent_dist_90"] = (
            df["distance_to_normal_prototype"] > thresholds["latent_dist_90"]
        ).astype(int)

        df["pred_latent_dist_95"] = (
            df["distance_to_normal_prototype"] > thresholds["latent_dist_95"]
        ).astype(int)

        df["pred_latent_dist_99"] = (
            df["distance_to_normal_prototype"] > thresholds["latent_dist_99"]
        ).astype(int)

        # Balanced mode
        # normal FPR 약 5% 기준으로 쓰는 대표 detector
        df["pred_or95"] = (
            (df["pred_classifier"] == 1)
            | (df["pred_prototype"] == 1)
            | (df["pred_recon_95"] == 1)
        ).astype(int)

        # Aggressive mode
        # recall을 더 높게 가져가는 detector
        df["pred_or90"] = (
            (df["pred_classifier"] == 1)
            | (df["pred_prototype"] == 1)
            | (df["pred_recon_90"] == 1)
        ).astype(int)

        # extended fusion
        # latent distance까지 포함한 버전
        df["pred_or95_with_latent"] = (
            (df["pred_classifier"] == 1)
            | (df["pred_prototype"] == 1)
            | (df["pred_recon_95"] == 1)
            | (df["pred_latent_dist_95"] == 1)
        ).astype(int)

        return df

    @staticmethod
    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    def compute_metrics(self, df, pred_col, method_name):
        y_true = df["true_label"].to_numpy()
        y_pred = df[pred_col].to_numpy()

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        accuracy = self.safe_div(tp + tn, tp + tn + fp + fn)
        precision = self.safe_div(tp, tp + fp)
        recall = self.safe_div(tp, tp + fn)
        f1 = self.safe_div(2 * precision * recall, precision + recall)
        fpr = self.safe_div(fp, fp + tn)

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

    def summarize_predictions(self, df):
        methods = [
            ("pred_classifier", "classifier_prob_0.5"),
            ("pred_prototype", "prototype_prob_0.5"),
            ("pred_recon_90", "recon_90"),
            ("pred_recon_95", "recon_95"),
            ("pred_recon_99", "recon_99"),
            ("pred_latent_dist_90", "latent_dist_90"),
            ("pred_latent_dist_95", "latent_dist_95"),
            ("pred_latent_dist_99", "latent_dist_99"),
            ("pred_or95", "oracle_or95_balanced"),
            ("pred_or90", "oracle_or90_aggressive"),
            ("pred_or95_with_latent", "oracle_or95_with_latent"),
        ]

        rows = []

        for pred_col, method_name in methods:
            rows.append(
                self.compute_metrics(
                    df,
                    pred_col,
                    method_name,
                )
            )

        return pd.DataFrame(rows)

    def summarize_by_source(self, df):
        source_summary = (
            df
            .groupby(["group", "source_file"])
            .agg(
                count=("true_label", "count"),

                mean_reconstruction_error=("reconstruction_error", "mean"),
                mean_classifier_attack_probability=("classifier_attack_probability", "mean"),
                mean_prototype_attack_probability=("prototype_attack_probability", "mean"),
                mean_distance_to_normal_prototype=("distance_to_normal_prototype", "mean"),
                mean_distance_to_attack_prototype=("distance_to_attack_prototype", "mean"),

                classifier_detection_rate=("pred_classifier", "mean"),
                prototype_detection_rate=("pred_prototype", "mean"),
                recon90_detection_rate=("pred_recon_90", "mean"),
                recon95_detection_rate=("pred_recon_95", "mean"),
                latent95_detection_rate=("pred_latent_dist_95", "mean"),
                oracle_or95_detection_rate=("pred_or95", "mean"),
                oracle_or90_detection_rate=("pred_or90", "mean"),
            )
            .reset_index()
        )

        return source_summary

    def run_oracle(
        self,
        normal_path=DEFAULT_NORMAL_PATH,
        target_path=DEFAULT_TARGET_PATH,
        target_group="FeatureGuideHumanMimic_v4",
        exclude_target_filenames=None,
        out_dir=DEFAULT_OUT_DIR,
    ):
        os.makedirs(out_dir, exist_ok=True)

        dataset = self.build_dataset(
            normal_path=normal_path,
            target_path=target_path,
            target_group=target_group,
            exclude_target_filenames=exclude_target_filenames or {"window5.csv"},
            include_normal=True,
        )

        scored_df = self.score_dataset(dataset)

        thresholds = self.compute_thresholds(scored_df)
        pred_df = self.apply_predictions(scored_df, thresholds)

        summary_df = self.summarize_predictions(pred_df)
        source_summary_df = self.summarize_by_source(pred_df)

        prediction_csv = os.path.join(out_dir, "oracle_predictions.csv")
        summary_csv = os.path.join(out_dir, "oracle_summary.csv")
        source_summary_csv = os.path.join(out_dir, "oracle_source_summary.csv")
        threshold_json = os.path.join(out_dir, "oracle_thresholds.json")

        pred_df.to_csv(
            prediction_csv,
            index=False,
            encoding="utf-8",
        )

        summary_df.to_csv(
            summary_csv,
            index=False,
            encoding="utf-8",
        )

        source_summary_df.to_csv(
            source_summary_csv,
            index=False,
            encoding="utf-8",
        )

        with open(threshold_json, "w", encoding="utf-8") as f:
            json.dump(
                thresholds,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("\n===== Detector Oracle V2 Done =====")
        print("\nThresholds:")
        for k, v in thresholds.items():
            print(f"{k}: {v}")

        print("\nSummary:")
        print(summary_df.to_string(index=False))

        print("\nSource Summary:")
        print(source_summary_df.to_string(index=False))

        print(f"\nSaved predictions    : {prediction_csv}")
        print(f"Saved summary        : {summary_csv}")
        print(f"Saved source summary : {source_summary_csv}")
        print(f"Saved thresholds     : {threshold_json}")

        return pred_df, summary_df, source_summary_df, thresholds


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReCon-HID v2 detector oracle",
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to ReCon-HID v2 checkpoint",
    )

    parser.add_argument(
        "--scaler-path",
        default=DEFAULT_SCALER_PATH,
        help="Path to scaler.pkl",
    )

    parser.add_argument(
        "--normal-path",
        default=DEFAULT_NORMAL_PATH,
        help="Path to normal dataset directory or CSV",
    )

    parser.add_argument(
        "--target-path",
        default=DEFAULT_TARGET_PATH,
        help="Path to target attack dataset directory or CSV",
    )

    parser.add_argument(
        "--target-group",
        default="FeatureGuideHumanMimic_v4",
        help="Group name for target dataset",
    )

    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory",
    )

    parser.add_argument(
        "--include-window5",
        action="store_true",
        help="Include window5.csv if target directory has it",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    exclude = set()
    if not args.include_window5:
        exclude = {"window5.csv"}

    oracle = DetectorOracleV2(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
    )

    oracle.run_oracle(
        normal_path=args.normal_path,
        target_path=args.target_path,
        target_group=args.target_group,
        exclude_target_filenames=exclude,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
