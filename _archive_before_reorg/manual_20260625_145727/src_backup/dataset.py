import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


WINDOW_SIZE = 50

SELECTED_FEATURES = [
    "hold_time",
    "flight_time",
    "press_to_press_time",
    "release_to_release_time",
    "overlap_ratio",
    "simultaneous_key_count",
    "modifier_count",
    "shortcut_flag",
    "correction_ratio",
    "keys_per_second",
    "burst_density",
    "timing_variance",
    "timing_entropy",
    "pause_duration",
    "current_key_category_id",
]


def resolve_csv_files(path_or_paths):
    """
    path_or_paths:
    - "data/processed/window.csv"
    - "data/processed"
    - ["data/processed/a.csv", "data/processed/b.csv"]
    """

    if isinstance(path_or_paths, (list, tuple)):
        csv_files = []
        for p in path_or_paths:
            csv_files.extend(resolve_csv_files(p))
        return csv_files

    path = Path(path_or_paths)

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Not a CSV file: {path}")
        return [str(path)]

    if path.is_dir():
        csv_files = sorted(str(p) for p in path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return csv_files

    raise FileNotFoundError(f"Path not found: {path}")


class HIDDataset(Dataset):
    def __init__(
        self,
        csv_path,
        scaler_path=None,
        fit_scaler=False,
        save_scaler=False,
        label=None,
    ):
        print("Loading dataset...")

        self.csv_files = resolve_csv_files(csv_path)

        dfs = []

        for file_path in self.csv_files:
            df = pd.read_csv(file_path)
            df["source_file"] = os.path.basename(file_path)

            if label is not None:
                df["label"] = label

            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        sequences = []

        for _, row in self.df.iterrows():
            sequence = []

            for t in range(WINDOW_SIZE):
                timestep_features = []

                for feature in SELECTED_FEATURES:
                    col_name = f"t{t}_{feature}"
                    value = row.get(col_name, 0.0)

                    if pd.isna(value):
                        value = 0.0

                    timestep_features.append(float(value))

                sequence.append(timestep_features)

            sequences.append(sequence)

        self.data = np.array(sequences, dtype=np.float32)

        n_samples, seq_len, feat_dim = self.data.shape
        flat_data = self.data.reshape(-1, feat_dim)

        if fit_scaler:
            self.scaler = StandardScaler()
            flat_data = self.scaler.fit_transform(flat_data)

            if save_scaler:
                if scaler_path is None:
                    raise ValueError("scaler_path is required when save_scaler=True")

                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                joblib.dump(self.scaler, scaler_path)
                print(f"Saved scaler to: {scaler_path}")

        else:
            if scaler_path is None:
                raise ValueError("scaler_path is required when fit_scaler=False")

            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")

            self.scaler = joblib.load(scaler_path)
            flat_data = self.scaler.transform(flat_data)

        self.data = flat_data.reshape(n_samples, seq_len, feat_dim)
        self.data = torch.tensor(self.data, dtype=torch.float32)

        print("\n===== DATASET INFO =====")
        print("CSV Files:")
        for file_path in self.csv_files:
            print(f"  - {file_path}")
        print(f"Samples      : {n_samples}")
        print(f"Sequence Len : {seq_len}")
        print(f"Feature Dim  : {feat_dim}")
        print(f"Tensor Shape : {self.data.shape}")
        print("========================\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]