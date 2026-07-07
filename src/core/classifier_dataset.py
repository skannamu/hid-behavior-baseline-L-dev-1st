import os
import glob
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import SELECTED_FEATURES, WINDOW_SIZE


def resolve_csv_files(path, exclude_filenames=None):
    exclude_filenames = exclude_filenames or set()

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    files = [
        f for f in files
        if os.path.basename(f) not in exclude_filenames
    ]

    if not files:
        raise FileNotFoundError(f"No CSV files found: {path}")

    return files


def load_windows_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    sequences = []

    for _, row in df.iterrows():
        sequence = []

        for t in range(WINDOW_SIZE):
            timestep = []

            for feature in SELECTED_FEATURES:
                col_name = f"t{t}_{feature}"
                value = row.get(col_name, 0.0)

                if pd.isna(value):
                    value = 0.0

                timestep.append(float(value))

            sequence.append(timestep)

        sequences.append(sequence)

    return np.array(sequences, dtype=np.float32)


class HIDClassifierDataset(Dataset):
    def __init__(
        self,
        samples,
        scaler_path="checkpoints/scaler.pkl",
    ):
        xs = []
        ys = []
        source_files = []
        group_names = []

        scaler = joblib.load(scaler_path)

        for item in samples:
            group = item["group"]
            path = item["path"]
            label = item["label"]
            exclude = item.get("exclude_filenames", set())

            files = resolve_csv_files(path, exclude_filenames=exclude)

            print(f"\n[{group}] label={label}")

            for file in files:
                arr = load_windows_from_csv(file)

                n, seq_len, feat_dim = arr.shape
                flat = arr.reshape(-1, feat_dim)
                flat_scaled = scaler.transform(flat)
                arr_scaled = flat_scaled.reshape(n, seq_len, feat_dim)

                xs.append(arr_scaled)
                ys.extend([label] * n)
                source_files.extend([os.path.basename(file)] * n)
                group_names.extend([group] * n)

                print(f"  - {file} samples={n}")

        self.x = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.array(ys, dtype=np.float32), dtype=torch.float32)
        self.source_files = source_files
        self.group_names = group_names

        print("\n===== HID CLASSIFIER DATASET =====")
        print(f"Samples : {len(self.x)}")
        print(f"Shape   : {self.x.shape}")
        print(f"Normal  : {(self.y == 0).sum().item()}")
        print(f"Attack  : {(self.y == 1).sum().item()}")
        print("==================================\n")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.group_names[idx], self.source_files[idx]
