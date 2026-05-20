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


class HIDDataset(Dataset):

    def __init__(self, csv_path):

        print("Loading dataset...")

        self.df = pd.read_csv(csv_path)

        self.scaler = StandardScaler()

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

        self.data = np.array(
            sequences,
            dtype=np.float32
        )

        n_samples, seq_len, feat_dim = self.data.shape

        self.data = self.data.reshape(
            -1,
            feat_dim
        )

        self.data = self.scaler.fit_transform(
            self.data
        )

        self.data = self.data.reshape(
            n_samples,
            seq_len,
            feat_dim
        )

        self.data = torch.tensor(
            self.data,
            dtype=torch.float32
        )

        print("\n===== DATASET INFO =====")
        print(f"Samples      : {n_samples}")
        print(f"Sequence Len : {seq_len}")
        print(f"Feature Dim  : {feat_dim}")
        print(f"Tensor Shape : {self.data.shape}")
        print("========================\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]