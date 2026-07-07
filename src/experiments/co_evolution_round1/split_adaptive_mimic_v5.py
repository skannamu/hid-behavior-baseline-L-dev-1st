import os
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_INPUT_DIR = "data/attack/AdaptiveMimic_v5_constrained_clean"
DEFAULT_TRAIN_DIR = "data/attack/AdaptiveMimic_v5_constrained_train"
DEFAULT_TEST_DIR = "data/attack/AdaptiveMimic_v5_constrained_test"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def reset_dir(path):
    os.makedirs(path, exist_ok=True)

    for file in glob.glob(os.path.join(path, "*.csv")):
        os.remove(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    reset_dir(args.train_dir)
    reset_dir(args.test_dir)

    files = sorted(glob.glob(os.path.join(args.input_dir, "window*.csv")))

    if not files:
        raise FileNotFoundError(f"No window*.csv files found in {args.input_dir}")

    total_train = 0
    total_test = 0

    print("\n===== Split AdaptiveMimic v5 =====")

    for file in files:
        source_file = os.path.basename(file)
        df = pd.read_csv(file)

        if len(df) < 2:
            print(f"[SKIP] too few rows: {source_file}")
            continue

        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        train_path = os.path.join(args.train_dir, source_file)
        test_path = os.path.join(args.test_dir, source_file)

        train_df.to_csv(train_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")

        total_train += len(train_df)
        total_test += len(test_df)

        print(
            f"{source_file}: "
            f"total={len(df)} train={len(train_df)} test={len(test_df)}"
        )

    print("=================================")
    print(f"Train total: {total_train}")
    print(f"Test total : {total_test}")
    print(f"Train dir  : {args.train_dir}")
    print(f"Test dir   : {args.test_dir}")


if __name__ == "__main__":
    main()
