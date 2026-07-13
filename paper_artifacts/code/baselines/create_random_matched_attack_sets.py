from pathlib import Path
import pandas as pd

SEED = 42

def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def sample_random_matched(candidate_path, guided_train_path, out_path, seed=42):
    cand = read_csv(candidate_path)
    guided = read_csv(guided_train_path)

    n = len(guided)
    if len(cand) < n:
        raise ValueError(f"candidate pool too small: cand={len(cand)}, target={n}")

    sampled = cand.sample(n=n, random_state=seed).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)

    return {
        "candidate_path": str(candidate_path),
        "guided_train_path": str(guided_train_path),
        "out_path": str(out_path),
        "candidate_rows": len(cand),
        "guided_train_rows": len(guided),
        "random_rows": len(sampled),
        "seed": seed,
    }

def main():
    rows = []

    rows.append(sample_random_matched(
        candidate_path="data/attack/AdaptiveMimic_v6_candidate/v6_candidates.csv",
        guided_train_path="data/attack/AdaptiveMimic_v6_D1_OR90_bypass_train/v6_or90_train.csv",
        out_path="data/attack/RandomMatched_D3_train/v6_random_matched_train.csv",
        seed=SEED,
    ))

    rows.append(sample_random_matched(
        candidate_path="data/attack/AdaptiveMimic_v7_candidate/v7_candidates.csv",
        guided_train_path="data/attack/AdaptiveMimic_v7_D2_OR90_bypass_train/v7_or90_train.csv",
        out_path="data/attack/RandomMatched_D3_train/v7_random_matched_train.csv",
        seed=SEED + 1,
    ))

    summary = pd.DataFrame(rows)
    out_summary = Path("paper_artifacts/tables/baselines/random_matched_dataset_summary.csv")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print(f"[saved] {out_summary}")

if __name__ == "__main__":
    main()
