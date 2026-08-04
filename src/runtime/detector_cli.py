from __future__ import annotations

import argparse
import json
from src.evaluation.evaluate_autoencoder import evaluate_d0


def parse_args():
    parser = argparse.ArgumentParser(description='Runtime-style D0 detector CLI for window CSV input')
    parser.add_argument('--config', required=True)
    parser.add_argument('--input', required=True, help='Window CSV file or directory to score as target input')
    parser.add_argument('--out-name', default='runtime_d0_detection')
    return parser.parse_args()


def main():
    from src.framework.config import load_config
    args = parse_args(); cfg = load_config(args.config)
    result = evaluate_d0(cfg, attack_path=args.input, out_name=args.out_name)
    print(json.dumps(result['summary'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
