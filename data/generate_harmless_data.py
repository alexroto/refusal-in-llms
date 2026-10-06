"""Sample harmless instructions from Alpaca and save them to CSV.
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent / "harmless"
DATASET_ID = "tatsu-lab/alpaca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="Number of prompts to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "alpaca_sample.csv")
    return parser.parse_args()


def load_alpaca() -> pd.DataFrame:
    """Load Alpaca, keeping only self-contained instructions (no extra input field)."""
    frame = load_dataset(DATASET_ID, split="train").to_pandas()
    frame = frame[frame["input"].str.strip() == ""]
    frame = frame[["instruction", "output"]].copy()
    frame["instruction"] = frame["instruction"].str.strip()
    frame["output"] = frame["output"].str.strip()
    return frame[(frame["instruction"] != "") & (frame["output"] != "")]


def main() -> None:
    args = parse_args()

    frame = load_alpaca()
    if args.n > len(frame):
        raise ValueError(f"Requested {args.n} rows but only {len(frame)} are available")

    sample = frame.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    sample["source"] = "alpaca"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    print(f"Saved {len(sample)} rows to {args.output}")


if __name__ == "__main__":
    main()