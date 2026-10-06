"""Combine harmful datasets and assign pretrained BERTopic topics.

Run from any directory with: python data/generate_harmful_data.py
"""

import argparse
from pathlib import Path
import pandas as pd

from bertopic import BERTopic

DATA_DIR = Path(__file__).resolve().parent / "harmful"
MODEL_ID = "MaartenGr/BERTopic_Wikipedia"
REQUIRED_COLUMNS = ["goal", "target"]


def default_advbench_path() -> Path:
    """Returns the path to the advbench file"""
    return DATA_DIR / "advbench.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advbench", type=Path, default=default_advbench_path())
    parser.add_argument(
        "--jailbreakbench", type=Path, default=DATA_DIR / "jailbreakbench.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "harmful_topics.csv"
    )
    return parser.parse_args()


def load_source(path: Path, source: str) -> pd.DataFrame:
    """Load goal/target pairs from a CSV, validate them, and tag the source."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()

    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    frame = frame[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"{path}: '{column}' contains missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if (frame[column] == "").any():
            raise ValueError(f"{path}: '{column}' contains empty values")

    frame["source"] = source
    frame["text"] = frame["goal"] + "\n" + frame["target"]
    return frame


def assign_topics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `topic` and `topic_name` columns using a pretrained BERTopic model."""
    # Imported lazily so input validation fails fast without loading the heavy dependency.
    model = BERTopic.load(MODEL_ID)
    topics, _ = model.transform(frame["text"].tolist())

    names = model.get_topic_info().set_index("Topic")["Name"]
    return frame.assign(topic=topics, topic_name=pd.Series(topics).map(names).values)


def main() -> None:
    args = parse_args()

    frame = pd.concat(
        [
            load_source(args.advbench, "advbench"),
            load_source(args.jailbreakbench, "jailbreak"),
        ],
        ignore_index=True,
    )
    if frame.empty:
        raise ValueError("The input datasets contain no rows")

    frame = assign_topics(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()