"""Combine harmful datasets and assign pretrained BERTopic topics.

Run from any directory with: python data/generate_harmful_data.py
"""

import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "harmful"
MODEL_ID = "MaartenGr/BERTopic_Wikipedia"


def load_source(path: Path, source: str) -> pd.DataFrame:
    """Keep goal/target pairs and their dataset provenance."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    missing = {"goal", "target"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame[["goal", "target"]].copy()
    for column in ("goal", "target"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{path}: {column} contains missing or empty values")
    frame["source"] = source
    frame["text"] = frame["goal"].str.strip() + "\n" + frame["target"].str.strip()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_advbench = DATA_DIR / "advbench_completions.csv"
    if not default_advbench.exists():
        default_advbench = DATA_DIR / "advbench.csv"
    parser.add_argument("--advbench", type=Path, default=default_advbench)
    parser.add_argument("--jailbreakbench", type=Path, default=DATA_DIR / "jailbreakbench.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "harmful_topics.csv")
    args = parser.parse_args()

    frame = pd.concat(
        [load_source(args.advbench, "advbench"), load_source(args.jailbreakbench, "jailbreak")],
        ignore_index=True,
    )
    if frame.empty:
        raise ValueError("The input datasets contain no rows")

    from bertopic import BERTopic

    topic_model = BERTopic.load(MODEL_ID)
    topics, _ = topic_model.transform(frame["text"].tolist())
    frame["topic"] = topics
    topic_names = topic_model.get_topic_info().set_index("Topic")["Name"]
    frame["topic_name"] = frame["topic"].map(topic_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
