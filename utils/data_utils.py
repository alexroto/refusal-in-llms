import json

import pandas as pd

from baseline_refusal import META_COLUMNS, save_failed
from utils.data_spec import DatasetSpec
from utils.run_utils import RunPaths


def load_prompts(spec: DatasetSpec) -> pd.DataFrame:
    """
    Return one row per prompt: prompt_id (input row index), prompt, plus any META_COLUMNS.
    """
    df = pd.read_csv(spec.input_path)
    if spec.prompt_column not in df.columns:
        raise ValueError(f"{spec.input_path}: missing column '{spec.prompt_column}'")

    prompts = pd.DataFrame(
        {"prompt_id": df.index, "prompt": df[spec.prompt_column].astype(str).str.strip()}
    )
    for column in META_COLUMNS:
        if column in df.columns:
            prompts[column] = df[column]
    return prompts


def load_resume_state(paths: RunPaths, prompts: pd.DataFrame) -> tuple[list[dict], list[int]]:
    """
    Recover completed results and failed prompt ids from a previous run.
    """
    failed_ids: list[int] = []
    if paths.failed_log.exists():
        failed_ids = pd.read_csv(paths.failed_log)["prompt_id"].tolist()

    if paths.marker.exists():
        crashed_ids = json.loads(paths.marker.read_text())
        print(f"Previous run crashed mid-batch; logging {len(crashed_ids)} prompts as failed.")
        failed_ids.extend(i for i in crashed_ids if i not in failed_ids)
        save_failed(paths, prompts, failed_ids)
        paths.marker.unlink()

    results: list[dict] = []
    if paths.checkpoint.exists():
        results = pd.read_csv(paths.checkpoint).to_dict("records")
        print(f"Resuming: {len(results)} prompts already completed.")

    return results, failed_ids
