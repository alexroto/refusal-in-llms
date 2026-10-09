from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    input_path: Path
    prompt_column: str
    output_dir: Path
