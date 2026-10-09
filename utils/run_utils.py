from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RunPaths:
    checkpoint: Path
    failed_log: Path
    marker: Path

