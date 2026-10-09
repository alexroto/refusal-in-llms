from dataclasses import dataclass, field
from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Model Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    dtype: str = "bfloat16"
    batch_size: int = 8
    system_prompt: str | None = None
    chat_template: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    trust_remote_code: bool = False
    generation: dict = field(default_factory=dict)
    # Strings whose first token marks the start of a refusal (used by refusal_direction.py).
    refusal_tokens: list = field(default_factory=list)

    @property
    def slug(self) -> str:
        """Filesystem-safe model id, e.g. 'meta-llama/Llama-2-7b-chat-hf' -> 'meta-llama__Llama-2-7b-chat-hf'."""
        return self.model_id.replace("/", "__")


def load_model_config(path: Path, name: str) -> ModelConfig:
    """Merge `defaults` with the named model's entry. Unknown keys raise a TypeError."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = raw["models"]
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(models)}")

    defaults = raw.get("defaults", {})
    merged = {**defaults, **models[name]}
    merged["generation"] = {
        **defaults.get("generation", {}),
        **models[name].get("generation", {}),
    }
    return ModelConfig(name=name, **merged)
