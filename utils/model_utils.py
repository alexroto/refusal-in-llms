from baseline_refusal import HF_TOKEN, DEVICE
from utils.model_config import ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Helper functions to:
# 1. Load in the model and its tokenizer
# 2. Format the chat prompt for the model per the model config
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, token=HF_TOKEN, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if cfg.chat_template is not None:
        tokenizer.chat_template = cfg.chat_template

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        token=HF_TOKEN,
        dtype=getattr(torch, cfg.dtype),
        trust_remote_code=cfg.trust_remote_code,
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def format_chat_prompt(tokenizer, cfg: ModelConfig, prompt: str) -> str:
    # system_prompt=None means no system message, so the baseline isn't confounded by
    # extra safety instructions. See model_configs.yaml for per-model exceptions.
    messages = []
    if cfg.system_prompt is not None:
        messages.append({"role": "system", "content": cfg.system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **cfg.chat_template_kwargs,
    )
