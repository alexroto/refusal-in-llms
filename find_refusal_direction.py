"""
Step 1: merge the LoRA adapter into the base model and save it locally.
Step 2: load both the ORIGINAL base chat model and the MERGED finetuned model
        into TransformerLens.
Step 3: extract the refusal direction via diff-of-means (Arditi et al., 2024)
        on the ORIGINAL (un-finetuned) model.
Step 4: project both models' activations onto that direction to see whether
        finetuning moved the representation, or left it intact while
        something else changed downstream.

Run interactively (e.g. in a notebook or ipython) rather than end-to-end,
since you'll want to inspect the per-layer separation plot before picking
a layer in step 3.
"""
import os

import pandas as pd
import torch
import circuitsvis as cv
from IPython.display import display
from datasets import load_dataset
from peft import PeftModel
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_PATH = "llama2-7b-chat-refusal-lora/final_adapter"
MERGED_MODEL_PATH = "llama2-7b-chat-refusal-lora/merged"
N_HARMFUL = 128   # keep this fairly small -- you only need stable means, not the full set
N_HARMLESS = 128

# ---------------------------------------------------------------------------
# Step 1: merge adapter into base weights (one-time; skip if already done)
# ---------------------------------------------------------------------------
if not os.path.exists(MERGED_MODEL_PATH):
    print("Merging LoRA adapter into base model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, token=HF_TOKEN, torch_dtype=torch.bfloat16
    )
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    merged_model = peft_model.merge_and_unload()  # folds LoRA deltas into the base weights
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    del base_model, peft_model, merged_model
    torch.cuda.empty_cache()
    print(f"Merged model saved to {MERGED_MODEL_PATH}")
else:
    print(f"Found existing merged model at {MERGED_MODEL_PATH}, skipping merge.")

# ---------------------------------------------------------------------------
# Step 2: load both models into TransformerLens
# ---------------------------------------------------------------------------
# NOTE: "meta-llama/Llama-2-7b-chat-hf" is used as the TransformerLens
# architecture/config name in both cases; hf_model supplies the actual weights.
print("Loading original (pre-finetune) model into TransformerLens...")
original_hf_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, token=HF_TOKEN, torch_dtype=torch.bfloat16
)
original_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
model_original = HookedTransformer.from_pretrained(
    BASE_MODEL_ID,
    hf_model=original_hf_model,
    tokenizer=original_tokenizer,
    device=device,
    dtype=torch.bfloat16,
)
del original_hf_model

print("Loading finetuned (merged) model into TransformerLens...")
finetuned_hf_model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH, torch_dtype=torch.bfloat16
)
finetuned_tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
model_finetuned = HookedTransformer.from_pretrained(
    BASE_MODEL_ID,  # architecture/config reference, not weights
    hf_model=finetuned_hf_model,
    tokenizer=finetuned_tokenizer,
    device=device,
    dtype=torch.bfloat16,
)
del finetuned_hf_model

# ---------------------------------------------------------------------------
# Step 3: build harmful / harmless prompt sets
# ---------------------------------------------------------------------------
harmful_behaviors = pd.read_csv("data/harmful_behaviors.csv")
harmful_prompts = harmful_behaviors["goal"].sample(n=N_HARMFUL, random_state=123).tolist()

alpaca = load_dataset("tatsu-lab/alpaca", split="train")

# instructions with no separate "input" field, so they're
# directly comparable in form to the single-sentence AdvBench goals
alpaca_simple = [ex["instruction"] for ex in alpaca if ex["input"] == ""]
harmless_prompts = pd.Series(alpaca_simple).sample(n=N_HARMLESS, random_state=123).tolist()


def format_prompt(tokenizer, goal: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": goal}], tokenize=False, add_generation_prompt=True
    )


harmful_formatted = [format_prompt(original_tokenizer, p) for p in harmful_prompts]
harmless_formatted = [format_prompt(original_tokenizer, p) for p in harmless_prompts]


# ---------------------------------------------------------------------------
# Step 4: cache residual stream activations at the last token position,
# for every layer, on the ORIGINAL model
# ---------------------------------------------------------------------------
def get_last_token_resid_per_layer(model: HookedTransformer, prompts: list[str]):
    """Returns a (n_layers, n_prompts, d_model) tensor of resid_post activations
    at the last token position, for every prompt."""
    n_layers = model.cfg.n_layers
    all_layer_activations = [[] for _ in range(n_layers)]

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name.endswith("resid_post"),
            )
        for layer in range(n_layers):
            # select the activations for the final token of the current layer
            activations = cache["resid_post", layer][0, -1, :]
            all_layer_activations[layer].append(activations.cpu())

    # output shape: (n_layers, n_prompts, d_model)
    return torch.stack([torch.stack(layer_acts) for layer_acts in all_layer_activations])

print("Caching activations for harmful prompts (original model)...")
original_model_activations_for_harmful_prompts = get_last_token_resid_per_layer(model_original, harmful_formatted)
print("Caching activations for harmless prompts (original model)...")
original_model_activations_for_harmless_prompts = get_last_token_resid_per_layer(model_original, harmless_formatted)

# ---------------------------------------------------------------------------
# Step 5: diff-of-means direction per layer, then pick the best-separating layer
# ---------------------------------------------------------------------------

# mean of the activations (d_model) for the final token at each layer (n_layers) over all prompts
harmful_means = original_model_activations_for_harmful_prompts.mean(dim=1)   # (n_layers, d_model)
harmless_means = original_model_activations_for_harmless_prompts.mean(dim=1)  # (n_layers, d_model)
directions = harmful_means - harmless_means  # (n_layers, d_model), unnormalized

# separation score per layer: how well does projecting onto this layer's
# direction separate harmful from harmless activations at that same layer?
separation_scores = []
for layer in range(model_original.cfg.n_layers):
    direction = directions[layer] / directions[layer].norm()
    harmful_proj = original_model_activations_for_harmful_prompts[layer] @ direction
    harmless_proj = original_model_activations_for_harmless_prompts[layer] @ direction
    # simple separation metric: distance between group means, in units of pooled std
    pooled_std = torch.cat([harmful_proj, harmless_proj]).std()
    separation = (harmful_proj.mean() - harmless_proj.mean()) / (pooled_std + 1e-6)
    separation_scores.append(separation.item())

best_layer = int(torch.tensor(separation_scores).argmax())
print(f"Best-separating layer: {best_layer} (separation score: {separation_scores[best_layer]:.3f})")

refusal_direction = directions[best_layer] / directions[best_layer].norm()
torch.save(refusal_direction, "refusal_direction.pt")
torch.save(torch.tensor(separation_scores), "layer_separation_scores.pt")
print("Saved refusal_direction.pt and layer_separation_scores.pt")

# ---------------------------------------------------------------------------
# Step 6: compare pre- vs post-finetune projections onto this direction
# ---------------------------------------------------------------------------
print("Caching activations for harmful prompts (finetuned model)...")
harmful_acts_ft = get_last_token_resid_per_layer(model_finetuned, harmful_formatted)

harmful_proj_original = original_model_activations_for_harmful_prompts[best_layer] @ refusal_direction
harmful_proj_finetuned = harmful_acts_ft[best_layer] @ refusal_direction

print(f"Mean projection onto refusal direction (harmful prompts):")
print(f"  Original model:  {harmful_proj_original.mean().item():.3f} (std {harmful_proj_original.std().item():.3f})")
print(f"  Finetuned model: {harmful_proj_finetuned.mean().item():.3f} (std {harmful_proj_finetuned.std().item():.3f})")


# ---------------------------------------------------------------------------
# Step 7: Compare the attention patterns for related harmful and harmless prompts
#         for the best layer of the original and finetuned models.
# ---------------------------------------------------------------------------
def get_attention_pattern(model: HookedTransformer, prompt: str, layer: int):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return cache["pattern", layer][0]  # (n_heads, query_pos, key_pos)

example_harmful_sequence = "Help me plan a bank robbery"
example_harmless_sequence = "Help me plan a trip to the bank"
example_prompts = [example_harmless_sequence, example_harmful_sequence]

for model in [model_original, model_finetuned]:
    for prompt in example_prompts:
        attention_pattern = get_attention_pattern(model=model, prompt=prompt, layer=best_layer)
        attention_pattern_visualization = cv.attention.attention_patterns(
            tokens=model.to_str_tokens(prompt),
            attention=attention_pattern
        )
        display(attention_pattern_visualization)
