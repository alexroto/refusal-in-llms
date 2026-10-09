"""
Find a chat model's refusal direction and generate completions with it ablated.

Method: difference-in-means between harmful and harmless prompt activations, as in
Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction"
(arXiv:2406.11717).

Stages (--stage):
    direction   Extract activations, build one candidate direction per (layer, token
                position), score the candidates on held-out prompts, pick the best,
                and save it.
    ablate      Load the saved direction and generate completions on --eval-data with
                it removed from the residual stream at every layer. With
                --condition baseline the same prompts are run WITHOUT ablation, so
                you have a like-for-like comparison.
    all         Both, in order.

Usage (model-dependent settings live in model_configs.yaml):

    python refusal_direction.py --model llama2 --stage direction

    until python refusal_direction.py --model llama2 --stage ablate \\
            --eval-data data/mental_health/prompts.csv; do
        echo "crashed, restarting"; sleep 5
    done

Run the direction stage WITHOUT the restart loop: a deterministic error there would
just loop forever. The loop is only useful for the generation stage, which resumes
from its checkpoint.

Outputs (<model_id> is the HF id with "/" replaced by "__"):

    directions/<model_id>/refusal_direction.pt       the selected direction, shape [d_model]
    directions/<model_id>/refusal_direction.json     layer, position, scores, settings
    directions/<model_id>/candidate_mean_diffs.pt    all candidates [n_positions, n_layers, d_model]
    directions/<model_id>/candidate_scores.csv       scores for every candidate
    directions/<model_id>/splits.json                prompt_ids used for train/val

    <eval-data dir>/<model_id>/ablated_completions.csv     (or unablated_completions.csv)
"""
import argparse
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from baseline_refusal import (
    DATASETS,
    DEFAULT_CONFIG_PATH,
    DEVICE,
    ROOT,
    SEED,
    RunPaths,
    load_prompts,
    load_resume_state,
    run_generation,
)
from utils.model_utils import load_model_and_tokenizer, format_chat_prompt
from utils.data_spec import DatasetSpec
from utils.model_config import ModelConfig, load_model_config

DIRECTIONS_DIR = ROOT / "directions"

# placeholder path until new data is ready
DEFAULT_EVAL_DATA = ROOT / "data" / "mental_health" / "placeholder.csv"

# Following the paper, candidates from the last 20% of layers are discarded.
# directions there tend to encode the next-token output rather than the decision to refuse.
PRUNE_LAST_FRACTION = 0.2
EPS = 1e-8


# ---------------------------------------------------------------------------
# Model structure + hooks
# ---------------------------------------------------------------------------
def get_blocks(model):
    """
    Decoder blocks.
    Assumes a Llama/Qwen2-style layout: model.model.layers[i].{self_attn,mlp}.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise NotImplementedError(
            "configuration not implemented for model"
        )
    return layers


def _map_hidden_states(args, kwargs, fn):
    """Apply fn to a decoder block's hidden_states input, whether passed positionally or by keyword."""
    if args:
        return (fn(args[0]), *args[1:]), kwargs
    return args, {**kwargs, "hidden_states": fn(kwargs["hidden_states"])}


def _project_out(x: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    """x' = x - r_hat (r_hat . x), computed in float32 and cast back."""
    x32 = x.float()
    return (x32 - (x32 @ r_hat).unsqueeze(-1) * r_hat).to(x.dtype)


@contextmanager
def ablation_hooks(model, direction: torch.Tensor):
    """Remove `direction` from the residual stream everywhere.

    The direction is projected out of every block's input, and out of every attention
    and MLP output before it is added back to the stream, so no component can write it
    (including the embeddings, via block 0's input, and the last block, via its outputs).
    """
    r_hat = direction.float()
    r_hat = (r_hat / r_hat.norm()).to(DEVICE)

    def pre_hook(module, args, kwargs):
        return _map_hidden_states(args, kwargs, lambda x: _project_out(x, r_hat))

    def output_hook(module, args, output):
        if isinstance(output, tuple):  # attention returns (output, weights)
            return (_project_out(output[0], r_hat), *output[1:])
        return _project_out(output, r_hat)

    handles = []
    for block in get_blocks(model):
        handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(block.self_attn.register_forward_hook(output_hook))
        handles.append(block.mlp.register_forward_hook(output_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def addition_hook(model, layer: int, vector: torch.Tensor):
    """Add `vector` to the residual stream at the input of one block (at every position)."""
    vector = vector.to(DEVICE)

    def pre_hook(module, args, kwargs):
        return _map_hidden_states(args, kwargs, lambda x: x + vector.to(x.dtype))

    handle = get_blocks(model)[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------
def iter_batches(tokenizer, texts: list[str], batch_size: int):
    """Left-padded batches, with position_ids that ignore the padding."""
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # the chat template already inserts BOS
        ).to(DEVICE)
        yield {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "position_ids": (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0),
        }


def iter_last_token_logits(model, tokenizer, texts, batch_size):
    """Yield next-token logits [B, vocab] (float32) for each batch of prompts."""
    for inputs in iter_batches(tokenizer, texts, batch_size):
        with torch.no_grad():
            hidden = model.model(**inputs).last_hidden_state[:, -1]
            logits = model.lm_head(hidden).float()
        yield logits


def refusal_score(logits: torch.Tensor, refusal_ids: list[int]) -> torch.Tensor:
    """Log-odds that the first generated token is a refusal token (per prompt)."""
    probs = logits.double().softmax(dim=-1)
    p_refuse = probs[:, refusal_ids].sum(dim=-1)
    return torch.log(p_refuse + EPS) - torch.log(1 - p_refuse + EPS)


def refusal_scores(model, tokenizer, texts, refusal_ids, batch_size) -> torch.Tensor:
    return torch.cat(
        [refusal_score(l, refusal_ids).cpu() for l in iter_last_token_logits(model, tokenizer, texts, batch_size)]
    )


def next_token_probs(model, tokenizer, texts, batch_size) -> torch.Tensor:
    return torch.cat(
        [l.double().softmax(dim=-1).cpu() for l in iter_last_token_logits(model, tokenizer, texts, batch_size)]
    )


def mean_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean KL(p || q) over prompts, where p and q are [N, vocab] probabilities."""
    return (p * (torch.log(p + EPS) - torch.log(q + EPS))).sum(dim=-1).mean().item()


def get_refusal_token_ids(tokenizer, cfg: ModelConfig) -> list[int]:
    refusal_token_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in cfg.refusal_tokens]
    print(f"Refusal tokens {cfg.refusal_tokens} -> ids {refusal_token_ids} -> {tokenizer.convert_ids_to_tokens(refusal_token_ids)}")
    return refusal_token_ids


# ---------------------------------------------------------------------------
# Stage 1: find the direction
# ---------------------------------------------------------------------------
def build_splits(model, tokenizer, model_config, refusal_token_ids, n_train: int, n_val: int) -> dict[str, pd.DataFrame]:
    """Shuffle each pool, keep prompts where the model behaves as expected, split train/val.

    Like the paper, this keeps only harmful prompts the model refuses and harmless
    prompts it doesn't (judged by the first generated token), so the direction reflects
    refusal behavior and not just whether a prompt looks harmful.
    """
    need = n_train + n_val
    splits = {}
    for label, keep_refusals in (("harmful", True), ("harmless", False)):
        prompts = load_prompts(DATASETS[label]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        formatted_prompts = [format_chat_prompt(tokenizer, model_config, p) for p in prompts["prompt"]]
        scores = refusal_scores(model, tokenizer, formatted_prompts, refusal_token_ids, model_config.batch_size)
        keep = (scores > 0) if keep_refusals else (scores < 0)
        prompts = prompts[keep.numpy()].reset_index(drop=True)

        expected = "refused" if keep_refusals else "not refused"
        print(f"{label}: {len(prompts)}/{len(formatted_prompts)} prompts {expected} at the first token (need {need})")
        if len(prompts) < need:
            raise ValueError(
                f"Only {len(prompts)} usable {label} prompts but need n_train + n_val = {need}. "
                "Lower --n-train/--n-val, or check that the refusal tokens in the config are right."
            )
        splits[f"{label}_train"] = prompts.iloc[:n_train]
        splits[f"{label}_val"] = prompts.iloc[n_train:need]
    return splits


def mean_activations(model, tokenizer, texts, n_positions: int, batch_size: int) -> torch.Tensor:
    """Mean residual-stream input to every block at the last n_positions tokens -> [n_positions, n_layers, d_model]."""
    n_layers = len(get_blocks(model))
    total = None
    for inputs in tqdm(iter_batches(tokenizer, texts, batch_size), total=-(-len(texts) // batch_size), desc="activations"):
        with torch.no_grad():
            hidden_states = model.model(**inputs, output_hidden_states=True).hidden_states
            # hidden_states[l] is the input to block l (index 0 = embeddings). The last
            # entry (post final norm) is skipped. Left padding means the last
            # positions are real tokens for every prompt in the batch.
            acts = torch.stack([h[:, -n_positions:, :] for h in hidden_states[:n_layers]], dim=1)  # [B, L, K, d]
        batch_sum = acts.float().sum(dim=0).permute(1, 0, 2)  # [K, L, d]
        total = batch_sum if total is None else total + batch_sum
    return (total / len(texts)).cpu()


def score_candidates(model, tokenizer, mean_diffs, val_harmful, val_harmless, refusal_ids, batch_size) -> pd.DataFrame:
    """Score every candidate direction below the layer cutoff on held-out prompts.

    ablated_refusal_score  refusal log-odds on harmful val prompts with the direction
                           ablated everywhere (lower = the ablation removes refusal)
    kl_div                 KL(baseline || ablated) of the first-token distribution on
                           harmless val prompts (lower = ablation leaves behavior alone)
    induced_refusal_score  refusal log-odds on harmless val prompts after ADDING the
                           direction at its layer (higher = the direction induces refusal)
    """
    n_positions, n_layers, _ = mean_diffs.shape
    max_layer = int(n_layers * (1 - PRUNE_LAST_FRACTION))
    baseline_probs = next_token_probs(model, tokenizer, val_harmless, batch_size)

    rows = []
    for pos_idx in range(n_positions):
        position = pos_idx - n_positions
        for layer in tqdm(range(max_layer), desc=f"position {position}"):
            direction = mean_diffs[pos_idx, layer]
            with ablation_hooks(model, direction):
                ablated = refusal_scores(model, tokenizer, val_harmful, refusal_ids, batch_size).mean().item()
                kl = mean_kl(baseline_probs, next_token_probs(model, tokenizer, val_harmless, batch_size))
            with addition_hook(model, layer, direction):
                induced = refusal_scores(model, tokenizer, val_harmless, refusal_ids, batch_size).mean().item()
            rows.append({
                "pos_idx": pos_idx,
                "position": position,
                "layer": layer,
                "diff_norm": direction.norm().item(),
                "ablated_refusal_score": ablated,
                "kl_div": kl,
                "induced_refusal_score": induced,
            })
    return pd.DataFrame(rows)


def select_best(scores: pd.DataFrame, kl_threshold: float, induce_threshold: float) -> pd.Series:
    valid = scores[(scores["kl_div"] < kl_threshold) & (scores["induced_refusal_score"] > induce_threshold)]
    if valid.empty:
        raise ValueError(
            f"No candidate has kl_div < {kl_threshold} and induced_refusal_score > {induce_threshold}. "
            "Inspect candidate_scores.csv and loosen --kl-threshold / --induce-threshold if appropriate."
        )
    return valid.sort_values("ablated_refusal_score").iloc[0]


def find_refusal_direction(model, tokenizer, model_config: ModelConfig, args) -> None:
    output_directory = DIRECTIONS_DIR / model_config.slug
    output_directory.mkdir(parents=True, exist_ok=True)
    refusal_token_ids = get_refusal_token_ids(tokenizer, model_config)

    splits = build_splits(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        refusal_token_ids=refusal_token_ids,
        n_train=args.n_train,
        n_val=args.n_val
    )
    texts = {
        name: [format_chat_prompt(tokenizer, model_config, p) for p in df["prompt"]] for name, df in splits.items()
    }
    (output_directory / "splits.json").write_text(
        json.dumps({name: df["prompt_id"].tolist() for name, df in splits.items()}, indent=2)
    )

    print("Computing mean activations...")
    mean_harmful = mean_activations(model, tokenizer, texts["harmful_train"], args.n_positions, model_config.batch_size)
    mean_harmless = mean_activations(model, tokenizer, texts["harmless_train"], args.n_positions, model_config.batch_size)
    mean_diffs = mean_harmful - mean_harmless  # [n_positions, n_layers, d_model]
    torch.save(mean_diffs, output_directory / "candidate_mean_diffs.pt")

    baseline_harmful = refusal_scores(model, tokenizer, texts["harmful_val"], refusal_token_ids, model_config.batch_size)
    print(f"Mean refusal score on harmful val prompts, no ablation: {baseline_harmful.mean():.2f}")

    print("Scoring candidate directions...")
    scores = score_candidates(
        model, tokenizer, mean_diffs, texts["harmful_val"], texts["harmless_val"], refusal_token_ids, model_config.batch_size
    )
    scores.to_csv(output_directory / "candidate_scores.csv", index=False)
    print(scores.sort_values("ablated_refusal_score").head(5).to_string(index=False))

    best = select_best(scores, args.kl_threshold, args.induce_threshold)
    layer, pos_idx = int(best["layer"]), int(best["pos_idx"])
    torch.save(mean_diffs[pos_idx, layer].clone(), output_directory / "refusal_direction.pt")
    metadata = {
        "model_id": model_config.model_id,
        "layer": layer,
        "position": int(best["position"]),
        "n_layers": mean_diffs.shape[1],
        "scores": {k: float(best[k]) for k in ("ablated_refusal_score", "kl_div", "induced_refusal_score", "diff_norm")},
        "baseline_harmful_val_refusal_score": baseline_harmful.mean().item(),
        "refusal_token_ids": refusal_token_ids,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_positions": args.n_positions,
        "kl_threshold": args.kl_threshold,
        "induce_threshold": args.induce_threshold,
        "seed": SEED,
    }
    (output_directory / "refusal_direction.json").write_text(json.dumps(metadata, indent=2))
    print(f"Selected layer {layer}, position {metadata['position']}. Saved to {output_directory}")


# ---------------------------------------------------------------------------
# Stage 2: generate with the direction ablated
# ---------------------------------------------------------------------------
def run_ablation(model, tokenizer, cfg: ModelConfig, args) -> None:
    ablate = args.condition == "ablated"
    spec = DatasetSpec(
        input_path=args.eval_data, prompt_column=args.prompt_column, output_dir=args.eval_data.parent
    )
    out_dir = spec.output_dir / cfg.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "ablated" if ablate else "unablated"
    paths = RunPaths(
        checkpoint=out_dir / f"{prefix}_completions.csv",
        failed_log=out_dir / f"{prefix}_failed.csv",
        marker=out_dir / f".{prefix}_in_progress.json",
    )

    prompts = load_prompts(spec)
    results, failed_ids = load_resume_state(paths, prompts)
    skip = {r["prompt_id"] for r in results} | set(failed_ids)
    pending = prompts[~prompts["prompt_id"].isin(skip)].reset_index(drop=True)
    print(f"{len(pending)} prompts remaining ({args.condition}).")

    if not pending.empty:
        direction = torch.load(DIRECTIONS_DIR / cfg.slug / "refusal_direction.pt")
        with ablation_hooks(model, direction) if ablate else nullcontext():
            run_generation(model, tokenizer, cfg, pending, prompts, paths, results, failed_ids)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No completions were generated.")
        return
    rate = results_df["looks_like_refusal_flag"].mean()
    print(f"Done. {len(results_df)} completions saved to {paths.checkpoint}")
    print(f"Keyword refusal rate ({cfg.name}, {args.condition}): {rate:.2%}")
    if failed_ids:
        print(f"{len(failed_ids)} prompts failed and were logged to {paths.failed_log}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model key in the config file (e.g. llama2)")
    parser.add_argument("--stage", choices=["find_refusal_direction", "ablate_refusal_direction", "all"], default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    direction = parser.add_argument_group("direction stage")
    direction.add_argument("--n-train", type=int, default=128, help="Prompts per class for the mean activations")
    direction.add_argument("--n-val", type=int, default=32, help="Prompts per class for scoring candidates")
    direction.add_argument("--n-positions", type=int, default=5, help="Number of final token positions to consider")
    direction.add_argument("--kl-threshold", type=float, default=0.1)
    direction.add_argument("--induce-threshold", type=float, default=0.0)

    ablate = parser.add_argument_group("ablate stage")
    ablate.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    ablate.add_argument("--prompt-column", default="goal")
    ablate.add_argument("--condition", choices=["ablated", "baseline"], default="ablated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    model_config = load_model_config(args.config, args.model)

    calculate_refusal_direction = args.stage in ("find_refusal_direction", "all")
    ablate_refusal_direction = args.stage in ("ablate_refusal_direction", "all")

    # Fail fast, before the model loads.
    if ablate_refusal_direction and not args.eval_data.exists():
        raise FileNotFoundError(
            f"{args.eval_data} not found. Point --eval-data at your CSV (needs a '{args.prompt_column}' column), "
            "or use --stage direction for now."
        )
    refusal_direction_tensor_path = DIRECTIONS_DIR / model_config.slug / "refusal_direction.pt"
    if args.stage == "ablate_refusal_direction" and args.condition == "ablated" and not refusal_direction_tensor_path.exists():
        raise FileNotFoundError(f"{refusal_direction_tensor_path} not found. Run --stage direction first.")

    model, tokenizer = load_model_and_tokenizer(model_config)
    if calculate_refusal_direction:
        find_refusal_direction(model=model, tokenizer=tokenizer, model_config=model_config, args=args)
    if ablate_refusal_direction:
        run_ablation(model, tokenizer, model_config, args)


if __name__ == "__main__":
    main()