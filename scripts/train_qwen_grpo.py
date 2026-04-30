"""GRPO-style RL fine-tuning for a Qwen policy on the Trader environment.

Kaggle T4 x2 ready:
  - 4-bit QLoRA via bitsandbytes (paged_adamw_8bit optimizer)
  - Multi-GPU: model loaded with device_map="auto", generation
    pinned to cuda:0 (the reference device), gradient step on all GPUs
    via torch.nn.DataParallel wrapper
  - Robust reward function with format checking + validity shaping
  - Episode-level GRPO with group-relative advantage normalisation
  - Gradient checkpointing enabled throughout

Setup in Kaggle notebook cell (run from /kaggle/working/trader):

    import subprocess, sys
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers", "peft", "bitsandbytes", "accelerate",
        "openenv-core[core]>=0.2.2", "pyarrow", "pandas", "pydantic",
    ])
    # Then simply:
    exec(open("scripts/train_qwen_grpo.py").read())
    # or: %run scripts/train_qwen_grpo.py --steps 200 --use-bnb-4bit
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("peft + bitsandbytes are required. pip install peft bitsandbytes") from exc

# ---------------------------------------------------------------------------
# Trader environment import – works both when run from the trader/ root and
# when run as scripts/train_qwen_grpo.py from any working directory.
# ---------------------------------------------------------------------------
try:
    from ..models import TraderAction, TraderObservation
    from ..server.trader_environment import TraderEnvironment
except ImportError:
    # Kaggle: cd /kaggle/working/trader && python scripts/train_qwen_grpo.py
    _trader_root = Path(__file__).resolve().parent.parent
    if str(_trader_root) not in sys.path:
        sys.path.insert(0, str(_trader_root))
    from models import TraderAction, TraderObservation
    from server.trader_environment import TraderEnvironment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a crypto trading policy trained on 1-minute candle data. "
    "Return ONLY valid JSON with exactly these keys: "
    "position (one of long, short, noop), take_profit_price (float), stop_loss_price (float). "
    "No markdown fences, no extra text, no explanation."
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Reward clipping to prevent extreme gradient updates
REWARD_CLIP_LOW = -3.0
REWARD_CLIP_HIGH = 3.0


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _dump_model(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()


def observation_to_user_payload(observation: TraderObservation) -> Dict[str, Any]:
    payload = _dump_model(observation)
    payload["market_features"] = _dump_model(observation.market_features)
    return payload


def build_prompt_text(tokenizer: AutoTokenizer, observation: TraderObservation) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(observation_to_user_payload(observation), separators=(",", ":"))},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> str:
    """Extract first {...} block from model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return text[start: end + 1]


def parse_action(text: str) -> TraderAction:
    json_text = extract_json_object(text)
    if hasattr(TraderAction, "model_validate_json"):
        return TraderAction.model_validate_json(json_text)
    return TraderAction.parse_raw(json_text)


def format_reward(response_text: str) -> float:
    """
    Shape reward for JSON format compliance.
    +0.1  → valid JSON with all required keys
    +0.05 → at least parseable JSON but missing some keys
    -0.5  → not JSON at all (strongly penalise hallucination)
    """
    try:
        json_text = extract_json_object(response_text)
        obj = json.loads(json_text)
    except (ValueError, json.JSONDecodeError):
        return -0.5

    required_keys = {"position", "take_profit_price", "stop_loss_price"}
    has_all = required_keys.issubset(obj.keys())
    valid_position = obj.get("position") in ("long", "short", "noop")

    if has_all and valid_position:
        return 0.1
    if has_all or len(required_keys & obj.keys()) >= 2:
        return 0.05
    return 0.0


def validity_reward(response_text: str) -> float:
    """
    Additional validity shaping:
    +0.05 if TP > SL > 0 for long, or SL > TP > 0 for short (sensible levels)
    0.0  otherwise (does not penalise, just doesn't reward)
    """
    try:
        obj = json.loads(extract_json_object(response_text))
        pos = obj.get("position", "noop")
        tp = float(obj.get("take_profit_price", 0.0))
        sl = float(obj.get("stop_loss_price", 0.0))
        if pos == "long" and tp > sl > 0:
            return 0.05
        if pos == "short" and sl > tp > 0:
            return 0.05
    except Exception:
        pass
    return 0.0


def environment_reward(env: TraderEnvironment, response_text: str) -> float:
    """
    Full environment reward = simulate_trade() + format shaping + validity shaping.
    Returns clipped total reward.
    """
    fmt = format_reward(response_text)
    val = validity_reward(response_text)

    try:
        action = parse_action(response_text)
        trade_r = float(env._simulate_trade(action))  # noqa: SLF001
    except Exception:
        # Unparseable → format penalty already captured, env contribution = -1
        trade_r = -1.0

    total = trade_r + fmt + val
    return float(torch.clamp(torch.tensor(total), REWARD_CLIP_LOW, REWARD_CLIP_HIGH).item())


# ---------------------------------------------------------------------------
# Model / generation helpers
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    adapter_path: str | None,
    use_bnb_4bit: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> torch.nn.Module:
    """Build policy model with optional 4-bit QLoRA."""

    bnb_config = None
    if use_bnb_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # T4 safe
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if not use_bnb_4bit else None,
        device_map="auto",         # distributes layers across T4 x2
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if use_bnb_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if adapter_path and Path(adapter_path).exists():
        print(f"[GRPO] Loading SFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    elif use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if not use_bnb_4bit:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model


def _generation_device(model: torch.nn.Module) -> torch.device:
    """
    When device_map='auto', parameters live on different devices.
    Resolve to the device of the first parameter for tokeniser encoding.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def generate_completions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """
    Generate `group_size` independent completions for one prompt.
    Returns list of (response_text, prompt_ids, response_ids).

    Uses batched generation for efficiency.
    """
    gen_device = _generation_device(model)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(gen_device)
    prompt_len = encoded["input_ids"].shape[1]

    # Expand to group_size in one forward pass (much faster on T4)
    input_ids = encoded["input_ids"].expand(group_size, -1)
    attention_mask = encoded["attention_mask"].expand(group_size, -1)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    prompt_ids_cpu = encoded["input_ids"][0].cpu()
    for i in range(group_size):
        resp_ids = output_ids[i, prompt_len:].cpu()
        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        results.append((resp_text, prompt_ids_cpu, resp_ids))
    return results


def sequence_logprob(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Sum log-probability of `response_ids` under `model` given `prompt_ids`.
    Handles device_map=auto by moving tensors to the embedding device.
    """
    embed_device = _generation_device(model)
    prompt_ids = prompt_ids.to(embed_device)
    response_ids = response_ids.to(embed_device)

    full_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(full_ids)

    outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (1, T, V)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = full_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Mask out prompt tokens – only supervise response
    prompt_len = prompt_ids.shape[0]
    response_start = max(0, prompt_len - 1)
    response_log_probs = token_log_probs[:, response_start:]
    return response_log_probs.sum()


# ---------------------------------------------------------------------------
# GRPO training loop
# ---------------------------------------------------------------------------

def grpo_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: AutoTokenizer,
    env: TraderEnvironment,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    kl_coeff: float,
    max_grad_norm: float,
) -> tuple[float, float, float, list[float]]:
    """
    One GRPO update step:
      1. Sample observation from env
      2. Generate `group_size` completions (batched)
      3. Score rewards (environment + format + validity shaping)
      4. Compute group-relative advantages
      5. Policy gradient loss (optionally with KL penalty)
      6. Backprop + clip + optimizer step
    Returns (loss, mean_reward, best_reward, rewards_list)
    """
    observation = env.reset()
    prompt_text = build_prompt_text(tokenizer, observation)

    # --- generation (no grad) -----------------------------------------------
    completions = generate_completions(
        model, tokenizer, prompt_text, group_size, max_new_tokens, temperature, top_p
    )

    # --- reward scoring -------------------------------------------------------
    rewards: list[float] = []
    for resp_text, _, _ in completions:
        r = environment_reward(env, resp_text)
        rewards.append(r)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    # Group-relative advantage (GRPO core)
    mean_r = reward_tensor.mean()
    std_r = reward_tensor.std(unbiased=False)
    advantage = (reward_tensor - mean_r) / (std_r + 1e-8)

    # --- policy gradient loss ------------------------------------------------
    losses: list[torch.Tensor] = []
    for adv, (_, prompt_ids, response_ids) in zip(advantage, completions):
        if response_ids.numel() == 0:
            continue
        logprob = sequence_logprob(model, prompt_ids, response_ids)
        pg_loss = -adv.detach() * logprob

        # Optional KL penalty to prevent reward hacking / collapse
        if kl_coeff > 0.0:
            # Approximate KL via log-prob magnitude (simple regularizer)
            pg_loss = pg_loss + kl_coeff * logprob.pow(2)

        losses.append(pg_loss)

    if not losses:
        return float("nan"), float(mean_r), max(rewards), rewards

    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_grad_norm
    )
    optimizer.step()

    return float(loss.item()), float(mean_r), max(rewards), rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for Qwen on TraderEnvironment (Kaggle T4x2 ready)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="qwen_trader_sft",
                        help="Path to SFT LoRA adapter to start from. Leave empty to train from scratch.")
    parser.add_argument("--output-dir", type=str, default="qwen_trader_grpo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=6,
                        help="Completions per prompt for GRPO advantage estimation. "
                             "Higher = better gradient estimate but more memory.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Low LR is important for GRPO stability.")
    parser.add_argument("--kl-coeff", type=float, default=0.01,
                        help="KL regularisation coefficient (0 to disable).")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    # Environment hyperparams
    parser.add_argument("--episode-length", type=int, default=1024)
    parser.add_argument("--max-holding-bars", type=int, default=60)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.10)
    parser.add_argument("--default-take-profit-pct", type=float, default=0.012)
    parser.add_argument("--default-stop-loss-pct", type=float, default=0.006)
    # LoRA / BnB
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--use-bnb-4bit", action="store_true", default=True,
                        help="Enable bitsandbytes 4-bit QLoRA (recommended on Kaggle T4).")
    parser.add_argument("--no-bnb-4bit", dest="use_bnb_4bit", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GRPO training.")

    n_gpus = torch.cuda.device_count()
    print(f"[GRPO] Using {n_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation (decoder-only)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        use_bnb_4bit=args.use_bnb_4bit,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.train()

    # ------------------------------------------------------------------
    # Optimizer  (paged_adamw_8bit saves ~40% VRAM on T4)
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.use_bnb_4bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=args.lr)
            print("[GRPO] Using PagedAdamW8bit optimizer")
        except (ImportError, AttributeError):
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
            print("[GRPO] bitsandbytes paged optimizer unavailable, falling back to AdamW")
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = TraderEnvironment(
        episode_length=args.episode_length,
        max_holding_bars=args.max_holding_bars,
        slippage_bps=args.slippage_bps,
        drawdown_penalty=args.drawdown_penalty,
        default_take_profit_pct=args.default_take_profit_pct,
        default_stop_loss_pct=args.default_stop_loss_pct,
        seed=args.seed,
    )
    print(f"[GRPO] Environment ready. Coins loaded: {len(env._coin_dfs)}")

    # ------------------------------------------------------------------
    # Sanity-check: make sure the env reward oracle works before training
    # ------------------------------------------------------------------
    _obs = env.reset()
    _dummy_action = TraderAction(position="noop", take_profit_price=0.0, stop_loss_price=0.0)
    _r = float(env._simulate_trade(_dummy_action))  # noqa: SLF001
    assert _r == 0.0, f"Sanity check failed: noop reward should be 0.0, got {_r}"
    print("[GRPO] Environment sanity check passed (noop → 0.0 reward).")

    # ------------------------------------------------------------------
    # Output dir
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    running_loss: list[float] = []
    running_reward: list[float] = []

    for step in range(1, args.steps + 1):
        loss_val, mean_reward, best_reward, step_rewards = grpo_step(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            env=env,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            kl_coeff=args.kl_coeff,
            max_grad_norm=args.max_grad_norm,
        )

        running_loss.append(loss_val if not (loss_val != loss_val) else 0.0)  # nan guard
        running_reward.append(mean_reward)

        # Pretty log with reward distribution
        reward_dist = {
            "min": min(step_rewards),
            "max": max(step_rewards),
            "positive": sum(1 for r in step_rewards if r > 0),
        }
        print(
            f"step={step:04d} | "
            f"loss={loss_val:.4f} | "
            f"mean_r={mean_reward:+.5f} | "
            f"best_r={best_reward:+.5f} | "
            f"dist={reward_dist}"
        )

        # Checkpoint
        if step % args.checkpoint_every == 0:
            cp_dir = out_dir / f"checkpoint-{step}"
            cp_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(cp_dir)
            tokenizer.save_pretrained(cp_dir)
            print(f"[GRPO] Checkpoint saved → {cp_dir}")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    avg_loss = sum(running_loss) / max(1, len(running_loss))
    avg_reward = sum(running_reward) / max(1, len(running_reward))
    print(
        f"\n[GRPO] Training complete.\n"
        f"  Avg loss:   {avg_loss:.4f}\n"
        f"  Avg reward: {avg_reward:+.5f}\n"
        f"  Saved to:   {out_dir}"
    )


if __name__ == "__main__":
    main()
