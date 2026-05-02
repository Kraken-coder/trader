"""GRPO-style training for a Qwen policy on the Trader environment.

This script is self-contained and does not depend on TRL.
It uses the live TraderEnvironment reward oracle, samples multiple completions
per prompt, computes group-relative advantages, and updates the policy with a
simple policy-gradient objective.

The intent is to start from the SFT-trained Qwen model or adapter and improve
it further on the trading environment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("peft is required for GRPO training. Install the sft extras first.") from exc

try:
    from ..models import TraderAction, TraderObservation
    from ..server.trader_environment import TraderEnvironment
except ImportError:
    from models import TraderAction, TraderObservation
    from server.trader_environment import TraderEnvironment


SYSTEM_PROMPT = (
    "You are a crypto trading policy trained on 1-minute candle data. "
    "Output exactly one raw JSON object and nothing else. "
    "Do not include markdown, code fences, commentary, or extra keys. "
    "Required schema: {\"position\":\"long|short|noop\",\"take_profit_price\":number,\"stop_loss_price\":number}. "
    "Rules: if position is noop, take_profit_price and stop_loss_price must both be 0.0. "
    "If position is long or short, both take_profit_price and stop_loss_price must be positive numbers. "
    "Prefer taking trades when signal exists; avoid repeating noop by default. "
    "Your objective is to maximize profit."
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_PROMPT_LEN = 256


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


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


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return text[start : end + 1]


def parse_action(text: str) -> TraderAction:
    json_text = extract_json_object(text)
    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise ValueError("Action payload must be a JSON object")

    # Normalize common model outputs so schema validation can succeed.
    payload["position"] = str(payload.get("position", "")).lower()
    for key in ("take_profit_price", "stop_loss_price"):
        value = payload.get(key, 0.0)
        if value is None:
            payload[key] = 0.0
            continue
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"", "na", "n/a", "nan", "null", "none"}:
                payload[key] = 0.0
                continue
        try:
            payload[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for {key}: {value!r}") from exc

    if hasattr(TraderAction, "model_validate"):
        return TraderAction.model_validate(payload)
    return TraderAction.parse_obj(payload)


def reward_stage(response_text: str) -> str:
    """Return parsing stage for diagnostics."""
    try:
        json_text = extract_json_object(response_text)
    except Exception:
        return "no_json"
    try:
        payload = json.loads(json_text)
    except Exception:
        return "bad_json"
    if not isinstance(payload, dict):
        return "bad_json"
    required = {"position", "take_profit_price", "stop_loss_price"}
    if not required.issubset(payload.keys()):
        return "missing_keys"
    try:
        parse_action(response_text)
    except Exception:
        return "schema_fail"
    return "schema_ok"


def score_action(env: TraderEnvironment, action: TraderAction) -> float:
    return float(env._simulate_trade(action))  # noqa: SLF001 - reward oracle is the environment itself


def shaped_reward(env: TraderEnvironment, response_text: str) -> float:
    """Dense reward to avoid flat gradients when parse failures dominate."""
    # Stage 0: no JSON-like structure at all.
    reward = -1.0
    try:
        json_text = extract_json_object(response_text)
        reward = -0.4
    except Exception:
        return reward

    # Stage 1: JSON object is syntactically valid and has expected keys.
    try:
        payload = json.loads(json_text)
        if isinstance(payload, dict):
            required = {"position", "take_profit_price", "stop_loss_price"}
            if required.issubset(payload.keys()):
                reward = -0.1
    except Exception:
        return reward

    # Stage 2: full schema validation + trading reward.
    try:
        action = parse_action(response_text)
    except Exception:
        return reward

    position = getattr(action, "position", None)
    if hasattr(position, "value"):
        position = position.value
    position = str(position).lower()

    take_profit = getattr(action, "take_profit_price", None)
    stop_loss = getattr(action, "stop_loss_price", None)

    if position == "noop":
        # Small inactivity penalty to avoid converging to noop-only behavior.
        reward = -0.05
        tp_num = float(take_profit or 0.0)
        sl_num = float(stop_loss or 0.0)
        if tp_num > 0.0 or sl_num > 0.0:
            reward -= 0.25
    else:
        reward = score_action(env, action) + 0.2  # incentive for valid active trades
        tp_valid = isinstance(take_profit, (int, float)) and float(take_profit) > 0.0
        sl_valid = isinstance(stop_loss, (int, float)) and float(stop_loss) > 0.0
        if not (tp_valid and sl_valid):
            reward -= 0.25

    return float(reward)


def build_model(
    model_name: str,
    adapter_path: str | None,
    use_lora: bool,
    bf16: bool,
) -> torch.nn.Module:
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    elif use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=TARGET_MODULES,
        )
        model = get_peft_model(model, lora_config)

    return model


def generate_completion(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    device = _model_device(model)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            min_new_tokens=16,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    response_ids = output_ids[0, prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text, encoded["input_ids"][0], response_ids


def sequence_logprob(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    max_prompt_len: int,
) -> torch.Tensor:
    device = _model_device(model)
    prompt_ids = prompt_ids[-max_prompt_len:].to(device)
    response_ids = response_ids.to(device)
    full_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(full_ids, device=device)

    outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = full_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    prompt_len = prompt_ids.shape[0]
    response_start = max(0, prompt_len - 1)
    response_log_probs = token_log_probs[:, response_start:]
    return response_log_probs.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO-style training for Qwen on TraderEnvironment")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="qwen_trader_sft")
    parser.add_argument("--output-dir", type=str, default="qwen_trader_grpo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=MAX_PROMPT_LEN,
        help="Max prompt tokens kept when computing logprobs.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--resample-non-noop-attempts",
        type=int,
        default=4,
        help="Number of generation attempts per sample to avoid noop-only batches.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length", type=int, default=1024)
    parser.add_argument("--max-holding-bars", type=int, default=60)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.10)
    parser.add_argument("--default-take-profit-pct", type=float, default=0.012)
    parser.add_argument("--default-stop-loss-pct", type=float, default=0.006)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GRPO training. Use a CUDA-enabled PyTorch build on an NVIDIA GPU.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = build_model(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        use_lora=args.use_lora,
        bf16=args.bf16,
    ).to(device)

    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    env = TraderEnvironment(
        episode_length=args.episode_length,
        max_holding_bars=args.max_holding_bars,
        slippage_bps=args.slippage_bps,
        drawdown_penalty=args.drawdown_penalty,
        default_take_profit_pct=args.default_take_profit_pct,
        default_stop_loss_pct=args.default_stop_loss_pct,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        observation = env.reset()
        prompt_text = build_prompt_text(tokenizer, observation)

        completions: list[str] = []
        prompt_ids_list: list[torch.Tensor] = []
        response_ids_list: list[torch.Tensor] = []
        rewards: list[float] = []
        stage_counts: dict[str, int] = {
            "no_json": 0,
            "bad_json": 0,
            "missing_keys": 0,
            "schema_fail": 0,
            "schema_ok": 0,
        }
        noop_count = 0
        non_noop_count = 0

        for _ in range(args.group_size):
            best_triplet: tuple[str, torch.Tensor, torch.Tensor] | None = None
            non_noop_found = False

            for attempt_idx in range(max(1, args.resample_non_noop_attempts)):
                # Gradually increase temperature on retries to escape deterministic noop loops.
                sample_temperature = min(1.35, args.temperature + 0.08 * attempt_idx)
                response_text, prompt_ids, response_ids = generate_completion(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=sample_temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                )
                best_triplet = (response_text, prompt_ids, response_ids)

                try:
                    parsed = parse_action(response_text)
                    pos = str(getattr(parsed, "position", "")).lower()
                    if pos != "noop":
                        non_noop_found = True
                        break
                except Exception:
                    # Keep retrying; malformed outputs are still useful fallback if needed.
                    pass

            assert best_triplet is not None
            response_text, prompt_ids, response_ids = best_triplet
            stage = reward_stage(response_text)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            if stage == "schema_ok":
                try:
                    action = parse_action(response_text)
                    if str(action.position).lower() == "noop":
                        noop_count += 1
                    else:
                        non_noop_count += 1
                except Exception:
                    # Keep training robust even if diagnostics parsing fails unexpectedly.
                    pass
            reward = shaped_reward(env, response_text)
            if not non_noop_found and stage == "schema_ok":
                # Extra penalty when repeated attempts still yielded only noop.
                reward -= 0.10

            completions.append(response_text)
            prompt_ids_list.append(prompt_ids)
            response_ids_list.append(response_ids)
            rewards.append(float(reward))

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        std = reward_tensor.std(unbiased=False)
        if std.item() < 1e-6:
            n_rewards = reward_tensor.numel()
            if n_rewards <= 1:
                print(
                    f"step={step} reward variance collapsed (mean={reward_tensor.mean().item():.4f}); "
                    "skipping optimizer step"
                )
                continue
            # Centered tie-breaker to preserve a meaningful gradient signal when
            # sampled rewards are identical. Use stronger scale if batch is all-noop.
            scale = 5e-2 if noop_count == args.group_size else 1e-2
            tie_breaker = torch.linspace(-1.0, 1.0, steps=n_rewards, device=device) * scale
            advantage = tie_breaker - tie_breaker.mean()
            print(
                f"step={step} reward variance collapsed (mean={reward_tensor.mean().item():.4f}); "
                f"using tie-breaker advantages (scale={scale:.4f})"
            )
        else:
            advantage = (reward_tensor - reward_tensor.mean()) / (std + 1e-6)

        losses: list[torch.Tensor] = []
        for adv, prompt_ids, response_ids in zip(advantage, prompt_ids_list, response_ids_list):
            if response_ids.numel() == 0:
                continue
            logprob = sequence_logprob(model, prompt_ids, response_ids, args.max_prompt_len)
            losses.append(-adv.detach() * logprob)

        if not losses:
            continue

        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        best_reward = max(rewards) if rewards else float("nan")
        mean_reward = sum(rewards) / max(1, len(rewards))
        reward_neg1 = sum(1 for r in rewards if r <= -0.999)
        reward_neg04 = sum(1 for r in rewards if -0.41 <= r <= -0.39)
        reward_neg01 = sum(1 for r in rewards if -0.11 <= r <= -0.09)
        reward_nonneg = sum(1 for r in rewards if r >= 0.0)
        print(f"step={step} loss={loss.item():.4f} mean_reward={mean_reward:.6f} best_reward={best_reward:.6f}")
        print(
            "  diag:"
            f" stages={stage_counts}"
            f" noop={noop_count}/{args.group_size}"
            f" non_noop={non_noop_count}/{args.group_size}"
            f" rewards[-1]={reward_neg1}"
            f" rewards[-0.4]={reward_neg04}"
            f" rewards[-0.1]={reward_neg01}"
            f" rewards[>=0]={reward_nonneg}"
        )

        if step % 25 == 0:
            checkpoint_dir = out_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved GRPO-tuned model to {out_dir}")


if __name__ == "__main__":
    main()
