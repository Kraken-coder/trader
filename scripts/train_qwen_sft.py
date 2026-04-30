"""Fine-tune a Qwen causal LM on the Trader JSON chat corpus.

Kaggle T4 x2 ready:
  - 4-bit QLoRA via bitsandbytes (BnbConfig)
  - Multi-GPU via HuggingFace Accelerate / DataParallel-aware Trainer
  - Gradient checkpointing enabled
  - fp16 safe defaults (T4 does not support bf16)

Run from the /kaggle/working/trader directory (after pip-installing deps):

    pip install -q transformers peft datasets accelerate bitsandbytes

    python scripts/train_qwen_sft.py \\
        --dataset sft_corpus/qwen_trader_seed.jsonl \\
        --output-dir /kaggle/working/qwen_trader_sft \\
        --use-bnb-4bit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "peft is required for this training script. "
        "pip install peft bitsandbytes"
    ) from exc


# ---------------------------------------------------------------------------
# All 7 linear projection layers – covers MLP gate + attention
# ---------------------------------------------------------------------------
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def format_example(example: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    return tokens


class ChatDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [{k: v for k, v in f.items() if k != "labels"} for f in features],
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    use_bnb_4bit: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> AutoModelForCausalLM:
    """Load model with optional 4-bit quantisation + LoRA."""

    bnb_config = None
    if use_bnb_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # T4 safe (no bf16)
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if not use_bnb_4bit else None,
        device_map="auto",           # Accelerate spreads across both T4s
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if use_bnb_4bit:
        # Required before PEFT wrapping when using kbit quantisation
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if use_lora:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on Trader seed corpus (Kaggle T4x2 ready)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="sft_corpus/qwen_trader_seed.jsonl")
    parser.add_argument("--output-dir", type=str, default="qwen_trader_sft")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=2,
                        help="Per-device batch size. 2 × 2 GPUs = effective 4 before grad-accum.")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Effective global batch = train_batch_size × n_gpus × grad_accum")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Higher LR is fine with 4-bit QLoRA (adapters only)")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-split", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--use-bnb-4bit", action="store_true", default=True,
                        help="Enable bitsandbytes 4-bit QLoRA (recommended on Kaggle T4)")
    parser.add_argument("--no-bnb-4bit", dest="use_bnb_4bit", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bf16 (only on A100/H100). T4 does NOT support bf16.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen SFT training.")

    n_gpus = torch.cuda.device_count()
    print(f"[SFT] Using {n_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if args.eval_split > 0:
        split = raw_dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
    else:
        split = {"train": raw_dataset, "test": None}

    train_dataset = split["train"].map(
        lambda example: format_example(example, tokenizer, args.max_length),
        remove_columns=split["train"].column_names,
        desc="Tokenising train",
    )
    eval_dataset = None
    if split.get("test") is not None:
        eval_dataset = split["test"].map(
            lambda example: format_example(example, tokenizer, args.max_length),
            remove_columns=split["test"].column_names,
            desc="Tokenising eval",
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(
        model_name=args.model_name,
        use_bnb_4bit=args.use_bnb_4bit,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ------------------------------------------------------------------
    # Training args
    # ------------------------------------------------------------------
    total_update_steps = max(
        1,
        int(
            (len(train_dataset) / max(1, args.train_batch_size * max(1, n_gpus)))
            / max(1, args.grad_accum)
            * max(1.0, args.epochs)
        ),
    )
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else max(1, int(total_update_steps * args.warmup_ratio))
    )

    # T4 does not support bf16 → always fall back to fp16 on T4
    use_bf16 = args.bf16 and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16 and not args.use_bnb_4bit  # 4-bit handles precision internally

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit" if args.use_bnb_4bit else "adamw_torch_fused",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        # Multi-GPU: Trainer automatically detects n_gpus when launched with
        # torchrun or Accelerate; device_map="auto" handles single-process multi-GPU.
    )

    collator = ChatDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"[SFT] Saved fine-tuned model to {out_dir}")


if __name__ == "__main__":
    main()
