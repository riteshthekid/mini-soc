"""
Mini SOC — GRPO Training Script
================================
Trains a language model to act as a SOC analyst using Group Relative
Policy Optimization (GRPO) from HuggingFace TRL.

The model learns to generate JSON actions that maximize the environment's
reward signal across all 3 SOC tasks.

Usage:
    # From project root with environment server running:
    python -m train.train_grpo

    # Or call from Colab:
    from train.train_grpo import run_training
    run_training(num_steps=200)
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mini_soc.train")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/mini-soc-grpo")
HF_REPO = os.environ.get("HF_REPO", "riteshthekid/mini-soc-grpo")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mini-soc-rl")
SOC_ENV_URL = os.environ.get("SOC_ENV_URL", "https://riteshp30-mini-soc.hf.space")


def run_training(
    num_steps: int = 200,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    num_generations: int = 4,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    num_prompts: int = 60,
    save_steps: int = 50,
    logging_steps: int = 5,
    push_to_hub: bool = False,
    use_wandb: bool = True,
    use_unsloth: bool = False,
    log_file: Optional[str] = None,
) -> str:
    """
    Run GRPO training on the Mini SOC environment.

    Args:
        num_steps: Total number of training steps.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate for AdamW optimizer.
        num_generations: K — group size for GRPO.
        max_new_tokens: Max tokens in model completions.
        temperature: Sampling temperature.
        num_prompts: Number of training prompts to generate.
        save_steps: Save checkpoint every N steps.
        logging_steps: Log metrics every N steps.
        push_to_hub: Push trained model to HuggingFace Hub.
        use_wandb: Enable Weights & Biases logging.
        use_unsloth: Use Unsloth for 2x faster training (Colab).
        log_file: Path to write training log (JSON lines).

    Returns:
        Path to the output directory with the trained model.
    """
    # -----------------------------------------------------------------------
    # Lazy imports — allows module to be imported without heavy deps
    # -----------------------------------------------------------------------
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError(
            "TRL is required for training. Install with: pip install trl>=0.15.0"
        )

    try:
        from peft import LoraConfig
    except ImportError:
        raise ImportError(
            "PEFT is required for LoRA. Install with: pip install peft>=0.14.0"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Ensure reward_wrapper uses the correct environment URL
    import train.reward_wrapper as rw
    rw.SOC_ENV_URL = os.environ.get("SOC_ENV_URL", SOC_ENV_URL)
    from train.reward_wrapper import soc_reward_function, build_soc_dataset

    # -----------------------------------------------------------------------
    # WandB setup
    # -----------------------------------------------------------------------
    report_to = "none"
    if use_wandb:
        try:
            import wandb
            wandb_key = os.environ.get("WANDB_API_KEY", "")
            if wandb_key:
                wandb.login(key=wandb_key)
            wandb.init(
                project=WANDB_PROJECT,
                name=f"grpo-{MODEL_NAME.split('/')[-1]}-{num_steps}steps",
                config={
                    "model": MODEL_NAME,
                    "num_steps": num_steps,
                    "batch_size": batch_size,
                    "lr": learning_rate,
                    "K": num_generations,
                    "temperature": temperature,
                },
            )
            report_to = "wandb"
            logger.info("WandB initialized: project=%s", WANDB_PROJECT)
        except Exception as e:
            logger.warning("WandB init failed (%s), continuing without it.", e)
            report_to = "none"

    # -----------------------------------------------------------------------
    # Model setup
    # -----------------------------------------------------------------------
    logger.info("Loading model: %s", MODEL_NAME)

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=2048,
                dtype=None,  # auto-detect
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            peft_config = None  # Already applied by Unsloth
            logger.info("Unsloth model loaded with built-in LoRA")
        except ImportError:
            logger.warning("Unsloth not available, falling back to standard loading")
            use_unsloth = False

    if not use_unsloth:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Model loaded successfully")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    logger.info("Building training dataset (%d prompts)...", num_prompts)
    dataset = build_soc_dataset(num_samples=num_prompts)
    logger.info("Dataset built: %d samples", len(dataset))

    # -----------------------------------------------------------------------
    # GRPO Config
    # -----------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        logging_steps=logging_steps,
        save_steps=save_steps,
        report_to=report_to,
        bf16=True,
        remove_unused_columns=False,
        log_level="info",
    )

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    logger.info("Initializing GRPOTrainer...")

    trainer_kwargs = {
        "model": model,
        "reward_funcs": soc_reward_function,
        "args": grpo_config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    logger.info("Starting GRPO training (%d steps)...", num_steps)
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Model saved to %s", OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Save training log
    # -----------------------------------------------------------------------
    log_path = log_file or os.path.join(OUTPUT_DIR, "training_log.jsonl")
    _save_training_log(trainer, log_path)
    logger.info("Training log saved to %s", log_path)

    # -----------------------------------------------------------------------
    # Push to Hub
    # -----------------------------------------------------------------------
    if push_to_hub:
        try:
            hf_token = os.environ.get("HF_TOKEN", "")
            trainer.push_to_hub(HF_REPO, token=hf_token if hf_token else None)
            logger.info("Model pushed to HuggingFace Hub: %s", HF_REPO)
        except Exception as e:
            logger.error("Failed to push to Hub: %s", e)

    return OUTPUT_DIR


def _save_training_log(trainer, log_path: str) -> None:
    """Save training metrics as JSON lines for plotting."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history if hasattr(trainer, "state") else []

    with open(log_path, "w") as f:
        for entry in log_history:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train Mini SOC GRPO agent")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--K", type=int, default=4, help="GRPO group size")
    parser.add_argument("--prompts", type=int, default=60, help="Training prompts")
    parser.add_argument("--push", action="store_true", help="Push to HF Hub")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF Hub repo to push model to")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--unsloth", action="store_true", help="Use Unsloth")
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument("--env-url", type=str, default=None, help="SOC environment URL")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for model")
    args = parser.parse_args()

    if args.model:
        MODEL_NAME = args.model
    if args.env_url:
        os.environ["SOC_ENV_URL"] = args.env_url
        SOC_ENV_URL = args.env_url
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    if args.push_to_hub:
        HF_REPO = args.push_to_hub

    output = run_training(
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_generations=args.K,
        num_prompts=args.prompts,
        push_to_hub=args.push or bool(args.push_to_hub),
        use_wandb=not args.no_wandb,
        use_unsloth=args.unsloth,
    )
    print(f"\nTraining complete. Output: {output}")
