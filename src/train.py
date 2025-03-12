import argparse
from typing import Any, Dict, List, Union, cast

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.load_model import load_llama_model


def preprocess_function(
    examples: Dict[str, List[str]], tokenizer: Any, max_length: int = 1024
) -> Dict[str, List[Any]]:
    """
    Preprocess examples for training by formatting them as structured step-by-step
    reasoning problems.

    Args:
        examples: Dictionary containing question and answer lists
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length

    Returns:
        Tokenized examples
    """
    questions = examples["question"]
    answers = examples["answer"]

    formatted_texts = []
    for q, a in zip(questions, answers, strict=False):
        # Format each example with clear step-by-step structure
        prompt = (
            "Solve this step-by-step:\n\n" + q + "\n\n"
            "Let me work through this carefully:\n"
            "1. I'll identify the key quantities and what I need to find.\n"
            "2. For each calculation, I'll write the exact numbers and operations.\n"
            "3. I'll compute each step explicitly, showing the result.\n"
            "4. I'll verify my calculations are correct.\n"
            "5. I'll conclude with 'Therefore, the answer is X.'\n\n"
        )

        # Format the answer to ensure it includes all reasoning steps
        # GSM8k answers typically have calculation steps followed by the final answer
        if not a.strip().startswith("Therefore"):
            a = a.strip() + "\nTherefore, the answer is " + extract_final_answer(a) + "."

        formatted_text = prompt + a
        formatted_texts.append(formatted_text)

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        print("No pad token found. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Add padding tokens and attention masks
    tokenized = tokenizer(
        formatted_texts, truncation=True, max_length=max_length, padding="max_length"
    )

    # Add labels for causal language modeling (same as input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return cast(Dict[str, List[Any]], tokenized)


def extract_final_answer(answer_text: str) -> str:
    """Extract the final numeric answer from the GSM8k answer format."""
    import re

    # GSM8k answers typically end with #### followed by the answer
    pattern = r"####\s*(\d+(?:\.\d+)?)"
    match = re.search(pattern, answer_text)

    if match:
        return str(match.group(1))

    # If no match, try to find the last number in the text
    numbers = re.findall(r"\d+(?:\.\d+)?", answer_text)
    if numbers:
        return str(numbers[-1])

    return ""


def train_model(
    base_model: str,
    output_dir: str,
    num_train_epochs: float = 3.0,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dataset_name: str = "openai/gsm8k",
    dataset_split: str = "main",
    sample_size: int = 1000,
    lora_r: int = 16,
    lora_alpha: int = 32,
    disable_quantization: bool = False,
) -> None:
    """
    Train a model using LoRA fine-tuning.

    Args:
        base_model: Name or path of the base model
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
        batch_size: Batch size for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        learning_rate: Learning rate
        device: Device to use (cuda or cpu)
        dataset_name: Name of the dataset to use
        dataset_split: Split of the dataset to use
        sample_size: Number of samples to use for training
        lora_r: Rank of LoRA adapters
        lora_alpha: Alpha parameter for LoRA
        disable_quantization: If True, disable quantization to avoid bitsandbytes dependency
    """
    # Load base model
    model, tokenizer = load_llama_model(base_model, device, disable_quantization)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        # For Llama models, target these modules
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        if "llama" in base_model.lower()
        else None,
        bias="none",
        fan_in_fan_out=False,
    )

    print("Applying LoRA configuration...")
    peft_model = get_peft_model(model, lora_config)
    print("LoRA configuration applied successfully")

    # Load dataset for training
    ds = load_dataset(dataset_name, dataset_split)

    # Use more examples for training
    sample_size = min(sample_size, len(ds["train"]))
    train_ds = ds["train"].shuffle(seed=42).select(range(sample_size))

    # Create a small validation set
    val_size = min(int(sample_size * 0.1), 100)
    if len(ds["train"]) > sample_size + val_size:
        val_ds = ds["train"].shuffle(seed=42).select(range(sample_size, sample_size + val_size))
    else:
        # If not enough examples, use 10% of training data
        split = train_ds.train_test_split(test_size=0.1)
        train_ds = split["train"]
        val_ds = split["test"]

    # Preprocess data
    def preprocess_fn(examples: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        return preprocess_function(examples, tokenizer, max_length=1024)

    train_ds = train_ds.map(preprocess_fn, batched=True)
    val_ds = val_ds.map(preprocess_fn, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # for causal LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # Larger eval batch size
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",  # Cosine schedule with warmup
        warmup_ratio=0.05,  # Warm up for 5% of training
        weight_decay=0.01,  # L2 regularization
        fp16=False,  # Disable fp16 to avoid unscaling issues
        logging_steps=1,  # Log every step for small test runs
        evaluation_strategy="steps",  # Evaluate during training
        eval_steps=5,  # Evaluate more frequently for small test runs
        save_steps=10,
        save_total_limit=1,  # Only save the best model to save disk space
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="eval_loss",  # Use eval loss as metric
        greater_is_better=False,  # Lower loss is better
        report_to=[],  # avoid wandb or others by default
        # For quick test runs, add these options:
        max_steps=10 if num_train_epochs < 0.1 else -1,  # Limit steps for tiny test runs
        save_safetensors=False,  # Skip safetensors for testing
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # Add validation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...\n")
    trainer.train()

    # Save the final model (LoRA weights + base config)
    print("Training complete. Saving model...")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")


def evaluate_fine_tuned_model(
    model_path: str, dataset_name: str = "openai/gsm8k", num_samples: int = 50
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate the fine-tuned model on GSM8k test set.

    Args:
        model_path: Path to the fine-tuned model
        dataset_name: Name of the dataset to evaluate on
        num_samples: Number of samples to evaluate
    """
    import torch

    from src.evaluate import evaluate_on_dataset
    from src.load_model import load_llama_model

    print(f"Evaluating fine-tuned model {model_path} on {dataset_name}...")

    # Load the fine-tuned model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_llama_model(model_path, device)

    # Run evaluation on the test set
    results = evaluate_on_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split="test",
        device=device,
        max_samples=num_samples,
    )

    # Calculate and report accuracy
    if isinstance(results, list):
        correct = sum(1 for res in results if res.get("is_correct", False))
        total = len(results)
        accuracy = (correct / total) * 100 if total > 0 else 0

        print(f"Evaluation results on {dataset_name} test set:")
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    else:
        print(f"Evaluation failed: {results}")

    # Return results
    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Llama3.2-3B on GSM8K")
    parser.add_argument(
        "--base_model", type=str, default="meta-llama/Llama-3.2-3B", help="Base HF model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_lora_finetuned",
        help="Where to save the fine-tuned model",
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=3.0, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="openai/gsm8k", help="Dataset to use for training"
    )
    parser.add_argument("--dataset_split", type=str, default="main", help="Dataset split to use")
    parser.add_argument(
        "--sample_size", type=int, default=1000, help="Number of samples to use for training"
    )
    parser.add_argument(
        "--evaluate_after_training", action="store_true", help="Evaluate model after training"
    )
    parser.add_argument(
        "--eval_samples", type=int, default=50, help="Number of samples to use for evaluation"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="Rank of LoRA adapters")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA")
    parser.add_argument(
        "--disable_quantization",
        action="store_true",
        help="Disable quantization to avoid bitsandbytes dependency",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the training script."""
    args = parse_args()

    train_model(
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        sample_size=args.sample_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        disable_quantization=args.disable_quantization,
    )

    # Evaluate the fine-tuned model if requested
    if args.evaluate_after_training:
        evaluate_fine_tuned_model(
            model_path=args.output_dir,
            dataset_name=args.dataset_name,
            num_samples=args.eval_samples,
        )


if __name__ == "__main__":
    main()
