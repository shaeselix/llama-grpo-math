import argparse
import datetime
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline
from vllm import LLM, SamplingParams  # type: ignore

from src.load_model import load_llama_model


class Evaluator(ABC):
    """Base abstract class for model evaluation on math datasets."""

    @staticmethod
    def extract_answer(text: str, dataset_name: str = "") -> Optional[str]:
        """Extract the final numeric answer from model output with dataset-specific handling."""
        import re

        text_lower = text.lower()

        # First, look for "Therefore, the answer is X" pattern
        therefore_pattern = r"therefore,?\s+the\s+answer\s+is\s+\$?(-?\d+(?:,\d+)*(?:\.\d+)?)"
        therefore_matches = re.findall(therefore_pattern, text_lower)
        if therefore_matches:
            match = therefore_matches[-1]
            return str(match).replace(",", "")

        # Look for explicit statements about the answer
        answer_patterns = [
            r"answer is (?:\$)?(-?\d+(?:,\d+)*(?:\.\d+)?)",
            r"answer: (?:\$)?(-?\d+(?:,\d+)*(?:\.\d+)?)",
            r"result is (?:\$)?(-?\d+(?:,\d+)*(?:\.\d+)?)",
            r"= (?:\$)?(-?\d+(?:,\d+)*)(?:\.\d+)?$",  # Match equals at the end of a line
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                match = matches[-1]
                return str(match).replace(",", "")

        # Look for dollar amounts
        dollar_pattern = r"\$(-?\d+(?:,\d+)*(?:\.\d+)?)"
        dollar_matches = re.findall(dollar_pattern, text)
        if dollar_matches:
            match = dollar_matches[-1]
            return str(match).replace(",", "")

        # Look for the last calculation in the text
        calculation_pattern = (
            r"(-?\d+(?:\.\d+)?)\s*(?:[\+\-\*\/])\s*(?:\$)?(-?\d+(?:\.\d+)?)"
            r"\s*=\s*(?:\$)?(-?\d+(?:\.\d+)?)"
        )
        calculations = re.findall(calculation_pattern, text)
        if calculations:
            # Return the result of the last calculation
            match = calculations[-1][2]
            return str(match).replace(",", "")

        # Look for negative numbers specifically
        negative_pattern = r"(-\d+(?:\.\d+)?)"
        negative_matches = re.findall(negative_pattern, text)
        if negative_matches:
            match = negative_matches[-1]
            return str(match).replace(",", "")

        # General case - find the last numeric value
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return str(numbers[-1])

        return None

    @staticmethod
    def get_dataset_prompt(dataset_name: str, question: str) -> str:
        """Get dataset-specific prompt template."""
        if "gsm8k" in dataset_name.lower():
            return (
                "Solve this step-by-step:\n\n" + question + "\n\n"
                "Let me work through this carefully:\n"
                "1. I'll identify the key quantities and what I need to find.\n"
                "2. For each calculation, I'll write the exact numbers and operations.\n"
                "3. I'll compute each step explicitly, showing the result.\n"
                "4. I'll verify my calculations are correct.\n"
                "5. I'll conclude with 'Therefore, the answer is X.'\n\n"
            )
        elif "math" in dataset_name.lower():
            return "Problem: " + question + "\n\nSolution: "
        return f"Q: {question}\nA:"

    @abstractmethod
    def evaluate(self, dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:
        """
        Evaluate the model on a specific dataset.

        Args:
            dataset_name: Name of the dataset to evaluate on
            split: Dataset split (e.g., "train", "test")
            max_samples: Maximum number of samples to evaluate

        Returns:
            List of evaluation results with details for each sample
        """
        pass

    @staticmethod
    def process_evaluation_results(
        results: List[Dict[str, Any]], dataset_name: str, split: str = "test"
    ) -> List[Dict[str, Any]]:
        """Process and print evaluation results."""
        n_correct = sum(1 for result in results if result.get("is_correct", False))
        n_total = len(results)
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0.0

        print(f"\nAccuracy on {dataset_name}({split}): {acc:.1f}% (based on {n_total} samples)")
        return results


class StandardEvaluator(Evaluator):
    """Evaluator for standard (non-VLLM) inference."""

    def __init__(self, model: Any, tokenizer: Any, device: str = "cuda"):
        """
        Initialize the standard evaluator.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(
        self, dataset_name: str, split: str = "test", max_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """Evaluate model on a dataset using standard pipeline."""
        print(f"Evaluating on {dataset_name} ({split} split), max_samples={max_samples}")
        try:
            ds = load_dataset(dataset_name, "main", split=split)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []

        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,  # Shorter responses for direct answers
            do_sample=False,  # Use greedy decoding for more precise calculations
            temperature=0.0,  # Deterministic generation
            repetition_penalty=1.0,  # No repetition penalty for direct answers
            num_return_sequences=1,
        )

        n_correct = 0
        n_total = 0
        results: List[Dict[str, Any]] = []

        for i, example in enumerate(ds):
            if i >= max_samples:
                break
            try:
                # Attempt to get question/answer fields (varies by dataset)
                question = example.get("question", "") or example.get("problem", "")
                correct_answer = example.get("answer", "") or example.get("solution", "")

                if not question or not correct_answer:
                    print(f"Skipping example {i}: missing question or answer")
                    continue

                # Format prompt using dataset-specific template
                prompt = self.get_dataset_prompt(dataset_name, question)

                output = generator(prompt, num_return_sequences=1)
                model_answer = output[0]["generated_text"]

                # Extract numeric answer
                extracted_answer = self.extract_answer(model_answer, dataset_name)
                correct_answer_numeric = self.extract_answer(str(correct_answer), dataset_name)

                # Compare numeric answers
                is_correct = False
                if extracted_answer and correct_answer_numeric:
                    # Allow for small floating point differences
                    is_correct = abs(float(extracted_answer) - float(correct_answer_numeric)) < 1e-6

                results.append(
                    {
                        "question": question,
                        "model_answer": model_answer,
                        "extracted_answer": extracted_answer,
                        "correct_answer": correct_answer,
                        "correct_answer_numeric": correct_answer_numeric,
                        "is_correct": is_correct,
                    }
                )

                print(f"\n=== Example {i} ===")
                print(f"Question: {question}")
                print(f"Model answer: {model_answer}")
                print(f"Extracted answer: {extracted_answer}")
                print(f"Correct answer: {correct_answer}")
                print(f"Correct? {is_correct}")

                if is_correct:
                    n_correct += 1
                n_total += 1

            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue

        self.process_evaluation_results(results, dataset_name, split)
        return results


class VLLMEvaluator(Evaluator):
    """Evaluator using VLLM for faster inference."""

    def __init__(self, model_dir: str):
        """
        Initialize the VLLM evaluator.

        Args:
            model_dir: Path to the model directory
        """
        self.model_dir = model_dir

    def evaluate(
        self, dataset_name: str, split: str = "test", max_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """Evaluate model on a dataset using VLLM."""

        llm = LLM(model=self.model_dir, dtype="bfloat16", max_model_len=512)
        ds = load_dataset(dataset_name, "main", split=split)

        sampling_params = SamplingParams(
            max_tokens=256,
            temperature=0.0,
            repetition_penalty=1.0,
        )
        questions = []
        correct_answers = []
        prompts = []
        for example in ds:
            question = example["question"]
            questions.append(question)
            prompts.append(self.get_dataset_prompt(dataset_name, question))
            correct_answers.append(example["answer"])
            if len(prompts) >= max_samples:
                break

        n_correct = 0
        n_total = 0
        outputs = llm.generate(prompts, sampling_params)
        results = []
        for i, (output, question, correct_answer) in enumerate(
            zip(outputs, questions, correct_answers, strict=False)
        ):
            generated_text = output.outputs[0].text
            # Extract numeric answer
            extracted_answer = self.extract_answer(generated_text, dataset_name)
            correct_answer_numeric = self.extract_answer(str(correct_answer), dataset_name)
            if extracted_answer is None or correct_answer_numeric is None:
                is_correct = False
            else:
                is_correct = abs(float(extracted_answer) - float(correct_answer_numeric)) < 1e-6
            results.append(
                {
                    "question": question,
                    "model_answer": generated_text,
                    "extracted_answer": extracted_answer,
                    "correct_answer": correct_answer,
                    "correct_answer_numeric": correct_answer_numeric,
                    "is_correct": is_correct,
                }
            )
            print(f"\n=== Example {i} ===")
            print(f"Question: {question}")
            print(f"Model answer: {generated_text}")
            print(f"Extracted answer: {extracted_answer}")
            print(f"Correct answer: {correct_answer}")
            print(f"Correct? {is_correct}")

            if is_correct:
                n_correct += 1
            n_total += 1

        self.process_evaluation_results(results, dataset_name, split)
        return results


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for evaluation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
    )
    return logging.getLogger("evaluator")


def visualize_results(results: List[Dict[str, Any]], dataset_name: str) -> None:
    """Generate visualizations of evaluation results."""
    folder = f"results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(folder, exist_ok=True)
    try:
        df = pd.DataFrame(results)

        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(
            ["Correct", "Incorrect"], [df["is_correct"].sum(), len(df) - df["is_correct"].sum()]
        )
        plt.title(f"Performance on {dataset_name}")
        plt.savefig(f"{folder}/{dataset_name}_results.png")

        # Save detailed results
        df.to_csv(f"{folder}/{dataset_name}_results.csv", index=False)

        print(
            f"Results saved to {folder}/{dataset_name}_results.png"
            f" and {folder}/{dataset_name}_results.csv"
        )
    except ImportError:
        print(
            "Visualization requires pandas and matplotlib. "
            "Install them with 'pip install pandas matplotlib'"
        )
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return


def create_evaluator(args: argparse.Namespace) -> Union[StandardEvaluator, VLLMEvaluator]:
    """
    Factory function to create the appropriate evaluator based on arguments.

    Args:
        args: Command line arguments

    Returns:
        An instance of the appropriate Evaluator subclass
    """
    if args.vllm:
        model_path = args.model_name_or_path
        # If LoRA model, create merged version
        if "/lora/" in args.model_name_or_path:
            model_path = args.model_name_or_path.replace("/lora/", "/merged/")
            if not os.path.exists(model_path):
                # Load model to create merged version
                model, tokenizer = load_llama_model(args.model_name_or_path, args.device)
                os.makedirs(model_path, exist_ok=True)
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                # Free memory
                del model
                del tokenizer
                torch.cuda.empty_cache()
        return VLLMEvaluator(model_path)
    else:
        # Standard evaluator
        model, tokenizer = load_llama_model(args.model_name_or_path, args.device)
        model.eval()
        return StandardEvaluator(model, tokenizer, args.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Llama3.2-3B on math datasets.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Path or ID of the model",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device"
    )
    parser.add_argument("--test_gsm8k", action="store_true", help="Evaluate on GSM8K")
    parser.add_argument("--test_math", action="store_true", help="Evaluate on MATH")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument(
        "--max_samples", type=int, default=5, help="Max number of samples to evaluate"
    )
    parser.add_argument("--vllm", action="store_true", help="Use VLLM for evaluation")
    args = parser.parse_args()

    # Create appropriate evaluator
    evaluator = create_evaluator(args)

    # Evaluate on selected datasets
    if args.test_gsm8k:
        # GSM8K dataset is 'openai/gsm8k' => typically has 'train'/'test' splits
        results = evaluator.evaluate("openai/gsm8k", "test", args.max_samples)
        if args.visualize and results:
            visualize_results(results, "GSM8K")

    if args.test_math:
        # MATH can be 'competition_math' or 'hendrycks_math'
        # We'll do 'hendrycks_math' for demonstration.
        results = evaluator.evaluate("hendrycks_test", "test", args.max_samples)
        if args.visualize and results:
            visualize_results(results, "MATH")


if __name__ == "__main__":
    main()
