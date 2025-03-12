import datetime
import glob
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Mock the load_model module
sys.modules["load_model"] = MagicMock()


class TestEvaluate(unittest.TestCase):
    def test_extract_answer(self):
        from src.evaluate import extract_answer

        # Test "Therefore, the answer is X" pattern
        therefore_text = "After calculating all steps, therefore, the answer is 42."
        self.assertEqual(extract_answer(therefore_text), "42")

        # Test with dollar amount
        dollar_text = "The total cost is $123.45."
        self.assertEqual(extract_answer(dollar_text), "123.45")

        # Test with calculation pattern
        calculation_text = "We add 10 + 20 = 30, then multiply by 2: 30 * 2 = 60."
        self.assertEqual(extract_answer(calculation_text), "60")

        # Test with explicit answer patterns
        answer_is_text = "The answer is 75."
        self.assertEqual(extract_answer(answer_is_text), "75")

        answer_colon_text = "Answer: 99"
        self.assertEqual(extract_answer(answer_colon_text), "99")

        # Test with decimal numbers
        decimal_text = "The final value is 3.14159."
        self.assertEqual(extract_answer(decimal_text), "3.14159")

        # Test with negative numbers
        negative_text = "The temperature dropped to -15.5 degrees."
        self.assertEqual(extract_answer(negative_text), "-15.5")

        # Test with no numbers
        no_numbers_text = "There is no numeric answer here."
        self.assertIsNone(extract_answer(no_numbers_text))

    def test_get_dataset_prompt(self):
        from src.evaluate import get_dataset_prompt

        # Test GSM8K prompt
        question = "If John has 5 apples and eats 2, how many does he have left?"
        gsm8k_prompt = get_dataset_prompt("gsm8k", question)
        self.assertIn("Solve this step-by-step:", gsm8k_prompt)
        self.assertIn(question, gsm8k_prompt)
        self.assertIn("Let me work through this carefully:", gsm8k_prompt)
        self.assertIn("I'll identify the key quantities", gsm8k_prompt)
        self.assertIn("I'll conclude with 'Therefore, the answer is X.'", gsm8k_prompt)

        # Test MATH prompt
        math_prompt = get_dataset_prompt("hendrycks_math", question)
        self.assertIn("Problem:", math_prompt)
        self.assertIn("Solution:", math_prompt)

        # Test default prompt
        default_prompt = get_dataset_prompt("unknown_dataset", question)
        self.assertEqual(default_prompt, f"Q: {question}\nA:")

    @pytest.mark.slow
    @patch("src.evaluate.load_dataset")
    @patch("src.evaluate.pipeline")
    def test_evaluate_on_dataset(self, mock_pipeline, mock_load_dataset):
        from src.evaluate import evaluate_on_dataset

        # Mock the dataset
        mock_ds = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3*3?", "answer": "9"},
            {"question": "What is 10-5?", "answer": "5"},
        ]
        mock_load_dataset.return_value = mock_ds

        # Mock the pipeline and its output
        mock_generator = MagicMock()
        mock_pipeline.return_value = mock_generator

        # Set up the generator to return different answers for different questions
        def side_effect(prompt, num_return_sequences):
            if "2+2" in prompt:
                return [{"generated_text": "Let's solve this. 2+2=4. Therefore, the answer is 4."}]
            elif "3*3" in prompt:
                return [{"generated_text": "Let's solve this. 3*3=9. Therefore, the answer is 9."}]
            else:
                return [{"generated_text": "Let's solve this. 10-5=5. Therefore, the answer is 5."}]

        mock_generator.side_effect = side_effect

        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Run the evaluation
        results = evaluate_on_dataset(
            model=mock_model,
            tokenizer=mock_tokenizer,
            dataset_name="gsm8k",
            split="test",
            device="cpu",
            max_samples=3,
        )

        # Verify the results are a list of dictionaries
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

        # Check that each result has the expected keys
        for result in results:
            self.assertIn("question", result)
            self.assertIn("model_answer", result)
            self.assertIn("extracted_answer", result)
            self.assertIn("correct_answer", result)
            self.assertIn("is_correct", result)

        # Verify all answers are correct
        correct_count = sum(1 for result in results if result["is_correct"])
        self.assertEqual(correct_count, 3)

        # Verify the pipeline was created correctly
        mock_pipeline.assert_called_once_with(
            "text-generation",
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=-1,  # -1 for CPU
            max_new_tokens=256,  # Updated parameter
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.0,  # Updated parameter
            num_return_sequences=1,
        )

    @patch("src.evaluate.load_dataset")
    def test_evaluate_on_dataset_error_handling(self, mock_load_dataset):
        from src.evaluate import evaluate_on_dataset

        # Test dataset loading error
        mock_load_dataset.side_effect = Exception("Dataset not found")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        results = evaluate_on_dataset(
            model=mock_model,
            tokenizer=mock_tokenizer,
            dataset_name="nonexistent_dataset",
            split="test",
            device="cpu",
            max_samples=3,
        )

        # Verify error handling - should return a dict with accuracy and samples_evaluated
        self.assertIsInstance(results, dict)
        self.assertEqual(results["accuracy"], 0.0)
        self.assertEqual(results["samples_evaluated"], 0)

    @patch("src.evaluate.logging")
    def test_setup_logging(self, mock_logging):
        from src.evaluate import setup_logging

        # Test with default verbosity (non-verbose)
        logger = setup_logging(verbose=False)

        # Verify logging was configured correctly
        mock_logging.basicConfig.assert_called_once()
        mock_logging.getLogger.assert_called_once_with("evaluator")

        # Verify the log level was set to INFO
        args, kwargs = mock_logging.basicConfig.call_args
        self.assertEqual(kwargs["level"], mock_logging.INFO)

        # Test with verbose=True
        mock_logging.reset_mock()
        logger = setup_logging(verbose=True)
        logger.info("Test message")

        # Verify the log level was set to DEBUG
        args, kwargs = mock_logging.basicConfig.call_args
        self.assertEqual(kwargs["level"], mock_logging.DEBUG)

    def test_visualize_results_with_dependencies(self):
        from src.evaluate import visualize_results

        # Create test results
        results = [
            {"question": "Q1", "model_answer": "A1", "is_correct": True},
            {"question": "Q2", "model_answer": "A2", "is_correct": False},
            {"question": "Q3", "model_answer": "A3", "is_correct": True},
        ]

        # Use a temporary dataset name for testing
        test_dataset_name = "test_dataset_temp"

        # Call the function
        visualize_results(results, test_dataset_name)

        # Get today's date in YYYY-MM-DD format for folder matching
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # Find the created files using glob
        png_files = glob.glob(f"results/{today}_*/{test_dataset_name}_results.png")
        csv_files = glob.glob(f"results/{today}_*/{test_dataset_name}_results.csv")

        self.assertTrue(len(png_files) > 0, f"PNG file for {test_dataset_name} was not created")
        self.assertTrue(len(csv_files) > 0, f"CSV file for {test_dataset_name} was not created")

        # Clean up test files
        try:
            for file in png_files + csv_files:
                os.remove(file)
            # Clean up empty directories
            for dir_path in glob.glob(f"results/{today}_*"):
                if os.path.isdir(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
        except Exception as e:
            print(f"Error during cleanup: {e}")
            pass  # Ignore errors during cleanup


if __name__ == "__main__":
    unittest.main()
