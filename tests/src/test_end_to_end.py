import unittest

import pytest
import torch


class TestEndToEnd(unittest.TestCase):
    @pytest.mark.slow
    def test_inference_on_gsm8k_first_question(self) -> None:
        """
        This test loads a publicly available model (gpt2-medium)
        and runs inference on the first sample of GSM8K's test set.
        """
        # Because this test might be slow or require GPU, we can optionally skip if not available.
        if not torch.cuda.is_available():
            self.skipTest("Skipping end-to-end test on CPU.")

        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        # Use a publicly available model instead of the gated Llama model
        model_name = "gpt2-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

        ds = load_dataset("openai/gsm8k", "main", split="test")
        first_sample = ds[0]
        question = first_sample["question"]

        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0,
            max_length=200,
            do_sample=False,
        )
        prompt = f"Q: {question}\nA:"
        output = gen_pipeline(prompt, num_return_sequences=1)
        model_answer = output[0]["generated_text"]

        self.assertIsInstance(model_answer, str)
        self.assertTrue(len(model_answer) > 0)


if __name__ == "__main__":
    unittest.main()
