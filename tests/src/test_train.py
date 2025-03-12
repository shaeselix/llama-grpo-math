import unittest
from unittest.mock import MagicMock, patch

import pytest


class TestTrain(unittest.TestCase):
    @pytest.mark.slow
    @patch("src.train.Trainer")
    @patch("src.train.load_dataset")
    @patch("src.train.load_llama_model")
    @patch("src.train.get_peft_model")
    def test_lora_finetuning(
        self,
        mock_get_peft_model: MagicMock,
        mock_load_model: MagicMock,
        mock_load_dataset: MagicMock,
        mock_trainer: MagicMock,
    ) -> None:
        from src.train import train_model

        # Mock the model/tokenizer from load_llama_model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Mock peft model
        mock_peft_model = MagicMock()
        mock_get_peft_model.return_value = mock_peft_model

        # Mock dataset with a train split that has a map method
        mock_train_ds = MagicMock()
        mock_train_ds.map.return_value = mock_train_ds

        mock_ds = {"train": MagicMock()}
        mock_ds["train"].shuffle.return_value.select.return_value = mock_train_ds
        mock_load_dataset.return_value = mock_ds

        # Mock the Trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Call train_model directly with test parameters
        train_model(
            base_model="mock_model",
            output_dir="./temp_lora",
            num_train_epochs=0.1,
            batch_size=1,
            learning_rate=2e-4,
            device="cpu",
            sample_size=10,
        )

        # Assertions
        mock_load_model.assert_called_once_with("mock_model", "cpu")
        mock_load_dataset.assert_called_once_with("openai/gsm8k", "main")
        mock_trainer_instance.train.assert_called_once()
        mock_peft_model.save_pretrained.assert_called_once_with("./temp_lora")
        mock_tokenizer.save_pretrained.assert_called_once_with("./temp_lora")


if __name__ == "__main__":
    unittest.main()
