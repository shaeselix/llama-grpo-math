import unittest
from unittest.mock import MagicMock, patch


class TestModel(unittest.TestCase):
    @patch("src.load_model.AutoModelForCausalLM")
    @patch("src.load_model.AutoTokenizer")
    def test_load_llama_model(self, mock_tokenizer: MagicMock, mock_model: MagicMock) -> None:
        from src.load_model import load_llama_model

        # Setup mock
        mock_tok_instance = MagicMock()
        mock_model_instance = MagicMock()
        # Configure the to() method to return the same mock instance
        mock_model_instance.to.return_value = mock_model_instance
        # Set up the device property
        device_mock = MagicMock()
        device_mock.type = "cpu"
        mock_model_instance.device = device_mock

        mock_tokenizer.from_pretrained.return_value = mock_tok_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        model, tokenizer = load_llama_model("meta-llama/Llama-3.2-3B", device="cpu")

        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        self.assertIs(model, mock_model_instance)
        self.assertIs(tokenizer, mock_tok_instance)
        self.assertEqual(model.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
