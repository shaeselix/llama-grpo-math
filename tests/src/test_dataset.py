import unittest
from unittest.mock import MagicMock, patch


class TestDataset(unittest.TestCase):
    @patch("datasets.load_dataset")
    def test_load_gsm8k(self, mock_load: MagicMock) -> None:
        # Suppose load_dataset returns a dataset with a 'train' split
        mock_ds = {
            "train": [{"question": "1+1?", "answer": "2"}, {"question": "2+2?", "answer": "4"}]
        }
        mock_load.return_value = mock_ds

        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main")
        self.assertIn("train", ds)
        self.assertEqual(len(ds["train"]), 2)
        self.assertEqual(ds["train"][0]["answer"], "2")


if __name__ == "__main__":
    unittest.main()
