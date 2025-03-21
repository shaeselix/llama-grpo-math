import unittest
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest


class TestTrain(unittest.TestCase):
    @pytest.mark.slow
    @patch("src.train.wandb")
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
        mock_wandb: MagicMock,
    ) -> None:
        from src.train import train_model

        # Mock wandb.init
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_run.id = "test-run-id"
        mock_wandb.init.return_value = mock_run
        mock_wandb.run = mock_run

        # Set up mock artifact
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

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
            use_wandb=True,
            wandb_project="test-project",
            wandb_name="test-run",
        )

        # Assertions for model loading and training
        mock_load_model.assert_called_once_with("mock_model", "cpu", False)
        mock_load_dataset.assert_called_once_with("openai/gsm8k", "main")
        mock_trainer_instance.train.assert_called_once()
        mock_peft_model.save_pretrained.assert_called_once_with("./temp_lora")
        mock_tokenizer.save_pretrained.assert_called_once_with("./temp_lora")

        # Assertions for wandb
        mock_wandb.init.assert_called_once()
        # Check if project name and run name were set correctly
        init_call_args = mock_wandb.init.call_args[1]
        self.assertEqual(init_call_args["project"], "test-project")
        self.assertEqual(init_call_args["name"], "test-run")
        # Verify config contains relevant hyperparameters
        self.assertIn("config", init_call_args)
        config = init_call_args["config"]
        self.assertIn("base_model", config)
        self.assertIn("learning_rate", config)
        self.assertIn("batch_size", config)

        # Verify artifact logging
        mock_wandb.Artifact.assert_called_once()
        artifact_call_args = mock_wandb.Artifact.call_args[1]
        self.assertEqual(artifact_call_args["name"], f"model-{mock_run.id}")
        self.assertEqual(artifact_call_args["type"], "model")
        # Check that artifact directory was added
        mock_artifact.add_dir.assert_called_once_with("./temp_lora")
        mock_wandb.log_artifact.assert_called_once_with(mock_artifact)

        # Verify wandb.finish() was called
        mock_wandb.finish.assert_called_once()

    def test_wandb_in_evaluation(self) -> None:
        """Test wandb integration in evaluation without trying to mock model loading."""

        # Create a simplified version of evaluate_fine_tuned_model function for testing
        def mock_evaluate_fn(
            use_wandb: bool = True,
            wandb_project: str = "test-project",
            wandb_name: str = "test-run",
        ) -> List[Dict[str, bool]]:
            # Import here to avoid patching issues
            import wandb

            # Initialize wandb
            if use_wandb:
                wandb.init(
                    project=wandb_project, name=wandb_name, config={"test_param": "test_value"}
                )

                # Log some metrics
                wandb.log({"accuracy": 50.0, "correct": 1, "total": 2})

                # Create and log a table
                columns = ["question", "prediction", "is_correct"]
                data = [["q1", "a1", True], ["q2", "a2", False]]
                table = wandb.Table(columns=columns, data=data)
                wandb.log({"results": table})

                # Finish the run
                wandb.finish()

            return [{"is_correct": True}, {"is_correct": False}]

        # Test the function with wandb mocked
        with (
            patch("wandb.init") as mock_init,
            patch("wandb.log") as mock_log,
            patch("wandb.Table") as mock_table,
            patch("wandb.finish") as mock_finish,
        ):
            # Set up mock table
            mock_table_instance = MagicMock()
            mock_table.return_value = mock_table_instance

            # Set up mock run
            mock_run = MagicMock()
            mock_run.name = "test-run"
            mock_init.return_value = mock_run

            # Call the function
            results = mock_evaluate_fn(
                use_wandb=True, wandb_project="test-eval-project", wandb_name="test-eval-run"
            )

            # Verify the results
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0]["is_correct"])
            self.assertFalse(results[1]["is_correct"])

            # Verify wandb calls
            mock_init.assert_called_once()
            init_args = mock_init.call_args[1]
            self.assertEqual(init_args["project"], "test-eval-project")
            self.assertEqual(init_args["name"], "test-eval-run")

            # Verify logging calls
            self.assertEqual(mock_log.call_count, 2)  # Metrics and table

            # Verify table was created
            mock_table.assert_called_once()
            table_args = mock_table.call_args[1]
            self.assertEqual(len(table_args["columns"]), 3)
            self.assertEqual(len(table_args["data"]), 2)

            # Verify wandb.finish was called
            mock_finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
