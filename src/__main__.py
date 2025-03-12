"""Main entry point for the package when run as a module."""

import argparse
import sys


def main() -> None:
    """Parse command line arguments and dispatch to the appropriate module."""
    parser = argparse.ArgumentParser(
        description="Llama-GRPO-Math: Fine-tuning and evaluation for math reasoning"
    )
    parser.add_argument(
        "command", choices=["train", "evaluate"], help="Command to run (train or evaluate)"
    )

    # Parse just the command for now
    args, remaining = parser.parse_known_args()

    # Remove the command from sys.argv and pass the rest to the appropriate module
    sys.argv = [sys.argv[0]] + remaining

    if args.command == "train":
        # Use absolute imports for consistency
        from src.train import main as train_main

        train_main()
    elif args.command == "evaluate":
        # Use absolute imports for consistency
        from src.evaluate import main as evaluate_main

        evaluate_main()


if __name__ == "__main__":
    main()
