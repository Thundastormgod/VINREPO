"""Entrypoint for running the ZenML training pipeline."""

import argparse

from pipelines.training_pipeline import training_pipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the ZenML training pipeline.")
    parser.add_argument("--data-dir", default="./data", help="Path to the training data root.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    """Run the ZenML pipeline end-to-end."""
    args = parse_args()
    training_pipeline.with_options(
        settings={"experiment_tracker.mlflow": {"experiment_name": "Model Development Example"}}
    )(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
