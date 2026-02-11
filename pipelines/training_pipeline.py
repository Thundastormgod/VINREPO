"""ZenML pipeline wiring data loading, training, and evaluation steps."""

from steps.data_loader import load_data_step
from steps.evaluate import evaluate_step
from steps.training import train_step
from zenml import pipeline


@pipeline  # type: ignore[untyped-decorator]
def training_pipeline(
    data_dir: str = "./data",
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> dict[str, float]:
    """Run the training workflow end-to-end.

    Args:
        data_dir: Root directory containing train/test data.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        learning_rate: Learning rate for the optimizer.
        seed: Random seed for reproducibility.

    Returns:
        dict[str, float]: Training metrics from the training step.
    """
    train_features, train_labels, test_features, test_labels = load_data_step(
        data_dir=data_dir,
    )
    model, metrics = train_step(
        train_features=train_features,
        train_labels=train_labels,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    _ = evaluate_step(
        model=model,
        test_features=test_features,
        test_labels=test_labels,
        batch_size=batch_size,
    )
    return metrics  # type: ignore[no-any-return]