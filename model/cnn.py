"""Neural network architectures used by the training script."""

import torch
from torch import nn


class CNN(nn.Module):  # type: ignore[misc]
    """Convolutional neural network for image classification.

    A simple CNN with two convolutional layers followed by two fully connected
    layers that automatically handles both 2D image inputs and flattened inputs.

    Args:
        input_channels: Number of input channels (default: 1 for grayscale).
        output_size: Number of output classes (default: 10).
    """

    def __init__(self, input_channels: int = 1, output_size: int = 10) -> None:
        """Initialise the CNN layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 7x7 is the spatial dimension after two 2x2 pooling layers on a 28x28 input
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Handles both image inputs [batch_size, channels, 28, 28] and
        flattened inputs [batch_size, 784].

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output predictions.
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x
