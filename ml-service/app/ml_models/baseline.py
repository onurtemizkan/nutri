"""
Baseline models for comparison with LSTM.

These simple models provide a performance baseline to ensure our complex
LSTM models actually provide value over simpler approaches.
"""

import torch
import torch.nn as nn


class BaselineLinearModel(nn.Module):
    """
    Simple linear regression baseline.

    Takes the average of the last N days and uses linear regression
    to predict the next day's value.

    This serves as a sanity check - our LSTM should outperform this.
    """

    def __init__(self, input_dim: int):
        super(BaselineLinearModel, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, sequence_length, input_dim)

        Returns:
            Predictions of shape (batch, 1)
        """
        # Take average across sequence (simple aggregation)
        x_avg = x.mean(dim=1)  # Shape: (batch, input_dim)

        # Linear prediction
        out = self.linear(x_avg)
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions (inference mode)."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class MovingAverageBaseline:
    """
    Moving average baseline (non-neural).

    Simply predicts tomorrow's value as the average of the last N days.
    This is the simplest possible baseline.
    """

    def __init__(self, window_size: int = 7):
        self.window_size = window_size

    def predict(self, historical_values: torch.Tensor) -> float:
        """
        Predict using moving average.

        Args:
            historical_values: Tensor of shape (sequence_length,)

        Returns:
            Predicted value (scalar)
        """
        # Take last window_size values
        recent = historical_values[-self.window_size:]
        return float(recent.mean())


class ExponentialSmoothingBaseline:
    """
    Exponential smoothing baseline.

    Gives more weight to recent values, less to older values.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing parameter (0-1)
                  - 0: Only use oldest value
                  - 1: Only use most recent value
                  - 0.3: Good default balance
        """
        self.alpha = alpha

    def predict(self, historical_values: torch.Tensor) -> float:
        """
        Predict using exponential smoothing.

        Args:
            historical_values: Tensor of shape (sequence_length,)

        Returns:
            Predicted value (scalar)
        """
        if len(historical_values) == 0:
            return 0.0

        smoothed = historical_values[0].item()

        for value in historical_values[1:]:
            smoothed = self.alpha * value.item() + (1 - self.alpha) * smoothed

        return smoothed
