"""
PyTorch LSTM Model for Health Metric Prediction (RHR, HRV, etc.)

This module implements a flexible LSTM architecture that can predict various
health metrics based on historical sequences of nutrition, activity, and health features.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""

    input_dim: int  # Number of input features
    hidden_dim: int = 128  # LSTM hidden dimension
    num_layers: int = 2  # Number of LSTM layers
    dropout: float = 0.2  # Dropout rate
    bidirectional: bool = False  # Bidirectional LSTM
    sequence_length: int = 30  # Input sequence length (days)

    # Output
    output_dim: int = 1  # Single value prediction (RHR, HRV, etc.)

    # Device
    device: str = "cpu"  # "cpu" or "cuda"


class HealthMetricLSTM(nn.Module):
    """
    LSTM neural network for health metric prediction.

    Architecture:
    1. Input Layer: (batch, sequence_length, input_dim)
    2. LSTM Layers: Stacked LSTM with dropout
    3. Fully Connected Layers: Dense layers with ReLU
    4. Output Layer: Single value (predicted metric)

    Example:
        >>> config = LSTMConfig(input_dim=51, hidden_dim=128, num_layers=2)
        >>> model = HealthMetricLSTM(config)
        >>> x = torch.randn(32, 30, 51)  # batch=32, seq=30, features=51
        >>> prediction = model(x)  # Output: (32, 1)
    """

    def __init__(self, config: LSTMConfig):
        super(HealthMetricLSTM, self).__init__()
        self.config = config

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )

        # Calculate LSTM output dimension
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        # Dropout after LSTM
        self.dropout = nn.Dropout(config.dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, config.hidden_dim // 2)
        self.fc2 = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
        self.fc_out = nn.Linear(config.hidden_dim // 4, config.output_dim)

        # Activation
        self.relu = nn.ReLU()

        # Batch normalization (optional, helps with training stability)
        self.bn1 = nn.BatchNorm1d(config.hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(config.hidden_dim // 4)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if "weight" in name:
                # Xavier initialization requires 2D+ tensors (fan_in, fan_out)
                # Skip 1D tensors like batch norm weights
                if param.dim() >= 2:
                    if "lstm" in name:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)
            hidden: Optional initial hidden state (for stateful predictions)

        Returns:
            Predicted values of shape (batch, output_dim)
        """
        _batch_size = x.size(0)  # noqa: F841 - documented for clarity

        # LSTM forward pass
        if hidden is not None:
            lstm_out, hidden_state = self.lstm(x, hidden)
        else:
            lstm_out, hidden_state = self.lstm(x)

        # Take the last output from the sequence
        # Shape: (batch, hidden_dim * num_directions)
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        out = self.dropout(last_output)

        # First fully connected layer
        out = self.fc1(out)
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)
        out = self.dropout(out)

        # Second fully connected layer
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Output layer (no activation for regression)
        out = self.fc_out(out)

        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make a prediction (inference mode).

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)

        Returns:
            Predicted values of shape (batch, output_dim)
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights for interpretability.

        This is a placeholder for future attention mechanism implementation.
        For now, it returns None.

        Args:
            x: Input tensor

        Returns:
            Attention weights (not implemented yet)
        """
        # TODO: Implement attention mechanism for interpretability
        return None

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_device(self, device: str):
        """Move model to specified device."""
        self.config.device = device
        return self.to(device)

    def summary(self):
        """Print model summary."""
        print("HealthMetricLSTM Model Summary:")
        print(f"  Input dimension: {self.config.input_dim}")
        print(f"  Hidden dimension: {self.config.hidden_dim}")
        print(f"  Number of layers: {self.config.num_layers}")
        print(f"  Bidirectional: {self.config.bidirectional}")
        print(f"  Dropout: {self.config.dropout}")
        print(f"  Sequence length: {self.config.sequence_length}")
        print(f"  Total parameters: {self.count_parameters():,}")


class MultiTaskLSTM(nn.Module):
    """
    Multi-task LSTM that predicts multiple health metrics simultaneously.

    This is useful for predicting RHR and HRV together, as they may share
    underlying patterns.

    Architecture:
    - Shared LSTM encoder
    - Separate prediction heads for each metric

    Example:
        >>> config = LSTMConfig(input_dim=51, hidden_dim=128)
        >>> model = MultiTaskLSTM(config, num_tasks=2)  # RHR + HRV
        >>> x = torch.randn(32, 30, 51)
        >>> predictions = model(x)  # Output: (32, 2)
    """

    def __init__(self, config: LSTMConfig, num_tasks: int = 2):
        super(MultiTaskLSTM, self).__init__()
        self.config = config
        self.num_tasks = num_tasks

        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )

        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        # Shared layer
        self.shared_fc = nn.Linear(lstm_output_dim, config.hidden_dim // 2)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

        # Task-specific heads
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim // 4, 1),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-task prediction.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)

        Returns:
            Predictions of shape (batch, num_tasks)
        """
        # Shared LSTM encoder
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        # Shared fully connected layer
        shared = self.shared_fc(last_output)
        shared = self.relu(shared)
        shared = self.dropout(shared)

        # Task-specific predictions
        predictions = []
        for head in self.task_heads:
            pred = head(shared)
            predictions.append(pred)

        # Concatenate predictions
        return torch.cat(predictions, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions (inference mode)."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class LSTMWithAttention(nn.Module):
    """
    LSTM with attention mechanism for better interpretability.

    The attention mechanism helps identify which time steps (days) are most
    important for the prediction. This allows us to answer questions like:
    - "Which days in the past 30 days most influenced tomorrow's prediction?"
    - "Was yesterday more important than last week?"

    Architecture:
    1. Input Layer: (batch, sequence_length, input_dim)
    2. LSTM Layers: Extract temporal features
    3. Attention Layer: Calculate importance weights for each time step
    4. Context Vector: Weighted sum of LSTM outputs
    5. Fully Connected Layers: Predict from context vector
    6. Output Layer: Single value (predicted metric)

    Example:
        >>> config = LSTMConfig(input_dim=51, hidden_dim=128, num_layers=2)
        >>> model = LSTMWithAttention(config)
        >>> x = torch.randn(32, 30, 51)  # batch=32, seq=30, features=51
        >>> prediction, attention_weights = model(x)
        >>> # prediction: (32, 1)
        >>> # attention_weights: (32, 30) - importance of each of 30 days
    """

    def __init__(self, config: LSTMConfig):
        super(LSTMWithAttention, self).__init__()
        self.config = config

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )

        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        # Attention mechanism
        # Maps LSTM output to attention scores
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1),
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Fully connected layers (same as HealthMetricLSTM)
        self.fc1 = nn.Linear(lstm_output_dim, config.hidden_dim // 2)
        self.fc2 = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
        self.fc_out = nn.Linear(config.hidden_dim // 4, config.output_dim)

        # Activation and batch normalization
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(config.hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(config.hidden_dim // 4)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if "weight" in name:
                # Xavier initialization requires 2D+ tensors (fan_in, fan_out)
                # Skip 1D tensors like batch norm weights
                if param.dim() >= 2:
                    if "lstm" in name:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with attention.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)
            return_attention: If True, return attention weights

        Returns:
            If return_attention=False:
                predictions: (batch, output_dim)
            If return_attention=True:
                (predictions, attention_weights)
                - predictions: (batch, output_dim)
                - attention_weights: (batch, sequence_length)
        """
        _batch_size = x.size(0)  # noqa: F841 - documented for clarity
        _seq_length = x.size(1)  # noqa: F841 - documented for clarity

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_length, lstm_output_dim)

        # Calculate attention scores for each time step
        # Shape: (batch, seq_length, lstm_output_dim) -> (batch, seq_length, 1)
        attention_scores = self.attention(lstm_out)

        # Apply softmax to get attention weights
        # Shape: (batch, seq_length, 1) -> (batch, seq_length)
        attention_weights = torch.softmax(attention_scores.squeeze(2), dim=1)

        # Create context vector: weighted sum of LSTM outputs
        # Shape: (batch, seq_length, 1) @ (batch, seq_length, lstm_output_dim)
        # Result: (batch, lstm_output_dim)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)

        # Apply dropout
        out = self.dropout(context_vector)

        # Fully connected layers (same as HealthMetricLSTM)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Output layer
        out = self.fc_out(out)

        if return_attention:
            return out, attention_weights
        else:
            return out, None

    def predict(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Make a prediction (inference mode).

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)
            return_attention: If True, return attention weights

        Returns:
            If return_attention=False:
                predictions: (batch, output_dim)
            If return_attention=True:
                (predictions, attention_weights)
        """
        self.eval()
        with torch.no_grad():
            predictions, attention_weights = self.forward(x, return_attention=True)

        if return_attention:
            return predictions, attention_weights
        else:
            return predictions, None

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights for interpretability.

        This method extracts which days (time steps) the model considers
        most important for the prediction.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)

        Returns:
            Attention weights: (batch, sequence_length)
            Each weight represents the importance of that time step (day).
            Weights sum to 1 across the sequence.

        Example:
            >>> x = torch.randn(1, 30, 51)  # 30 days, 51 features
            >>> weights = model.get_attention_weights(x)
            >>> weights.shape  # (1, 30)
            >>> # High weight for day 29 means yesterday was very important
            >>> # High weight for day 0 means 30 days ago was very important
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_device(self, device: str):
        """Move model to specified device."""
        self.config.device = device
        return self.to(device)

    def summary(self):
        """Print model summary."""
        print("LSTMWithAttention Model Summary:")
        print(f"  Input dimension: {self.config.input_dim}")
        print(f"  Hidden dimension: {self.config.hidden_dim}")
        print(f"  Number of layers: {self.config.num_layers}")
        print(f"  Bidirectional: {self.config.bidirectional}")
        print(f"  Dropout: {self.config.dropout}")
        print(f"  Sequence length: {self.config.sequence_length}")
        print("  Attention: Yes")
        print(f"  Total parameters: {self.count_parameters():,}")
