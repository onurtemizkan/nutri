"""
Advanced LSTM Architectures for Health Metric Prediction

Implements three state-of-the-art model architectures based on research:
1. EnhancedLSTMWithAttention - Best balance of accuracy and interpretability
2. BiLSTMWithResiduals - Robust training with bidirectional processing
3. TemporalConvNet (TCN) - Fast training and long memory

Based on research from:
- "Time Series Modeling for Heart Rate Prediction" (arXiv 2024)
- "Unlocking the Power of LSTM for Long-Term Forecasting" (arXiv 2024)
- "Temporal Attention LSTM for Clinical Prediction" (BMJ 2024)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class AdvancedLSTMConfig:
    """Configuration for advanced LSTM models."""
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    sequence_length: int = 30
    output_dim: int = 1

    # Attention settings
    attention_heads: int = 4
    attention_dim: int = 64

    # Residual settings
    use_residual: bool = True
    use_layer_norm: bool = True

    # Device
    device: str = "cpu"


@dataclass
class TCNConfig:
    """Configuration for Temporal Convolutional Network."""
    input_dim: int
    hidden_channels: int = 64
    num_layers: int = 5
    kernel_size: int = 3
    dropout: float = 0.2
    output_dim: int = 1
    device: str = "cpu"


# =============================================================================
# Model 1: Enhanced LSTM with Temporal Attention
# =============================================================================

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for time series.

    Learns to focus on the most relevant time steps for prediction.
    Provides interpretable attention weights showing which days matter most.

    Based on research:
    - "Temporal Attention LSTM for COVID-19 Prediction" (BMJ 2024)
    - "Using Attention for ICU Mortality Prediction" (ScienceDirect 2022)
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # Learnable query vector (what we're looking for)
        self.query = nn.Parameter(torch.randn(attention_dim))

        # Projections for keys and values
        self.key_proj = nn.Linear(hidden_dim, attention_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Temperature for attention softmax
        self.temperature = math.sqrt(attention_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.key_proj.weight)
        nn.init.xavier_normal_(self.value_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.query, mean=0, std=0.1)

    def forward(
        self,
        lstm_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention over LSTM outputs.

        Args:
            lstm_output: (batch, seq_len, hidden_dim)
            mask: Optional mask for padding (batch, seq_len)

        Returns:
            context: (batch, hidden_dim) - weighted sum of outputs
            attention_weights: (batch, seq_len) - interpretable weights
        """
        batch_size, seq_len, _ = lstm_output.shape

        # Project to keys and values
        keys = self.key_proj(lstm_output)  # (batch, seq_len, attention_dim)
        values = self.value_proj(lstm_output)  # (batch, seq_len, hidden_dim)

        # Compute attention scores
        # Query: (attention_dim,) -> broadcast to (batch, seq_len, attention_dim)
        scores = torch.matmul(keys, self.query) / self.temperature  # (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum of values
        # attention_weights: (batch, seq_len) -> (batch, seq_len, 1)
        # values: (batch, seq_len, hidden_dim)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            values  # (batch, seq_len, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)

        return context, attention_weights


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head temporal attention for richer representations.

    Allows the model to attend to different aspects of the time series
    simultaneously (e.g., recent trends vs long-term patterns).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention over sequence.

        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional attention mask

        Returns:
            output: (batch, hidden_dim) - context from last position
            attention_weights: (batch, num_heads, seq_len) - per-head weights
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x[:, -1:, :])  # Query from last position (batch, 1, hidden_dim)
        k = self.k_proj(x)  # Keys from all positions
        v = self.v_proj(x)  # Values from all positions

        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim) for k, v
        # And: (batch, num_heads, 1, head_dim) for q

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # scores: (batch, num_heads, 1, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        # context: (batch, num_heads, 1, head_dim)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)
        output = self.out_proj(context).squeeze(1)  # (batch, hidden_dim)

        # Return attention weights averaged over heads for interpretability
        avg_attention = attention_weights.squeeze(2).mean(dim=1)  # (batch, seq_len)

        return output, avg_attention


class EnhancedLSTMWithAttention(nn.Module):
    """
    Enhanced LSTM with Temporal Attention for Health Metric Prediction.

    Architecture:
    1. LSTM layers for temporal feature extraction
    2. Temporal attention for focusing on relevant time steps
    3. Fully connected layers for prediction

    Key features:
    - Interpretable attention weights showing which days matter
    - Layer normalization for stable training
    - Residual connections for gradient flow
    - Dropout for regularization

    Based on research:
    - "Time Series Modeling for Heart Rate Prediction" (arXiv 2024)
    - "Temporal Attention LSTM for Clinical Prediction" (BMJ 2024)
    """

    def __init__(self, config: AdvancedLSTMConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )

        # Calculate LSTM output dimension
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_dim)
        else:
            self.layer_norm = nn.Identity()

        # Temporal attention
        self.attention = TemporalAttention(
            hidden_dim=lstm_output_dim,
            attention_dim=config.attention_dim,
        )

        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(lstm_output_dim, config.hidden_dim // 2)
        self.fc2 = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
        self.fc_out = nn.Linear(config.hidden_dim // 4, config.output_dim)

        # Activation
        self.relu = nn.ReLU()

        # Batch normalization for FC layers
        self.bn1 = nn.BatchNorm1d(config.hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(config.hidden_dim // 4)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional attention weights.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_attention: If True, return attention weights

        Returns:
            predictions: (batch, output_dim)
            attention_weights: (batch, seq_len) if return_attention=True
        """
        # Input projection
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Temporal attention
        context, attention_weights = self.attention(lstm_out)

        # Output layers
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc_out(out)

        if return_attention:
            return out, attention_weights
        return out

    def predict(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Make predictions in evaluation mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_attention=return_attention)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Get model summary."""
        return (
            f"EnhancedLSTMWithAttention(\n"
            f"  input_dim={self.config.input_dim},\n"
            f"  hidden_dim={self.config.hidden_dim},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  attention_dim={self.config.attention_dim},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )


# =============================================================================
# Model 2: Bidirectional LSTM with Residual Connections
# =============================================================================

class ResidualLSTMBlock(nn.Module):
    """
    LSTM block with residual connection for stable training.

    Residual connections help with:
    - Gradient flow during backpropagation
    - Training stability
    - Faster convergence
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Projection for residual if dimensions don't match
        if input_dim != lstm_output_dim:
            self.residual_proj = nn.Linear(input_dim, lstm_output_dim)
        else:
            self.residual_proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        residual = self.residual_proj(x)
        lstm_out, _ = self.lstm(x)
        out = self.layer_norm(lstm_out + residual)
        out = self.dropout(out)
        return out


class BiLSTMWithResiduals(nn.Module):
    """
    Bidirectional LSTM with Residual Connections.

    Architecture:
    1. Multiple BiLSTM blocks with residual connections
    2. Global pooling (concatenation of mean and last hidden state)
    3. Fully connected layers with skip connections

    Key features:
    - Bidirectional processing captures both past and future context
    - Residual connections enable training of deeper networks
    - Robust to vanishing gradients

    Based on research:
    - "Unlocking the Power of LSTM for Long-Term Forecasting" (arXiv 2024)
    - Deep residual learning principles
    """

    def __init__(self, config: AdvancedLSTMConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Residual LSTM blocks
        self.lstm_blocks = nn.ModuleList()
        current_dim = config.hidden_dim

        for i in range(config.num_layers):
            block_hidden = config.hidden_dim // (2 ** min(i, 2))  # Gradually decrease
            self.lstm_blocks.append(
                ResidualLSTMBlock(
                    input_dim=current_dim,
                    hidden_dim=block_hidden,
                    bidirectional=True,
                    dropout=config.dropout,
                )
            )
            current_dim = block_hidden * 2  # Bidirectional doubles output

        # Global pooling combines mean and last state
        pooled_dim = current_dim * 2  # Mean + last concatenated

        # Output layers with skip connection
        self.fc1 = nn.Linear(pooled_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.fc_out = nn.Linear(config.hidden_dim // 2, config.output_dim)

        # Skip connection for FC layers
        self.skip_proj = nn.Linear(pooled_dim, config.hidden_dim // 2)

        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim // 2)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual BiLSTM."""
        # Input projection
        out = self.input_proj(x)

        # Pass through residual LSTM blocks
        for block in self.lstm_blocks:
            out = block(out)

        # Global pooling: concatenate mean and last hidden state
        mean_pool = out.mean(dim=1)
        last_hidden = out[:, -1, :]
        pooled = torch.cat([mean_pool, last_hidden], dim=1)

        # FC layers with skip connection
        skip = self.skip_proj(pooled)

        out = self.fc1(pooled)
        out = self.layer_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.layer_norm2(out + skip)  # Skip connection
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc_out(out)
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions in evaluation mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        return (
            f"BiLSTMWithResiduals(\n"
            f"  input_dim={self.config.input_dim},\n"
            f"  hidden_dim={self.config.hidden_dim},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )


# =============================================================================
# Model 3: Temporal Convolutional Network (TCN)
# =============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with dilation.

    Ensures no information leakage from future time steps.
    Dilation expands receptive field exponentially.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.chomp = Chomp1d(self.padding)  # Remove future padding
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal padding."""
        out = self.conv(x)
        out = self.chomp(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class Chomp1d(nn.Module):
    """Remove the future padding from causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size]
        return x


class TCNBlock(nn.Module):
    """
    TCN residual block with two causal convolutions.

    Architecture:
    - Conv -> ReLU -> Dropout -> Conv -> ReLU -> Dropout
    - Residual connection
    - Optional 1x1 conv for dimension matching
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation, dropout
        )
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation, dropout
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual."""
        out = self.conv1(x)
        out = self.conv2(out)
        res = self.residual(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for Health Metric Prediction.

    Architecture:
    1. Input projection
    2. Stack of TCN blocks with exponentially increasing dilation
    3. Global average pooling
    4. Fully connected output layers

    Key features:
    - Causal convolutions prevent future information leakage
    - Dilated convolutions provide exponentially large receptive field
    - Parallel computation is faster than RNNs
    - Stable gradients (no vanishing gradient problem)

    Based on research:
    - "TCN for Clinical Event Prediction" (PMC 2020)
    - "An Empirical Evaluation of Generic Convolutional Networks"
    """

    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config

        # Input projection (features -> channels)
        self.input_proj = nn.Conv1d(
            in_channels=config.input_dim,
            out_channels=config.hidden_channels,
            kernel_size=1,
        )

        # TCN blocks with exponential dilation
        self.tcn_blocks = nn.ModuleList()
        for i in range(config.num_layers):
            dilation = 2 ** i
            self.tcn_blocks.append(
                TCNBlock(
                    in_channels=config.hidden_channels,
                    out_channels=config.hidden_channels,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                )
            )

        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field()

        # Output layers
        self.fc1 = nn.Linear(config.hidden_channels, config.hidden_channels // 2)
        self.fc2 = nn.Linear(config.hidden_channels // 2, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.input_proj.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def _calculate_receptive_field(self) -> int:
        """Calculate the receptive field of the TCN."""
        rf = 1
        for i in range(self.config.num_layers):
            dilation = 2 ** i
            rf += (self.config.kernel_size - 1) * dilation * 2
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            predictions: (batch, output_dim)
        """
        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Input projection
        out = self.input_proj(x)

        # TCN blocks
        for block in self.tcn_blocks:
            out = block(out)

        # Global average pooling over time dimension
        out = out.mean(dim=2)  # (batch, hidden_channels)

        # Output layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions in evaluation mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        return (
            f"TemporalConvNet(\n"
            f"  input_dim={self.config.input_dim},\n"
            f"  hidden_channels={self.config.hidden_channels},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  receptive_field={self.receptive_field} time steps,\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """Factory for creating health prediction models."""

    MODELS = {
        "lstm_attention": EnhancedLSTMWithAttention,
        "bilstm_residual": BiLSTMWithResiduals,
        "tcn": TemporalConvNet,
    }

    @classmethod
    def create(
        cls,
        model_type: str,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ) -> nn.Module:
        """
        Create a model instance.

        Args:
            model_type: One of 'lstm_attention', 'bilstm_residual', 'tcn'
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of layers
            dropout: Dropout rate
            **kwargs: Additional model-specific arguments

        Returns:
            Initialized model
        """
        if model_type not in cls.MODELS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Choose from: {list(cls.MODELS.keys())}")

        if model_type == "tcn":
            config = TCNConfig(
                input_dim=input_dim,
                hidden_channels=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs,
            )
        else:
            config = AdvancedLSTMConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs,
            )

        return cls.MODELS[model_type](config)

    @classmethod
    def list_models(cls) -> List[str]:
        """List available model types."""
        return list(cls.MODELS.keys())


# =============================================================================
# Utility Functions
# =============================================================================

def compare_models(
    models: Dict[str, nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module = nn.MSELoss(),
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same data.

    Args:
        models: Dictionary mapping model names to model instances
        x: Input tensor (batch, seq_len, features)
        y: Target tensor (batch, 1)
        criterion: Loss function

    Returns:
        Dictionary with metrics for each model
    """
    results = {}

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            predictions = model(x)
            loss = criterion(predictions, y).item()

            # Calculate additional metrics
            mae = torch.abs(predictions - y).mean().item()
            mse = ((predictions - y) ** 2).mean().item()

            results[name] = {
                "loss": loss,
                "mae": mae,
                "mse": mse,
                "parameters": model.count_parameters(),
            }

    return results


if __name__ == "__main__":
    # Test the models
    batch_size = 32
    seq_len = 30
    input_dim = 51

    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Test each model type
    for model_type in ModelFactory.list_models():
        print(f"\n{'='*50}")
        print(f"Testing {model_type}")
        print('='*50)

        model = ModelFactory.create(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
        )

        print(model.summary())

        # Forward pass
        output = model(x)
        print(f"Output shape: {output.shape}")

        # Test attention weights if available
        if hasattr(model, 'get_attention_weights'):
            weights = model.get_attention_weights(x)
            print(f"Attention weights shape: {weights.shape}")
            print(f"Attention weights sum: {weights.sum(dim=1).mean():.4f}")
