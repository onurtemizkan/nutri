"""
LSTM Temporal Pattern Analyzer for Food Sensitivity Detection.

This module provides deep learning-based temporal analysis for detecting
food sensitivity patterns in HRV time series data using:
- Bidirectional LSTM for sequence modeling
- Multi-head self-attention for temporal focus
- Temporal convolutional layers for multi-scale features
- Variational inference for uncertainty quantification
- Attention visualization for interpretability

Architecture:
1. Input preprocessing and feature extraction
2. Temporal convolution for multi-scale patterns
3. Bidirectional LSTM for sequence dependencies
4. Multi-head attention for temporal weighting
5. Classification head with uncertainty estimation

Research basis:
- LSTMs capture long-term dependencies in HRV patterns
- Attention identifies critical time windows post-ingestion
- 2-6 hour typical response window for food sensitivities
"""

import math
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from collections import defaultdict
import logging

# PyTorch imports (graceful handling)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = nn = F = Dataset = DataLoader = None
    TORCH_AVAILABLE = False

# NumPy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of temporal patterns detected."""
    IMMEDIATE_RESPONSE = "immediate"  # 0-30 min
    DELAYED_RESPONSE = "delayed"  # 30 min - 2 hr
    LATE_RESPONSE = "late"  # 2-6 hr
    CUMULATIVE = "cumulative"  # Build-up over days
    CIRCADIAN = "circadian"  # Time-of-day dependent
    RECOVERY = "recovery"  # Return to baseline pattern


class ResponseSeverity(Enum):
    """Severity classification of detected responses."""
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


@dataclass
class TemporalPattern:
    """Detected temporal pattern."""
    pattern_type: PatternType
    start_time_minutes: int
    end_time_minutes: int
    peak_time_minutes: int
    severity: ResponseSeverity
    confidence: float
    attention_weights: List[float]
    feature_importance: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type.value,
            "start_time_minutes": self.start_time_minutes,
            "end_time_minutes": self.end_time_minutes,
            "peak_time_minutes": self.peak_time_minutes,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "attention_weights": self.attention_weights[:20],  # Truncate for display
            "feature_importance": self.feature_importance,
        }


@dataclass
class AnalysisResult:
    """Complete temporal analysis result."""
    detected_patterns: List[TemporalPattern]
    overall_sensitivity_score: float
    hrv_trajectory: List[float]
    predicted_recovery_time_hours: float
    uncertainty: float
    recommendations: List[str]
    model_confidence: float
    raw_predictions: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_patterns": [p.to_dict() for p in self.detected_patterns],
            "overall_sensitivity_score": self.overall_sensitivity_score,
            "hrv_trajectory": self.hrv_trajectory[:48],  # First 48 time points
            "predicted_recovery_time_hours": self.predicted_recovery_time_hours,
            "uncertainty": self.uncertainty,
            "recommendations": self.recommendations,
            "model_confidence": self.model_confidence,
        }


# ==================== PyTorch Models ====================

if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """
        Sinusoidal positional encoding for temporal awareness.

        Adds position information to input embeddings, allowing the model
        to understand the relative ordering of time steps.
        """

        def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Create positional encodings
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )

            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)

            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Tensor of shape [seq_len, batch_size, d_model]
            """
            x = x + self.pe[: x.size(0)]
            return self.dropout(x)


    class TemporalConvBlock(nn.Module):
        """
        Temporal convolutional block for multi-scale feature extraction.

        Uses dilated convolutions to capture patterns at different time scales:
        - Small dilation: Immediate responses (minutes)
        - Medium dilation: Delayed responses (hours)
        - Large dilation: Late responses (many hours)
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            dilation: int = 1,
            dropout: float = 0.2,
        ):
            super().__init__()

            padding = (kernel_size - 1) * dilation // 2

            self.conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=padding, dilation=dilation
            )
            self.conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size,
                padding=padding, dilation=dilation
            )

            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)

            self.dropout = nn.Dropout(dropout)

            # Residual connection
            self.residual = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, channels, seq_len]
            """
            residual = self.residual(x)

            out = self.conv1(x)
            out = self.norm1(out)
            out = F.gelu(out)
            out = self.dropout(out)

            out = self.conv2(out)
            out = self.norm2(out)

            out = out + residual
            out = F.gelu(out)

            return out


    class MultiHeadTemporalAttention(nn.Module):
        """
        Multi-head self-attention for temporal focus.

        Learns to attend to relevant time windows that indicate
        food sensitivity responses. Different heads can focus on:
        - Immediate response windows
        - Delayed response windows
        - Recovery patterns
        """

        def __init__(
            self,
            d_model: int,
            num_heads: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()

            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads

            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)

            # Store attention weights for visualization
            self.attention_weights: Optional[torch.Tensor] = None

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_attention: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Args:
                x: [batch, seq_len, d_model]
                mask: Optional attention mask
                return_attention: Whether to return attention weights

            Returns:
                Output tensor and optionally attention weights
            """
            batch_size, seq_len, _ = x.shape

            # Project to queries, keys, values
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # Store for visualization
            self.attention_weights = attn.detach()

            # Apply attention to values
            out = torch.matmul(attn, v)

            # Reshape back
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            out = self.out_proj(out)

            if return_attention:
                return out, attn
            return out


    class BiLSTMEncoder(nn.Module):
        """
        Bidirectional LSTM encoder for sequence modeling.

        Captures both forward and backward temporal dependencies,
        which is important for understanding:
        - How HRV changes after food ingestion (forward)
        - What baseline state preceded the change (backward)
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = True,
        ):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )

            self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        def forward(
            self,
            x: torch.Tensor,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Args:
                x: [batch, seq_len, input_size]
                hidden: Optional initial hidden state

            Returns:
                Output tensor and final hidden state
            """
            out, hidden = self.lstm(x, hidden)
            out = self.layer_norm(out)
            return out, hidden


    class VariationalDropout(nn.Module):
        """
        Variational dropout for uncertainty estimation.

        Applies the same dropout mask across the sequence dimension,
        enabling Monte Carlo dropout for uncertainty quantification.
        """

        def __init__(self, dropout: float = 0.1):
            super().__init__()
            self.dropout = dropout

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.training or self.dropout == 0:
                return x

            # Create mask for (batch, 1, features) to broadcast across seq
            mask = torch.bernoulli(
                torch.ones(x.size(0), 1, x.size(2), device=x.device) * (1 - self.dropout)
            )
            mask = mask / (1 - self.dropout)

            return x * mask


    class LSTMTemporalAnalyzer(nn.Module):
        """
        Main LSTM-based temporal analyzer for food sensitivity detection.

        Architecture:
        1. Input projection and positional encoding
        2. Multi-scale temporal convolution
        3. Bidirectional LSTM encoding
        4. Multi-head temporal attention
        5. Classification and regression heads

        The model outputs:
        - Sensitivity probability (binary classification)
        - Severity score (regression)
        - Recovery time prediction (regression)
        - Attention weights for interpretability
        """

        def __init__(
            self,
            input_features: int = 8,
            hidden_size: int = 128,
            num_lstm_layers: int = 2,
            num_attention_heads: int = 4,
            num_conv_scales: int = 3,
            dropout: float = 0.2,
            num_severity_classes: int = 5,
        ):
            super().__init__()

            self.input_features = input_features
            self.hidden_size = hidden_size

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_features, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Positional encoding
            self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

            # Multi-scale temporal convolution
            self.conv_blocks = nn.ModuleList([
                TemporalConvBlock(
                    hidden_size, hidden_size,
                    kernel_size=3, dilation=2**i, dropout=dropout
                )
                for i in range(num_conv_scales)
            ])

            # BiLSTM encoder
            self.lstm = BiLSTMEncoder(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                dropout=dropout,
                bidirectional=True,
            )

            # Multi-head attention
            self.attention = MultiHeadTemporalAttention(
                d_model=hidden_size * 2,  # BiLSTM output is 2x hidden
                num_heads=num_attention_heads,
                dropout=dropout,
            )

            # Variational dropout for uncertainty
            self.var_dropout = VariationalDropout(dropout)

            # Output heads
            lstm_out_size = hidden_size * 2

            # Binary sensitivity detection
            self.sensitivity_head = nn.Sequential(
                nn.Linear(lstm_out_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

            # Severity classification
            self.severity_head = nn.Sequential(
                nn.Linear(lstm_out_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_severity_classes),
            )

            # Recovery time regression (hours)
            self.recovery_head = nn.Sequential(
                nn.Linear(lstm_out_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Softplus(),  # Ensure positive output
            )

            # Sequence-to-sequence trajectory prediction
            self.trajectory_head = nn.Sequential(
                nn.Linear(lstm_out_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                x: Input tensor [batch, seq_len, input_features]
                return_attention: Whether to return attention weights

            Returns:
                Dictionary with model outputs
            """
            batch_size, seq_len, _ = x.shape

            # Project input
            x = self.input_proj(x)

            # Add positional encoding
            x = x.transpose(0, 1)  # [seq, batch, hidden]
            x = self.pos_encoder(x)
            x = x.transpose(0, 1)  # [batch, seq, hidden]

            # Multi-scale convolution (expects [batch, channels, seq])
            x_conv = x.transpose(1, 2)
            for conv_block in self.conv_blocks:
                x_conv = conv_block(x_conv)
            x = x_conv.transpose(1, 2)  # Back to [batch, seq, hidden]

            # BiLSTM encoding
            lstm_out, _ = self.lstm(x)

            # Apply variational dropout
            lstm_out = self.var_dropout(lstm_out)

            # Multi-head attention
            if return_attention:
                attn_out, attn_weights = self.attention(lstm_out, return_attention=True)
            else:
                attn_out = self.attention(lstm_out)
                attn_weights = None

            # Global pooling for classification
            # Use both mean and max pooling
            mean_pool = attn_out.mean(dim=1)
            max_pool = attn_out.max(dim=1)[0]
            pooled = (mean_pool + max_pool) / 2

            # Compute outputs
            sensitivity_logit = self.sensitivity_head(pooled)
            severity_logits = self.severity_head(pooled)
            recovery_time = self.recovery_head(pooled)
            trajectory = self.trajectory_head(lstm_out).squeeze(-1)

            outputs = {
                "sensitivity_logit": sensitivity_logit,
                "sensitivity_prob": torch.sigmoid(sensitivity_logit),
                "severity_logits": severity_logits,
                "severity_probs": F.softmax(severity_logits, dim=-1),
                "recovery_time_hours": recovery_time,
                "trajectory": trajectory,
            }

            if return_attention:
                outputs["attention_weights"] = attn_weights

            return outputs

        def predict_with_uncertainty(
            self,
            x: torch.Tensor,
            num_samples: int = 20
        ) -> Dict[str, torch.Tensor]:
            """
            Make predictions with uncertainty estimation using MC Dropout.

            Args:
                x: Input tensor
                num_samples: Number of forward passes for MC dropout

            Returns:
                Predictions with mean and standard deviation
            """
            self.train()  # Enable dropout

            sensitivity_samples = []
            severity_samples = []
            recovery_samples = []
            trajectory_samples = []

            with torch.no_grad():
                for _ in range(num_samples):
                    outputs = self(x)
                    sensitivity_samples.append(outputs["sensitivity_prob"])
                    severity_samples.append(outputs["severity_probs"])
                    recovery_samples.append(outputs["recovery_time_hours"])
                    trajectory_samples.append(outputs["trajectory"])

            # Stack and compute statistics
            sensitivity_stack = torch.stack(sensitivity_samples, dim=0)
            severity_stack = torch.stack(severity_samples, dim=0)
            recovery_stack = torch.stack(recovery_samples, dim=0)
            trajectory_stack = torch.stack(trajectory_samples, dim=0)

            self.eval()  # Disable dropout

            return {
                "sensitivity_mean": sensitivity_stack.mean(dim=0),
                "sensitivity_std": sensitivity_stack.std(dim=0),
                "severity_mean": severity_stack.mean(dim=0),
                "severity_std": severity_stack.std(dim=0),
                "recovery_mean": recovery_stack.mean(dim=0),
                "recovery_std": recovery_stack.std(dim=0),
                "trajectory_mean": trajectory_stack.mean(dim=0),
                "trajectory_std": trajectory_stack.std(dim=0),
            }


    class SensitivityDataset(Dataset):
        """Dataset for food sensitivity temporal data."""

        def __init__(
            self,
            sequences: List[np.ndarray],
            labels: Optional[List[Dict[str, Any]]] = None,
        ):
            self.sequences = sequences
            self.labels = labels

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            seq = torch.tensor(self.sequences[idx], dtype=torch.float32)

            item = {"sequence": seq}

            if self.labels is not None:
                label = self.labels[idx]
                item["sensitivity"] = torch.tensor(
                    label.get("sensitivity", 0), dtype=torch.float32
                )
                item["severity"] = torch.tensor(
                    label.get("severity", 0), dtype=torch.long
                )
                item["recovery_hours"] = torch.tensor(
                    label.get("recovery_hours", 0), dtype=torch.float32
                )

            return item


# ==================== High-Level Analyzer ====================

class LSTMTemporalPatternAnalyzer:
    """
    High-level interface for LSTM-based temporal pattern analysis.

    Provides:
    - Model initialization and loading
    - Preprocessing and feature extraction
    - Pattern detection and classification
    - Result interpretation and visualization
    """

    # Expected HRV features
    HRV_FEATURES = [
        "hrv_sdnn",
        "hrv_rmssd",
        "hrv_pnn50",
        "hrv_lf",
        "hrv_hf",
        "hrv_lf_hf_ratio",
        "heart_rate",
        "respiratory_rate",
    ]

    # Time constants
    SAMPLING_INTERVAL_MINUTES = 5  # Expected data point interval
    ANALYSIS_WINDOW_HOURS = 12  # Default analysis window

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
    ):
        """
        Initialize the analyzer.

        Args:
            model_path: Path to saved model weights
            device: Computation device ('cuda', 'cpu', or None for auto)
            hidden_size: LSTM hidden size
            num_lstm_layers: Number of LSTM layers
        """
        self.device = self._get_device(device)
        self.model: Optional[Any] = None
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # Feature normalization parameters
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        # Pattern detection thresholds
        self.sensitivity_threshold = 0.5
        self.severity_thresholds = [0.2, 0.4, 0.6, 0.8]

    def _get_device(self, device: Optional[str]) -> str:
        """Determine computation device."""
        if device:
            return device
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        if TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def initialize(self) -> bool:
        """Initialize the model."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, LSTM analyzer disabled")
            return False

        try:
            self.model = LSTMTemporalAnalyzer(
                input_features=len(self.HRV_FEATURES),
                hidden_size=self.hidden_size,
                num_lstm_layers=self.num_lstm_layers,
            )
            self.model = self.model.to(self.device)

            if self.model_path:
                self.load_model(self.model_path)

            self.model.eval()
            logger.info(f"LSTM analyzer initialized on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LSTM analyzer: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load model weights from file."""
        if not TORCH_AVAILABLE or not self.model:
            return False

        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict["model"])

            if "feature_means" in state_dict:
                self.feature_means = state_dict["feature_means"]
                self.feature_stds = state_dict["feature_stds"]

            logger.info(f"Loaded model from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def save_model(self, path: str) -> bool:
        """Save model weights to file."""
        if not TORCH_AVAILABLE or not self.model:
            return False

        try:
            state_dict = {
                "model": self.model.state_dict(),
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds,
            }
            torch.save(state_dict, path)
            logger.info(f"Saved model to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def preprocess(
        self,
        hrv_data: List[Dict[str, float]],
        timestamps: Optional[List[float]] = None,
    ) -> Optional[np.ndarray]:
        """
        Preprocess HRV data for model input.

        Args:
            hrv_data: List of HRV measurements with feature dict
            timestamps: Optional timestamps for each measurement

        Returns:
            Preprocessed numpy array [seq_len, features]
        """
        if not NUMPY_AVAILABLE:
            return None

        # Extract features in expected order
        features = []
        for measurement in hrv_data:
            feature_vec = []
            for feat_name in self.HRV_FEATURES:
                value = measurement.get(feat_name, 0.0)
                # Handle missing values
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    value = 0.0
                feature_vec.append(value)
            features.append(feature_vec)

        data = np.array(features, dtype=np.float32)

        # Normalize
        if self.feature_means is not None and self.feature_stds is not None:
            data = (data - self.feature_means) / (self.feature_stds + 1e-8)
        else:
            # Simple standardization if no stored params
            data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        return data

    def fit_normalization(self, training_data: List[np.ndarray]) -> None:
        """Fit normalization parameters from training data."""
        if not NUMPY_AVAILABLE:
            return

        all_data = np.concatenate(training_data, axis=0)
        self.feature_means = all_data.mean(axis=0)
        self.feature_stds = all_data.std(axis=0)

    def analyze(
        self,
        hrv_data: List[Dict[str, float]],
        timestamps: Optional[List[float]] = None,
        with_uncertainty: bool = True,
    ) -> AnalysisResult:
        """
        Analyze HRV temporal patterns for food sensitivity.

        Args:
            hrv_data: List of HRV measurements
            timestamps: Optional timestamps
            with_uncertainty: Whether to compute uncertainty estimates

        Returns:
            AnalysisResult with detected patterns
        """
        # Default result for fallback
        default_result = AnalysisResult(
            detected_patterns=[],
            overall_sensitivity_score=0.0,
            hrv_trajectory=[],
            predicted_recovery_time_hours=0.0,
            uncertainty=1.0,
            recommendations=["Insufficient data for analysis"],
            model_confidence=0.0,
        )

        if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
            default_result.recommendations = ["PyTorch/NumPy not available"]
            return default_result

        if not self.model:
            if not self.initialize():
                return default_result

        # Preprocess data
        data = self.preprocess(hrv_data, timestamps)
        if data is None or len(data) < 10:
            default_result.recommendations = ["Insufficient data points"]
            return default_result

        # Convert to tensor
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Run model
        try:
            with torch.no_grad():
                if with_uncertainty:
                    outputs = self.model.predict_with_uncertainty(x)
                    sensitivity_score = outputs["sensitivity_mean"].item()
                    sensitivity_uncertainty = outputs["sensitivity_std"].item()
                    severity_probs = outputs["severity_mean"].squeeze().cpu().numpy()
                    recovery_hours = outputs["recovery_mean"].item()
                    recovery_uncertainty = outputs["recovery_std"].item()
                    trajectory = outputs["trajectory_mean"].squeeze().cpu().numpy()
                else:
                    outputs = self.model(x, return_attention=True)
                    sensitivity_score = outputs["sensitivity_prob"].item()
                    sensitivity_uncertainty = 0.0
                    severity_probs = outputs["severity_probs"].squeeze().cpu().numpy()
                    recovery_hours = outputs["recovery_time_hours"].item()
                    recovery_uncertainty = 0.0
                    trajectory = outputs["trajectory"].squeeze().cpu().numpy()

                # Get attention weights if available
                attn_weights = outputs.get("attention_weights")

        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return default_result

        # Detect patterns from attention and trajectory
        patterns = self._detect_patterns(
            trajectory,
            attn_weights,
            sensitivity_score,
            severity_probs,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            sensitivity_score,
            severity_probs,
            recovery_hours,
            patterns,
        )

        # Compute model confidence
        model_confidence = self._compute_confidence(
            sensitivity_score,
            sensitivity_uncertainty,
            len(hrv_data),
        )

        return AnalysisResult(
            detected_patterns=patterns,
            overall_sensitivity_score=sensitivity_score,
            hrv_trajectory=trajectory.tolist(),
            predicted_recovery_time_hours=recovery_hours,
            uncertainty=sensitivity_uncertainty + recovery_uncertainty,
            recommendations=recommendations,
            model_confidence=model_confidence,
            raw_predictions=[sensitivity_score] + severity_probs.tolist(),
        )

    def _detect_patterns(
        self,
        trajectory: np.ndarray,
        attention_weights: Optional[Any],
        sensitivity_score: float,
        severity_probs: np.ndarray,
    ) -> List[TemporalPattern]:
        """Detect temporal patterns from model outputs."""
        patterns = []

        if sensitivity_score < self.sensitivity_threshold:
            return patterns  # No sensitivity detected

        # Determine severity
        severity_idx = int(np.argmax(severity_probs))
        severity = ResponseSeverity(severity_idx)

        # Analyze trajectory for response timing
        # Find significant deviations from baseline
        baseline = trajectory[:10].mean() if len(trajectory) > 10 else 0
        deviations = trajectory - baseline

        # Find response windows using smoothed derivative
        if len(deviations) > 5:
            smoothed = np.convolve(deviations, np.ones(5)/5, mode='valid')

            # Find significant drops (assuming HRV drops indicate stress response)
            threshold = -np.std(smoothed)
            response_mask = smoothed < threshold

            if response_mask.any():
                # Find contiguous response regions
                start_idx = None
                for i, is_response in enumerate(response_mask):
                    if is_response and start_idx is None:
                        start_idx = i
                    elif not is_response and start_idx is not None:
                        # End of response region
                        end_idx = i
                        peak_idx = start_idx + np.argmin(smoothed[start_idx:end_idx])

                        # Determine pattern type based on timing
                        start_minutes = start_idx * self.SAMPLING_INTERVAL_MINUTES
                        pattern_type = self._classify_timing(start_minutes)

                        # Extract attention for this window
                        if attention_weights is not None:
                            attn = attention_weights[0, :, start_idx:end_idx, :].mean()
                            attn_list = [float(attn)] * (end_idx - start_idx)
                        else:
                            attn_list = [1.0] * (end_idx - start_idx)

                        pattern = TemporalPattern(
                            pattern_type=pattern_type,
                            start_time_minutes=start_minutes,
                            end_time_minutes=end_idx * self.SAMPLING_INTERVAL_MINUTES,
                            peak_time_minutes=peak_idx * self.SAMPLING_INTERVAL_MINUTES,
                            severity=severity,
                            confidence=sensitivity_score,
                            attention_weights=attn_list,
                            feature_importance=self._get_feature_importance(),
                        )
                        patterns.append(pattern)

                        start_idx = None

        # If no patterns detected but sensitivity is high, add generic pattern
        if not patterns and sensitivity_score > self.sensitivity_threshold:
            patterns.append(TemporalPattern(
                pattern_type=PatternType.DELAYED_RESPONSE,
                start_time_minutes=60,
                end_time_minutes=180,
                peak_time_minutes=120,
                severity=severity,
                confidence=sensitivity_score,
                attention_weights=[sensitivity_score] * 24,
                feature_importance=self._get_feature_importance(),
            ))

        return patterns

    def _classify_timing(self, start_minutes: int) -> PatternType:
        """Classify pattern type based on onset timing."""
        if start_minutes < 30:
            return PatternType.IMMEDIATE_RESPONSE
        elif start_minutes < 120:
            return PatternType.DELAYED_RESPONSE
        elif start_minutes < 360:
            return PatternType.LATE_RESPONSE
        else:
            return PatternType.CUMULATIVE

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (simplified version)."""
        # In a full implementation, this would use gradient-based attribution
        return {
            feat: 1.0 / len(self.HRV_FEATURES)
            for feat in self.HRV_FEATURES
        }

    def _generate_recommendations(
        self,
        sensitivity_score: float,
        severity_probs: np.ndarray,
        recovery_hours: float,
        patterns: List[TemporalPattern],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if sensitivity_score < 0.3:
            recommendations.append("No significant sensitivity detected")
            return recommendations

        severity_idx = int(np.argmax(severity_probs))

        if sensitivity_score > 0.7:
            recommendations.append("Strong sensitivity signal detected")
            recommendations.append(f"Consider avoiding this food trigger")

        if severity_idx >= 2:
            recommendations.append(f"Moderate to severe response pattern observed")
            recommendations.append(
                f"Expected recovery time: ~{recovery_hours:.1f} hours"
            )

        for pattern in patterns:
            if pattern.pattern_type == PatternType.IMMEDIATE_RESPONSE:
                recommendations.append(
                    "Immediate response suggests possible IgE-mediated reaction"
                )
            elif pattern.pattern_type == PatternType.DELAYED_RESPONSE:
                recommendations.append(
                    "Delayed response typical of food intolerance"
                )
            elif pattern.pattern_type == PatternType.LATE_RESPONSE:
                recommendations.append(
                    "Late response may indicate slow-acting trigger compound"
                )

        if not recommendations:
            recommendations.append("Analysis complete - monitor for patterns")

        return recommendations

    def _compute_confidence(
        self,
        sensitivity_score: float,
        uncertainty: float,
        data_points: int,
    ) -> float:
        """Compute overall model confidence."""
        # Higher confidence with:
        # - Clear sensitivity signal (close to 0 or 1)
        # - Low uncertainty
        # - More data points

        signal_clarity = abs(sensitivity_score - 0.5) * 2  # 0-1
        uncertainty_factor = max(0, 1 - uncertainty * 2)  # 0-1
        data_factor = min(1.0, data_points / 100)  # Saturates at 100 points

        return (signal_clarity + uncertainty_factor + data_factor) / 3


# ==================== Singleton Instance ====================

_lstm_analyzer: Optional[LSTMTemporalPatternAnalyzer] = None


def get_lstm_analyzer() -> LSTMTemporalPatternAnalyzer:
    """Get or create the LSTM analyzer singleton."""
    global _lstm_analyzer

    if _lstm_analyzer is None:
        _lstm_analyzer = LSTMTemporalPatternAnalyzer()
        _lstm_analyzer.initialize()

    return _lstm_analyzer


# ==================== Convenience Functions ====================

def analyze_hrv_patterns(
    hrv_data: List[Dict[str, float]],
    timestamps: Optional[List[float]] = None,
) -> AnalysisResult:
    """
    Analyze HRV temporal patterns (convenience function).

    Args:
        hrv_data: List of HRV measurements
        timestamps: Optional timestamps

    Returns:
        AnalysisResult with detected patterns
    """
    analyzer = get_lstm_analyzer()
    return analyzer.analyze(hrv_data, timestamps)
