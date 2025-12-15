"""
PyTorch ML models for health metric prediction.

Models available:
- Original LSTM models (lstm.py)
- Advanced models (advanced_lstm.py):
  - EnhancedLSTMWithAttention: LSTM with temporal attention for interpretability
  - BiLSTMWithResiduals: Bidirectional LSTM with residual connections
  - TemporalConvNet: TCN for efficient long-range dependencies
- Ensemble models (ensemble.py):
  - WeightedEnsemble: Learned weight combination
  - StackingEnsemble: Meta-learner based combination
"""

from .lstm import HealthMetricLSTM, LSTMConfig, MultiTaskLSTM, LSTMWithAttention
from .baseline import (
    BaselineLinearModel,
    MovingAverageBaseline,
    ExponentialSmoothingBaseline,
)
from .advanced_lstm import (
    AdvancedLSTMConfig,
    EnhancedLSTMWithAttention,
    BiLSTMWithResiduals,
    TemporalConvNet,
    TCNConfig,
    ModelFactory,
    TemporalAttention,
)
from .ensemble import (
    SimpleAverageEnsemble,
    WeightedEnsemble,
    StackingEnsemble,
    DynamicEnsemble,
    EnsembleFactory,
)

__all__ = [
    # Original models
    "HealthMetricLSTM",
    "LSTMConfig",
    "MultiTaskLSTM",
    "LSTMWithAttention",
    # Baseline models
    "BaselineLinearModel",
    "MovingAverageBaseline",
    "ExponentialSmoothingBaseline",
    # Advanced models
    "AdvancedLSTMConfig",
    "EnhancedLSTMWithAttention",
    "BiLSTMWithResiduals",
    "TemporalConvNet",
    "TCNConfig",
    "ModelFactory",
    "TemporalAttention",
    # Ensemble models
    "SimpleAverageEnsemble",
    "WeightedEnsemble",
    "StackingEnsemble",
    "DynamicEnsemble",
    "EnsembleFactory",
]
