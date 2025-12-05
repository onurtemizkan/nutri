"""
PyTorch ML models for health metric prediction and food classification.
"""

from .lstm import HealthMetricLSTM, LSTMConfig, MultiTaskLSTM, LSTMWithAttention
from .baseline import (
    BaselineLinearModel,
    MovingAverageBaseline,
    ExponentialSmoothingBaseline,
)

# HuggingFace food classifier
from .food_classifier_hf import (
    HuggingFaceFoodClassifier,
    HFClassifierConfig,
    get_hf_food_classifier,
    format_hf_class_name,
    HF_AVAILABLE,
)

__all__ = [
    # Health metric models
    "HealthMetricLSTM",
    "LSTMConfig",
    "MultiTaskLSTM",
    "LSTMWithAttention",
    "BaselineLinearModel",
    "MovingAverageBaseline",
    "ExponentialSmoothingBaseline",
    # Food classification (HuggingFace)
    "HuggingFaceFoodClassifier",
    "HFClassifierConfig",
    "get_hf_food_classifier",
    "format_hf_class_name",
    "HF_AVAILABLE",
]
