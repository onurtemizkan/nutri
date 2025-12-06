"""
Pydantic schemas for ML predictions (RHR, HRV forecasting).
"""

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionMetric(str, Enum):
    """Health metrics that can be predicted."""

    RHR = "RESTING_HEART_RATE"
    HRV_SDNN = "HEART_RATE_VARIABILITY_SDNN"
    HRV_RMSSD = "HEART_RATE_VARIABILITY_RMSSD"
    SLEEP_DURATION = "SLEEP_DURATION"
    RECOVERY_SCORE = "RECOVERY_SCORE"


class ModelArchitecture(str, Enum):
    """ML model architectures."""

    LSTM = "lstm"  # PyTorch LSTM
    XGBOOST = "xgboost"  # XGBoost regressor
    LINEAR = "linear"  # Linear regression (baseline)


# ============================================================================
# Training Request/Response
# ============================================================================


class TrainModelRequest(BaseModel):
    """Request to train a prediction model."""

    user_id: str = Field(..., description="User ID")
    metric: PredictionMetric = Field(..., description="Health metric to predict")
    architecture: ModelArchitecture = Field(
        default=ModelArchitecture.LSTM, description="Model architecture to use"
    )

    # Training data parameters
    lookback_days: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Days of historical data for training (30-365)",
    )
    sequence_length: int = Field(
        default=30, ge=7, le=60, description="Length of input sequences (7-60 days)"
    )

    # Training hyperparameters
    epochs: int = Field(default=50, ge=10, le=200, description="Training epochs")
    batch_size: int = Field(default=32, ge=8, le=128, description="Batch size")
    learning_rate: float = Field(
        default=0.001, ge=0.0001, le=0.1, description="Learning rate"
    )
    validation_split: float = Field(
        default=0.2, ge=0.1, le=0.3, description="Validation split"
    )

    # Model architecture hyperparameters (for LSTM)
    hidden_dim: int = Field(
        default=128, ge=32, le=512, description="LSTM hidden dimension"
    )
    num_layers: int = Field(default=2, ge=1, le=4, description="Number of LSTM layers")
    dropout: float = Field(default=0.2, ge=0.0, le=0.5, description="Dropout rate")

    # Force retraining
    force_retrain: bool = Field(
        default=False, description="Force retraining even if model exists"
    )


class TrainingMetrics(BaseModel):
    """Training metrics for model evaluation."""

    # Loss metrics
    train_loss: float = Field(..., description="Final training loss (MSE)")
    val_loss: float = Field(..., description="Final validation loss (MSE)")
    best_val_loss: float = Field(..., description="Best validation loss achieved")

    # Accuracy metrics
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    r2_score: float = Field(..., description="R² coefficient of determination")
    mape: float = Field(..., description="Mean Absolute Percentage Error")

    # Training info
    epochs_trained: int = Field(..., description="Number of epochs trained")
    early_stopped: bool = Field(..., description="True if early stopping triggered")
    training_time_seconds: float = Field(..., description="Total training time")


class TrainModelResponse(BaseModel):
    """Response from model training."""

    user_id: str
    metric: PredictionMetric
    architecture: ModelArchitecture

    model_id: str = Field(..., description="Unique model identifier")
    model_version: str = Field(..., description="Model version (e.g., 'v1.0.0')")

    trained_at: datetime
    training_metrics: TrainingMetrics

    # Data info
    total_samples: int = Field(..., description="Total training samples")
    sequence_length: int = Field(..., description="Input sequence length")
    num_features: int = Field(..., description="Number of input features")

    # Model info
    model_path: str = Field(..., description="Path to saved model file")
    model_size_mb: float = Field(..., description="Model file size in MB")

    # Quality assessment
    is_production_ready: bool = Field(
        ..., description="True if model meets quality thresholds"
    )
    quality_issues: List[str] = Field(
        default=[], description="Quality issues if not production-ready"
    )


# ============================================================================
# Prediction Request/Response
# ============================================================================


class PredictRequest(BaseModel):
    """Request to make a prediction."""

    user_id: str = Field(..., description="User ID")
    metric: PredictionMetric = Field(..., description="Health metric to predict")
    target_date: date = Field(..., description="Date to predict for (usually tomorrow)")

    # Model selection
    model_version: Optional[str] = Field(
        None, description="Specific model version (default: latest)"
    )
    architecture: Optional[ModelArchitecture] = Field(
        None, description="Model architecture (default: LSTM)"
    )

    # Feature engineering
    force_recompute_features: bool = Field(
        default=False, description="Force recomputation of input features"
    )


class PredictionResult(BaseModel):
    """Result of a single prediction."""

    metric: PredictionMetric
    target_date: date
    predicted_at: datetime

    # Prediction
    predicted_value: float = Field(..., description="Predicted metric value")
    confidence_interval_lower: float = Field(..., description="95% CI lower bound")
    confidence_interval_upper: float = Field(..., description="95% CI upper bound")
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Prediction confidence (0-1)"
    )

    # Context
    historical_average: float = Field(..., description="User's 30-day average")
    deviation_from_average: float = Field(
        ..., description="Predicted value - historical average"
    )
    percentile: float = Field(
        ..., ge=0, le=100, description="Percentile compared to user's history"
    )

    # Model info
    model_id: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version")
    architecture: ModelArchitecture = Field(..., description="Model architecture")


class PredictResponse(BaseModel):
    """Response containing prediction."""

    user_id: str
    prediction: PredictionResult

    # Input features summary
    features_used: int = Field(..., description="Number of features used")
    sequence_length: int = Field(..., description="Days of input sequence")
    data_quality_score: float = Field(..., ge=0, le=1, description="Input data quality")

    # Interpretation
    interpretation: str = Field(..., description="Natural language interpretation")
    recommendation: Optional[str] = Field(None, description="Actionable recommendation")

    # Cached or fresh
    cached: bool = Field(..., description="True if prediction was cached")


# ============================================================================
# Batch Prediction
# ============================================================================


class BatchPredictRequest(BaseModel):
    """Request to predict multiple metrics at once."""

    user_id: str = Field(..., description="User ID")
    metrics: List[PredictionMetric] = Field(
        ..., description="List of metrics to predict"
    )
    target_date: date = Field(..., description="Date to predict for")

    model_version: Optional[str] = Field(
        None, description="Model version (all metrics)"
    )
    force_recompute_features: bool = Field(default=False)


class BatchPredictResponse(BaseModel):
    """Response containing multiple predictions."""

    user_id: str
    target_date: date
    predicted_at: datetime

    predictions: Dict[str, PredictionResult] = Field(
        ..., description="Predictions keyed by metric name"
    )

    # Overall quality
    overall_data_quality: float = Field(..., ge=0, le=1)
    all_predictions_successful: bool = Field(
        ..., description="True if all metrics predicted successfully"
    )
    failed_metrics: List[str] = Field(
        default=[], description="Metrics that failed to predict"
    )


# ============================================================================
# Model Management
# ============================================================================


class ModelInfo(BaseModel):
    """Information about a trained model."""

    model_id: str
    user_id: str
    metric: PredictionMetric
    architecture: ModelArchitecture

    version: str
    trained_at: datetime
    training_metrics: TrainingMetrics

    sequence_length: int
    num_features: int
    model_size_mb: float

    is_active: bool = Field(..., description="True if currently in use")
    is_production_ready: bool


class ListModelsResponse(BaseModel):
    """Response listing user's models."""

    user_id: str
    models: List[ModelInfo]
    total_models: int


class ModelPerformanceHistory(BaseModel):
    """Historical performance of a model."""

    model_id: str
    metric: PredictionMetric

    # Performance over time
    predictions_made: int = Field(..., description="Total predictions made")
    avg_mae: float = Field(..., description="Average MAE on actuals")
    avg_confidence: float = Field(..., description="Average confidence score")

    # Accuracy when actuals are available
    actual_vs_predicted: List[Dict[str, float]] = Field(
        default=[], description="List of {date, predicted, actual, error}"
    )

    # Drift detection
    is_drifting: bool = Field(..., description="True if performance is degrading")
    should_retrain: bool = Field(..., description="True if model should be retrained")


# ============================================================================
# What-If Scenarios
# ============================================================================


class WhatIfScenario(BaseModel):
    """A hypothetical nutrition/activity scenario."""

    scenario_name: str = Field(..., description="Name of scenario")

    # Hypothetical changes to today's data
    nutrition_changes: Optional[Dict[str, float]] = Field(
        None, description="Changes to nutrition features (e.g., {'protein_daily': 200})"
    )
    activity_changes: Optional[Dict[str, float]] = Field(
        None,
        description="Changes to activity features (e.g., {'workout_intensity': 0.8})",
    )


class WhatIfRequest(BaseModel):
    """Request to test what-if scenarios."""

    user_id: str = Field(..., description="User ID")
    metric: PredictionMetric = Field(..., description="Metric to predict")
    target_date: date = Field(..., description="Date to predict for")

    scenarios: List[WhatIfScenario] = Field(
        ..., min_length=1, max_length=5, description="Scenarios to test (1-5)"
    )


class WhatIfResult(BaseModel):
    """Result of a what-if scenario."""

    scenario_name: str
    predicted_value: float
    confidence_score: float

    change_from_baseline: float = Field(
        ..., description="Difference from baseline prediction"
    )
    percent_change: float = Field(..., description="Percentage change from baseline")


class WhatIfResponse(BaseModel):
    """Response containing what-if scenario results."""

    user_id: str
    metric: PredictionMetric
    target_date: date

    baseline_prediction: float = Field(..., description="Prediction with current data")

    scenarios: List[WhatIfResult] = Field(..., description="Scenario results")

    best_scenario: str = Field(..., description="Scenario with best predicted outcome")
    worst_scenario: str = Field(
        ..., description="Scenario with worst predicted outcome"
    )

    interpretation: str = Field(
        ..., description="Natural language interpretation of scenarios"
    )
