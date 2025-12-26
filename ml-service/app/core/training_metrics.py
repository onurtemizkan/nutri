"""
Prometheus Metrics for LSTM Model Training

Provides observability into training performance, model quality, and job status.
Follows the same pattern as app/core/queue/metrics.py.
"""

from typing import Optional
from dataclasses import dataclass, field
from prometheus_client import Counter, Gauge, Histogram, Info


@dataclass
class TrainingMetricsConfig:
    """Configuration for training metrics."""

    enabled: bool = True
    prefix: str = "nutri_ml_training"


@dataclass
class TrainingMetrics:
    """
    Prometheus metrics for LSTM model training.

    Metrics exposed:
        - nutri_ml_training_active_jobs: Currently running training jobs
        - nutri_ml_training_jobs_total: Total training jobs by status
        - nutri_ml_training_duration_seconds: Training time histogram
        - nutri_ml_training_model_quality_r2: R² score histogram
        - nutri_ml_training_model_quality_mape: MAPE histogram
        - nutri_ml_training_epochs: Epochs trained histogram
        - nutri_ml_training_samples: Training samples histogram
    """

    config: TrainingMetricsConfig = field(default_factory=TrainingMetricsConfig)

    # Gauges (current values)
    active_training_jobs: Optional[Gauge] = field(default=None)

    # Counters (cumulative)
    training_jobs_total: Optional[Counter] = field(default=None)
    models_trained_total: Optional[Counter] = field(default=None)

    # Histograms (distributions)
    training_duration_seconds: Optional[Histogram] = field(default=None)
    model_quality_r2: Optional[Histogram] = field(default=None)
    model_quality_mape: Optional[Histogram] = field(default=None)
    epochs_trained: Optional[Histogram] = field(default=None)
    training_samples: Optional[Histogram] = field(default=None)
    model_size_mb: Optional[Histogram] = field(default=None)

    # Info
    config_info: Optional[Info] = field(default=None)

    def __post_init__(self) -> None:
        """Initialize Prometheus metrics."""
        if not self.config.enabled:
            return

        p = self.config.prefix

        # Gauges
        self.active_training_jobs = Gauge(
            f"{p}_active_jobs",
            "Current number of training jobs running",
        )

        # Counters
        self.training_jobs_total = Counter(
            f"{p}_jobs_total",
            "Total training jobs by status",
            ["status"],  # success, error, timeout, cancelled
        )

        self.models_trained_total = Counter(
            f"{p}_models_total",
            "Total models trained by metric and production readiness",
            ["metric", "production_ready"],  # RESTING_HEART_RATE, etc. + true/false
        )

        # Histograms
        # Training duration in seconds (from 10s to 1 hour)
        self.training_duration_seconds = Histogram(
            f"{p}_duration_seconds",
            "Training duration in seconds",
            buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600],
        )

        # R² score histogram (-1 to 1, with focus on 0.5-1.0 range)
        self.model_quality_r2 = Histogram(
            f"{p}_model_quality_r2",
            "R² score distribution of trained models",
            buckets=[-0.5, 0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        )

        # MAPE histogram (0 to 100%, with focus on 0-20% range)
        self.model_quality_mape = Histogram(
            f"{p}_model_quality_mape",
            "MAPE (Mean Absolute Percentage Error) distribution",
            buckets=[1, 2, 5, 8, 10, 15, 20, 30, 50, 100],
        )

        # Epochs trained histogram
        self.epochs_trained = Histogram(
            f"{p}_epochs_trained",
            "Number of epochs trained before completion",
            buckets=[10, 20, 30, 40, 50, 75, 100, 150, 200],
        )

        # Training samples histogram
        self.training_samples = Histogram(
            f"{p}_samples",
            "Number of training samples used",
            buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
        )

        # Model size histogram (in MB)
        self.model_size_mb = Histogram(
            f"{p}_model_size_mb",
            "Trained model size in MB",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        )

        # Info
        self.config_info = Info(
            f"{p}_config",
            "Training configuration",
        )
        self.config_info.info(
            {
                "default_epochs": "50",
                "default_batch_size": "32",
                "default_hidden_dim": "128",
                "default_num_layers": "2",
                "early_stopping_patience": "10",
            }
        )

    @property
    def enabled(self) -> bool:
        """Check if metrics are enabled."""
        return self.config.enabled

    def record_training_start(self) -> None:
        """Record that a training job has started."""
        if self.enabled and self.active_training_jobs:
            self.active_training_jobs.inc()

    def record_training_end(
        self,
        duration_seconds: float,
        status: str,
        metric: str,
        r2_score: float,
        mape: float,
        epochs_trained: int,
        total_samples: int,
        model_size_mb: float,
        is_production_ready: bool,
    ) -> None:
        """
        Record that a training job has completed.

        Args:
            duration_seconds: Training time in seconds
            status: Job status (success, error, timeout, cancelled)
            metric: Health metric being predicted (e.g., RESTING_HEART_RATE)
            r2_score: Model R² score
            mape: Model MAPE (percentage)
            epochs_trained: Number of epochs trained
            total_samples: Number of training samples
            model_size_mb: Model file size in MB
            is_production_ready: Whether model meets quality thresholds
        """
        if not self.enabled:
            return

        # Decrement active jobs
        if self.active_training_jobs:
            self.active_training_jobs.dec()

        # Increment job counter
        if self.training_jobs_total:
            self.training_jobs_total.labels(status=status).inc()

        # Record model counts
        if self.models_trained_total and status == "success":
            self.models_trained_total.labels(
                metric=metric,
                production_ready=str(is_production_ready).lower(),
            ).inc()

        # Record histograms
        if self.training_duration_seconds:
            self.training_duration_seconds.observe(duration_seconds)

        if self.model_quality_r2:
            self.model_quality_r2.observe(r2_score)

        if self.model_quality_mape:
            self.model_quality_mape.observe(mape)

        if self.epochs_trained:
            self.epochs_trained.observe(epochs_trained)

        if self.training_samples:
            self.training_samples.observe(total_samples)

        if self.model_size_mb:
            self.model_size_mb.observe(model_size_mb)

    def record_training_error(self, error_type: str = "error") -> None:
        """
        Record a training error without full metrics.

        Args:
            error_type: Type of error (error, timeout, cancelled)
        """
        if not self.enabled:
            return

        if self.active_training_jobs:
            self.active_training_jobs.dec()

        if self.training_jobs_total:
            self.training_jobs_total.labels(status=error_type).inc()

    def get_active_jobs(self) -> int:
        """Get the current number of active training jobs."""
        if self.enabled and self.active_training_jobs:
            return int(self.active_training_jobs._value.get())
        return 0


# Singleton instance
training_metrics = TrainingMetrics()
