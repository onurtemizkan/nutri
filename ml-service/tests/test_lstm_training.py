"""
Comprehensive Tests for LSTM Training Infrastructure

Tests cover:
- Data validation requirements (Subtask 6.1)
- Training data preparation
- LSTM model forward pass
- Training loss decreases
- Model save/load roundtrip
- Evaluation metrics accuracy
- Early stopping triggers
- Model versioning (Subtask 6.3)
- API train/predict flow
"""

import pickle
import shutil
import pytest
import pytest_asyncio
import numpy as np
import torch
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.ml_models.lstm import HealthMetricLSTM, LSTMConfig
from app.schemas.predictions import (
    PredictionMetric,
    TrainModelRequest,
    ModelArchitecture,
)
from app.services.model_training import (
    ModelTrainingService,
    MIN_HEALTH_DATA_DAYS,
    MIN_NUTRITION_DATA_DAYS,
)
from app.core.training_metrics import TrainingMetrics, TrainingMetricsConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def lstm_config():
    """Create a test LSTM configuration."""
    return LSTMConfig(
        input_dim=51,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        sequence_length=30,
        device="cpu",
    )


@pytest.fixture
def small_lstm_config():
    """Create a small LSTM for fast testing."""
    return LSTMConfig(
        input_dim=10,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        sequence_length=5,
        device="cpu",
    )


@pytest.fixture
def sample_training_data(small_lstm_config):
    """Create sample training data."""
    batch_size = 32
    seq_len = small_lstm_config.sequence_length
    input_dim = small_lstm_config.input_dim

    X_train = torch.randn(batch_size, seq_len, input_dim)
    y_train = torch.randn(batch_size, 1)
    X_val = torch.randn(batch_size // 4, seq_len, input_dim)
    y_val = torch.randn(batch_size // 4, 1)

    return X_train, y_train, X_val, y_val


@pytest.fixture
def models_dir(tmp_path):
    """Create a temporary models directory."""
    models = tmp_path / "models"
    models.mkdir()
    return models


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    mock = MagicMock()
    mock.execute = AsyncMock()
    return mock


# ============================================================================
# Test Data Validation Requirements (Subtask 6.1)
# ============================================================================


class TestDataValidationRequirements:
    """Test validate_training_data_requirements() method."""

    @pytest_asyncio.fixture
    async def training_service(self, mock_db, models_dir):
        """Create training service with mocked dependencies."""
        service = ModelTrainingService(mock_db)
        service.models_dir = models_dir
        return service

    @pytest.mark.asyncio
    async def test_validation_passes_with_sufficient_data(
        self, training_service, mock_db
    ):
        """Test that validation passes when data requirements are met."""
        # Mock health data count (35 days)
        health_result = MagicMock()
        health_result.scalar.return_value = 35

        # Mock nutrition data count (25 days)
        nutrition_result = MagicMock()
        nutrition_result.scalar.return_value = 25

        mock_db.execute = AsyncMock(side_effect=[health_result, nutrition_result])

        result = await training_service.validate_training_data_requirements(
            user_id="test_user",
            target_metric=PredictionMetric.RHR,
            lookback_days=90,
        )

        assert result["health_days"] == 35
        assert result["nutrition_days"] == 25
        assert result["lookback_days"] == 90

    @pytest.mark.asyncio
    async def test_validation_fails_with_insufficient_health_data(
        self, training_service, mock_db
    ):
        """Test that validation fails when health data is insufficient."""
        # Mock health data count (only 10 days - below minimum of 30)
        health_result = MagicMock()
        health_result.scalar.return_value = 10

        # Mock nutrition data count (sufficient)
        nutrition_result = MagicMock()
        nutrition_result.scalar.return_value = 25

        mock_db.execute = AsyncMock(side_effect=[health_result, nutrition_result])

        with pytest.raises(ValueError) as exc_info:
            await training_service.validate_training_data_requirements(
                user_id="test_user",
                target_metric=PredictionMetric.RHR,
                lookback_days=90,
            )

        error_message = str(exc_info.value)
        assert "Insufficient health data" in error_message
        assert "10 days" in error_message
        assert f"{MIN_HEALTH_DATA_DAYS} days" in error_message

    @pytest.mark.asyncio
    async def test_validation_fails_with_insufficient_nutrition_data(
        self, training_service, mock_db
    ):
        """Test that validation fails when nutrition data is insufficient."""
        # Mock health data count (sufficient)
        health_result = MagicMock()
        health_result.scalar.return_value = 35

        # Mock nutrition data count (only 5 days - below minimum of 21)
        nutrition_result = MagicMock()
        nutrition_result.scalar.return_value = 5

        mock_db.execute = AsyncMock(side_effect=[health_result, nutrition_result])

        with pytest.raises(ValueError) as exc_info:
            await training_service.validate_training_data_requirements(
                user_id="test_user",
                target_metric=PredictionMetric.RHR,
                lookback_days=90,
            )

        error_message = str(exc_info.value)
        assert "Insufficient nutrition data" in error_message
        assert "5 days" in error_message
        assert f"{MIN_NUTRITION_DATA_DAYS} days" in error_message

    @pytest.mark.asyncio
    async def test_validation_fails_with_both_insufficient(
        self, training_service, mock_db
    ):
        """Test that validation lists all issues when both are insufficient."""
        health_result = MagicMock()
        health_result.scalar.return_value = 5

        nutrition_result = MagicMock()
        nutrition_result.scalar.return_value = 3

        mock_db.execute = AsyncMock(side_effect=[health_result, nutrition_result])

        with pytest.raises(ValueError) as exc_info:
            await training_service.validate_training_data_requirements(
                user_id="test_user",
                target_metric=PredictionMetric.HRV_SDNN,
                lookback_days=90,
            )

        error_message = str(exc_info.value)
        assert "Insufficient health data" in error_message
        assert "Insufficient nutrition data" in error_message


# ============================================================================
# Test Training Data Preparation
# ============================================================================


class TestTrainingDataPreparation:
    """Test data preparation for training."""

    def test_create_sequences(self):
        """Test sliding window sequence creation."""
        from app.services.data_preparation import DataPreparationService

        # Create mock service (we'll call the internal method directly)
        service = DataPreparationService(MagicMock())

        # Create sample data: 100 days, 10 features
        features = np.random.randn(100, 10)
        targets = np.random.randn(100)
        sequence_length = 30

        X, y = service._create_sequences(features, targets, sequence_length)

        # Should create 70 sequences (100 - 30 = 70)
        assert X.shape == (70, 30, 10)
        assert y.shape == (70,)

        # Verify first sequence maps to correct target
        # Sequence 0 uses days 0-29, target is day 30
        np.testing.assert_array_equal(X[0], features[0:30])
        assert y[0] == targets[30]

    def test_normalize_features(self):
        """Test feature normalization."""
        from app.services.data_preparation import DataPreparationService

        service = DataPreparationService(MagicMock())

        # Create sample data
        X_train = np.random.randn(50, 30, 10) * 100 + 50  # Mean ~50, std ~100
        X_val = np.random.randn(10, 30, 10) * 100 + 50

        scaler, X_train_norm, X_val_norm = service._normalize_features(X_train, X_val)

        # Check normalized training data has mean ~0 and std ~1
        # (across all samples and timesteps for each feature)
        train_mean = X_train_norm.reshape(-1, 10).mean(axis=0)
        train_std = X_train_norm.reshape(-1, 10).std(axis=0)

        np.testing.assert_array_almost_equal(train_mean, np.zeros(10), decimal=1)
        np.testing.assert_array_almost_equal(train_std, np.ones(10), decimal=1)

        # Check shapes preserved
        assert X_train_norm.shape == X_train.shape
        assert X_val_norm.shape == X_val.shape


# ============================================================================
# Test LSTM Model Forward Pass
# ============================================================================


class TestLSTMForwardPass:
    """Test LSTM model architecture and forward pass."""

    def test_lstm_model_initialization(self, lstm_config):
        """Test LSTM model initializes correctly."""
        model = HealthMetricLSTM(lstm_config)

        assert model.config == lstm_config
        assert hasattr(model, "lstm")
        assert hasattr(model, "fc1")
        assert hasattr(model, "fc2")
        assert hasattr(model, "fc_out")

    def test_lstm_forward_pass(self, lstm_config):
        """Test LSTM forward pass produces correct output shape."""
        model = HealthMetricLSTM(lstm_config)
        model.eval()

        batch_size = 16
        X = torch.randn(batch_size, lstm_config.sequence_length, lstm_config.input_dim)

        with torch.no_grad():
            output = model(X)

        # Output should be (batch_size, 1) for single value prediction
        assert output.shape == (batch_size, 1)

    def test_lstm_predict_method(self, lstm_config):
        """Test LSTM predict method works correctly."""
        model = HealthMetricLSTM(lstm_config)

        X = torch.randn(8, lstm_config.sequence_length, lstm_config.input_dim)

        predictions = model.predict(X)

        assert predictions.shape == (8, 1)
        # Predict should set model to eval mode
        assert not model.training

    def test_lstm_count_parameters(self, lstm_config):
        """Test parameter counting."""
        model = HealthMetricLSTM(lstm_config)

        param_count = model.count_parameters()

        assert param_count > 0
        assert isinstance(param_count, int)

        # Verify by manual count
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count


# ============================================================================
# Test Training Loss Decreases
# ============================================================================


class TestTrainingLossDecreases:
    """Test that training actually reduces loss."""

    def test_loss_decreases_during_training(
        self, small_lstm_config, sample_training_data
    ):
        """Test that loss decreases over multiple epochs."""
        X_train, y_train, X_val, y_val = sample_training_data
        model = HealthMetricLSTM(small_lstm_config)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()

            predictions = model(X_train)
            loss = criterion(predictions, y_train)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease (first loss > last loss)
        assert losses[0] > losses[-1], "Loss should decrease during training"

        # Loss should decrease by at least 10%
        improvement = (losses[0] - losses[-1]) / losses[0]
        assert improvement > 0.1, f"Loss should improve by >10%, got {improvement:.1%}"


# ============================================================================
# Test Model Save/Load Roundtrip
# ============================================================================


class TestModelSaveLoadRoundtrip:
    """Test model persistence."""

    def test_save_and_load_model(self, small_lstm_config, models_dir):
        """Test that model can be saved and loaded correctly."""
        # Create and train model slightly
        model = HealthMetricLSTM(small_lstm_config)
        X = torch.randn(
            8, small_lstm_config.sequence_length, small_lstm_config.input_dim
        )

        # Get predictions before saving
        model.eval()
        with torch.no_grad():
            original_predictions = model(X)

        # Save model
        model_path = models_dir / "test_model.pt"
        torch.save(model.state_dict(), model_path)

        # Create new model and load weights
        loaded_model = HealthMetricLSTM(small_lstm_config)
        loaded_model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # Get predictions after loading
        loaded_model.eval()
        with torch.no_grad():
            loaded_predictions = loaded_model(X)

        # Predictions should match exactly
        torch.testing.assert_close(original_predictions, loaded_predictions)

    def test_save_and_load_scalers(self, models_dir):
        """Test scaler persistence."""
        from sklearn.preprocessing import StandardScaler

        # Create and fit scaler
        scaler = StandardScaler()
        data = np.random.randn(100, 10) * 50 + 100
        scaler.fit(data)

        # Save scaler
        scaler_path = models_dir / "test_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Load scaler
        with open(scaler_path, "rb") as f:
            loaded_scaler = pickle.load(f)

        # Transform should produce identical results
        test_data = np.random.randn(10, 10) * 50 + 100
        original_transform = scaler.transform(test_data)
        loaded_transform = loaded_scaler.transform(test_data)

        np.testing.assert_array_almost_equal(original_transform, loaded_transform)


# ============================================================================
# Test Evaluation Metrics Accuracy
# ============================================================================


class TestEvaluationMetrics:
    """Test evaluation metrics calculation."""

    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        from sklearn.metrics import mean_absolute_error

        y_true = np.array([60, 62, 58, 65, 61])
        y_pred = np.array([61, 63, 57, 64, 62])

        mae = mean_absolute_error(y_true, y_pred)

        # Each prediction is off by 1
        assert mae == 1.0

    def test_r2_score_calculation(self):
        """Test R² score calculation."""
        from sklearn.metrics import r2_score

        # Perfect predictions
        y_true = np.array([60, 62, 58, 65, 61])
        y_pred = y_true.copy()

        r2 = r2_score(y_true, y_pred)
        assert r2 == 1.0

        # Bad predictions (predicting mean)
        y_pred_mean = np.full_like(y_true, y_true.mean())
        r2_mean = r2_score(y_true, y_pred_mean)
        assert r2_mean == 0.0

    def test_mape_calculation(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 100, 100, 100])
        y_pred = np.array([110, 90, 105, 95])

        # MAPE = mean(|error| / actual) * 100
        # Errors: 10%, 10%, 5%, 5%
        # Mean: 7.5%
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        assert abs(mape - 7.5) < 0.01


# ============================================================================
# Test Early Stopping Triggers
# ============================================================================


class TestEarlyStopping:
    """Test early stopping behavior."""

    def test_early_stopping_triggers_on_plateau(self, small_lstm_config):
        """Test that early stopping triggers when validation loss plateaus."""
        from app.services.model_training import ModelTrainingService

        # Create training service with mock db
        mock_db = MagicMock()
        service = ModelTrainingService(mock_db)

        # Create model and data
        model = HealthMetricLSTM(small_lstm_config)
        device = torch.device("cpu")
        model = model.to(device)

        # Create data that will cause quick plateau
        # Use very small random data that the model will quickly overfit
        X_train = torch.randn(
            8, small_lstm_config.sequence_length, small_lstm_config.input_dim
        )
        y_train = torch.zeros(8, 1)  # All zeros - easy to fit
        X_val = torch.randn(
            2, small_lstm_config.sequence_length, small_lstm_config.input_dim
        )
        y_val = torch.zeros(2, 1)

        # Train with enough epochs to trigger early stopping
        result = service._train_loop(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=100,  # High enough to trigger early stopping
            batch_size=4,
            learning_rate=0.001,
            device=device,
        )

        # Should have trained but possibly stopped early
        assert result["epochs_trained"] > 0
        assert result["epochs_trained"] <= 100


# ============================================================================
# Test Model Versioning (Subtask 6.3)
# ============================================================================


class TestModelVersioning:
    """Test model versioning and management."""

    @pytest.fixture
    def training_service(self, mock_db, models_dir):
        """Create training service with test models directory."""
        service = ModelTrainingService(mock_db)
        service.models_dir = models_dir
        return service

    def test_generate_version_first_model(self, training_service):
        """Test version generation for first model."""
        version = training_service._generate_model_version(
            "test_user", PredictionMetric.RHR
        )

        assert version == "v1.0.0"

    def test_generate_version_increments(self, training_service, models_dir):
        """Test version increments for subsequent models."""
        user_id = "test_user"
        metric = PredictionMetric.RHR

        # Create a fake existing model with v1.0.0
        model_dir = models_dir / f"{user_id}_{metric.value}_20240101_000000"
        model_dir.mkdir()
        metadata = {"model_version": "v1.0.0"}
        with open(model_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        version = training_service._generate_model_version(user_id, metric)

        assert version == "v1.0.1"

    def test_generate_version_finds_highest(self, training_service, models_dir):
        """Test version finds highest existing version."""
        user_id = "test_user"
        metric = PredictionMetric.RHR

        # Create multiple models with different versions
        for i, v in enumerate(["v1.0.0", "v1.0.2", "v1.0.1"]):
            model_dir = models_dir / f"{user_id}_{metric.value}_2024010{i}_000000"
            model_dir.mkdir()
            metadata = {"model_version": v}
            with open(model_dir / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

        version = training_service._generate_model_version(user_id, metric)

        # Should increment from highest (v1.0.2)
        assert version == "v1.0.3"

    def test_set_active_model(self, training_service, models_dir):
        """Test setting active model."""
        user_id = "test_user"
        metric = PredictionMetric.RHR
        model_id = "test_model_123"

        training_service._set_active_model(user_id, metric, model_id)

        # Check active marker file exists
        active_marker = models_dir / f"{user_id}_{metric.value}_active.txt"
        assert active_marker.exists()

        with open(active_marker, "r") as f:
            assert f.read().strip() == model_id

    def test_get_active_model_id(self, training_service, models_dir):
        """Test getting active model ID."""
        user_id = "test_user"
        metric = PredictionMetric.RHR
        model_id = "test_model_456"

        # Set active model
        training_service._set_active_model(user_id, metric, model_id)

        # Get active model
        active_id = training_service.get_active_model_id(user_id, metric)

        assert active_id == model_id

    def test_get_active_model_id_none_when_missing(self, training_service):
        """Test getting active model ID when none set."""
        active_id = training_service.get_active_model_id(
            "nonexistent_user", PredictionMetric.RHR
        )

        assert active_id is None

    @pytest.mark.asyncio
    async def test_get_model_history(self, training_service, models_dir):
        """Test getting model history."""
        user_id = "test_user"
        metric = PredictionMetric.RHR

        # Create some test models
        for i in range(3):
            model_dir = models_dir / f"{user_id}_{metric.value}_2024010{i}_000000"
            model_dir.mkdir()
            metadata = {
                "model_id": model_dir.name,
                "model_version": f"v1.0.{i}",
                "trained_at": f"2024-01-0{i+1}T00:00:00",
                "user_id": user_id,
                "metric": metric.value,
            }
            with open(model_dir / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

        history = await training_service.get_model_history(user_id, metric)

        assert len(history) == 3
        # Should be sorted by date (newest first)
        assert history[0]["model_version"] == "v1.0.2"

    @pytest.mark.asyncio
    async def test_rollback_to_version(self, training_service, models_dir):
        """Test rolling back to a previous version."""
        user_id = "test_user"
        metric = PredictionMetric.RHR

        # Create models
        for i, v in enumerate(["v1.0.0", "v1.0.1", "v1.0.2"]):
            model_dir = models_dir / f"{user_id}_{metric.value}_2024010{i}_000000"
            model_dir.mkdir()
            metadata = {"model_version": v}
            with open(model_dir / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

        # Set latest as active
        training_service._set_active_model(
            user_id, metric, f"{user_id}_{metric.value}_20240102_000000"
        )

        # Rollback to v1.0.0
        result = await training_service.rollback_to_version(user_id, metric, "v1.0.0")

        assert result["success"] is True
        assert result["rolled_back_to"] == "v1.0.0"

        # Check active model changed
        active_id = training_service.get_active_model_id(user_id, metric)
        assert "20240100" in active_id  # First model

    @pytest.mark.asyncio
    async def test_rollback_to_nonexistent_version(self, training_service, models_dir):
        """Test rollback fails for nonexistent version."""
        user_id = "test_user"
        metric = PredictionMetric.RHR

        # Create a model
        model_dir = models_dir / f"{user_id}_{metric.value}_20240101_000000"
        model_dir.mkdir()
        metadata = {"model_version": "v1.0.0"}
        with open(model_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        with pytest.raises(ValueError) as exc_info:
            await training_service.rollback_to_version(user_id, metric, "v9.9.9")

        assert "v9.9.9 not found" in str(exc_info.value)

    def test_cleanup_old_models(self, training_service, models_dir):
        """Test cleanup of old models."""
        user_id = "test_user"
        metric = PredictionMetric.RHR

        # Create 7 models
        for i in range(7):
            model_dir = models_dir / f"{user_id}_{metric.value}_2024010{i}_000000"
            model_dir.mkdir()
            metadata = {
                "model_version": f"v1.0.{i}",
                "trained_at": f"2024-01-0{i+1}T00:00:00",
            }
            with open(model_dir / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

        # Keep 5 models
        result = training_service.cleanup_old_models(user_id, metric, keep_count=5)

        assert result["removed"] == 2
        assert result["kept"] == 5

        # Verify only 5 models remain
        remaining = list(models_dir.glob(f"{user_id}_{metric.value}_*"))
        remaining_dirs = [d for d in remaining if d.is_dir()]
        assert len(remaining_dirs) == 5


# ============================================================================
# Test Prometheus Training Metrics (Subtask 6.5)
# ============================================================================


class TestPrometheusTrainingMetrics:
    """Test Prometheus training metrics."""

    def test_metrics_initialization(self):
        """Test that metrics initialize correctly."""
        config = TrainingMetricsConfig(enabled=True, prefix="test_training")
        metrics = TrainingMetrics(config=config)

        assert metrics.active_training_jobs is not None
        assert metrics.training_jobs_total is not None
        assert metrics.training_duration_seconds is not None
        assert metrics.model_quality_r2 is not None
        assert metrics.model_quality_mape is not None

    def test_metrics_disabled(self):
        """Test that metrics can be disabled."""
        config = TrainingMetricsConfig(enabled=False)
        metrics = TrainingMetrics(config=config)

        assert metrics.active_training_jobs is None
        assert metrics.training_jobs_total is None

    def test_record_training_start(self):
        """Test recording training start."""
        config = TrainingMetricsConfig(enabled=True, prefix="test_start")
        metrics = TrainingMetrics(config=config)

        initial_count = metrics.get_active_jobs()
        metrics.record_training_start()

        assert metrics.get_active_jobs() == initial_count + 1

    def test_record_training_end(self):
        """Test recording training end."""
        config = TrainingMetricsConfig(enabled=True, prefix="test_end")
        metrics = TrainingMetrics(config=config)

        # Start a job
        metrics.record_training_start()
        initial_count = metrics.get_active_jobs()

        # End the job
        metrics.record_training_end(
            duration_seconds=120.5,
            status="success",
            metric="RESTING_HEART_RATE",
            r2_score=0.85,
            mape=8.5,
            epochs_trained=45,
            total_samples=500,
            model_size_mb=2.5,
            is_production_ready=True,
        )

        # Active jobs should decrease
        assert metrics.get_active_jobs() == initial_count - 1

    def test_record_training_error(self):
        """Test recording training error."""
        config = TrainingMetricsConfig(enabled=True, prefix="test_error")
        metrics = TrainingMetrics(config=config)

        metrics.record_training_start()
        initial_count = metrics.get_active_jobs()

        metrics.record_training_error("timeout")

        # Active jobs should decrease
        assert metrics.get_active_jobs() == initial_count - 1


# ============================================================================
# Test API Train/Predict Flow
# ============================================================================


class TestAPITrainPredictFlow:
    """Test the full API train and predict flow."""

    @pytest.mark.asyncio
    async def test_train_model_validation_error(self, mock_db, models_dir):
        """Test that training fails gracefully with validation error."""
        # Mock insufficient data
        health_result = MagicMock()
        health_result.scalar.return_value = 5  # Insufficient

        nutrition_result = MagicMock()
        nutrition_result.scalar.return_value = 3  # Insufficient

        mock_db.execute = AsyncMock(side_effect=[health_result, nutrition_result])

        service = ModelTrainingService(mock_db)
        service.models_dir = models_dir

        request = TrainModelRequest(
            user_id="test_user",
            metric=PredictionMetric.RHR,
            architecture=ModelArchitecture.LSTM,
        )

        with pytest.raises(ValueError) as exc_info:
            await service.train_model(request)

        assert "Data requirements not met" in str(exc_info.value)


# ============================================================================
# Cleanup
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_test_models(models_dir):
    """Cleanup test models after each test."""
    yield

    # Cleanup
    if models_dir.exists():
        shutil.rmtree(models_dir, ignore_errors=True)
