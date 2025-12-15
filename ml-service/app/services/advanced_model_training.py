"""
Advanced Model Training Service

Production-ready training service that integrates the best-performing
model architectures (TCN, LSTM+Attention) with the existing ML service.

Features:
- Multi-model architecture support
- Automatic model selection based on metric type
- Hyperparameter optimization integration
- Ensemble model support
- Model versioning and persistence
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml_models.advanced_lstm import (
    AdvancedLSTMConfig,
    BiLSTMWithResiduals,
    EnhancedLSTMWithAttention,
    ModelFactory,
    TCNConfig,
    TemporalConvNet,
)
from app.ml_models.ensemble import (
    EnsembleFactory,
    WeightedEnsemble,
    StackingEnsemble,
)
from app.schemas.predictions import (
    PredictionMetric,
    ModelArchitecture,
    TrainRequest,
    TrainResponse,
    TrainingResult,
    ModelMetrics,
)
from app.services.data_preparation import DataPreparationService


# Mapping of prediction metrics to recommended model architectures
RECOMMENDED_ARCHITECTURES = {
    PredictionMetric.RHR: "tcn",  # TCN performed best for RHR
    PredictionMetric.HRV_SDNN: "tcn",  # TCN performed best for HRV
    PredictionMetric.HRV_RMSSD: "tcn",
    PredictionMetric.SLEEP_DURATION: "lstm_attention",
    PredictionMetric.RECOVERY_SCORE: "lstm_attention",
}

# Default hyperparameters based on optimization experiments
OPTIMIZED_HYPERPARAMS = {
    "tcn": {
        "hidden_channels": 64,
        "num_layers": 5,
        "kernel_size": 3,
        "dropout": 0.2,
    },
    "lstm_attention": {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "attention_dim": 64,
        "attention_heads": 4,
    },
    "bilstm_residual": {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.25,
    },
}


class AdvancedModelTrainingService:
    """
    Production-ready training service for health metric prediction models.

    Supports:
    - Multiple model architectures (TCN, LSTM+Attention, BiLSTM)
    - Automatic architecture selection based on metric type
    - Hyperparameter optimization
    - Ensemble models
    - Model versioning
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_prep_service = DataPreparationService(db)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        # Training configuration
        self.default_epochs = 100
        self.early_stopping_patience = 15
        self.batch_size = 32
        self.learning_rate = 0.001

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def train_model(
        self,
        request: TrainRequest,
        architecture: Optional[str] = None,
        hyperparams: Optional[Dict] = None,
    ) -> TrainResponse:
        """
        Train a model for health metric prediction.

        Args:
            request: Training request with user_id, metric, etc.
            architecture: Model architecture ('tcn', 'lstm_attention', 'bilstm_residual')
                         If None, uses recommended architecture for the metric.
            hyperparams: Custom hyperparameters. If None, uses optimized defaults.

        Returns:
            TrainResponse with training results and model ID
        """
        print(f"\n{'='*70}")
        print(f"🚀 Training Advanced Model for {request.metric.value}")
        print(f"   User: {request.user_id}")
        print(f"   Architecture: {architecture or 'auto-select'}")
        print(f"{'='*70}\n")

        start_time = datetime.now()

        # Select architecture
        if architecture is None:
            architecture = RECOMMENDED_ARCHITECTURES.get(request.metric, "tcn")
            print(f"📌 Auto-selected architecture: {architecture}")

        # Get hyperparameters
        if hyperparams is None:
            hyperparams = OPTIMIZED_HYPERPARAMS.get(architecture, {}).copy()
            print(f"📌 Using optimized hyperparameters: {hyperparams}")

        # Prepare training data
        print("\n📊 Preparing training data...")
        training_data = await self.data_prep_service.prepare_training_data(
            user_id=request.user_id,
            target_metric=request.metric,
            lookback_days=request.lookback_days or 90,
            sequence_length=request.sequence_length or 30,
            validation_split=request.validation_split or 0.2,
        )

        num_features = training_data["num_features"]
        sequence_length = training_data["sequence_length"]

        print(f"✅ Data prepared:")
        print(f"   - Training samples: {len(training_data['X_train'])}")
        print(f"   - Validation samples: {len(training_data['X_val'])}")
        print(f"   - Features: {num_features}")
        print(f"   - Sequence length: {sequence_length}")

        # Create model
        print(f"\n🔧 Creating {architecture} model...")
        model = self._create_model(
            architecture=architecture,
            input_dim=num_features,
            hyperparams=hyperparams,
        )
        model = model.to(self.device)

        print(f"✅ Model created: {model.count_parameters():,} parameters")

        # Train model
        print("\n🏋️ Training model...")
        training_result = self._train_model(
            model=model,
            X_train=training_data["X_train"],
            y_train=training_data["y_train"],
            X_val=training_data["X_val"],
            y_val=training_data["y_val"],
            epochs=request.epochs or self.default_epochs,
        )

        # Calculate validation metrics
        print("\n📈 Calculating validation metrics...")
        val_metrics = self._calculate_metrics(
            model=model,
            X=training_data["X_val"],
            y=training_data["y_val"],
            label_scaler=training_data["label_scaler"],
        )

        print(f"✅ Validation Metrics:")
        print(f"   - MAE: {val_metrics['mae']:.4f}")
        print(f"   - RMSE: {val_metrics['rmse']:.4f}")
        print(f"   - R²: {val_metrics['r2']:.4f}")

        # Save model
        model_id = self._generate_model_id(request.user_id, request.metric, architecture)
        print(f"\n💾 Saving model: {model_id}")

        self._save_model(
            model=model,
            model_id=model_id,
            training_data=training_data,
            architecture=architecture,
            hyperparams=hyperparams,
            metrics=val_metrics,
            training_result=training_result,
        )

        training_time = (datetime.now() - start_time).total_seconds()

        print(f"\n{'='*70}")
        print(f"✅ Training complete!")
        print(f"   Model ID: {model_id}")
        print(f"   Training time: {training_time:.1f}s")
        print(f"{'='*70}\n")

        # Create response
        return TrainResponse(
            user_id=request.user_id,
            metric=request.metric,
            model_id=model_id,
            architecture=ModelArchitecture(architecture.upper()),
            training_samples=len(training_data["X_train"]),
            validation_samples=len(training_data["X_val"]),
            sequence_length=sequence_length,
            features_used=num_features,
            training_time_seconds=training_time,
            epochs_completed=training_result["epochs_completed"],
            best_epoch=training_result["best_epoch"],
            training_loss=training_result["final_train_loss"],
            validation_loss=training_result["best_val_loss"],
            metrics=ModelMetrics(
                mae=val_metrics["mae"],
                rmse=val_metrics["rmse"],
                mape=val_metrics["mape"],
                r2_score=val_metrics["r2"],
            ),
            model_version="v2.0.0",  # Advanced models
        )

    async def train_ensemble(
        self,
        request: TrainRequest,
        architectures: List[str] = None,
    ) -> TrainResponse:
        """
        Train an ensemble of models for improved predictions.

        Args:
            request: Training request
            architectures: List of architectures to include.
                          Default: ['tcn', 'lstm_attention', 'bilstm_residual']

        Returns:
            TrainResponse for the ensemble model
        """
        if architectures is None:
            architectures = ["tcn", "lstm_attention"]

        print(f"\n{'='*70}")
        print(f"🎯 Training Ensemble Model for {request.metric.value}")
        print(f"   Architectures: {architectures}")
        print(f"{'='*70}\n")

        start_time = datetime.now()

        # Prepare data
        training_data = await self.data_prep_service.prepare_training_data(
            user_id=request.user_id,
            target_metric=request.metric,
            lookback_days=request.lookback_days or 90,
            sequence_length=request.sequence_length or 30,
            validation_split=request.validation_split or 0.2,
        )

        num_features = training_data["num_features"]

        # Train individual models
        models = []
        individual_metrics = []

        for arch in architectures:
            print(f"\n📌 Training {arch} base model...")

            hyperparams = OPTIMIZED_HYPERPARAMS.get(arch, {}).copy()
            model = self._create_model(
                architecture=arch,
                input_dim=num_features,
                hyperparams=hyperparams,
            )
            model = model.to(self.device)

            self._train_model(
                model=model,
                X_train=training_data["X_train"],
                y_train=training_data["y_train"],
                X_val=training_data["X_val"],
                y_val=training_data["y_val"],
                epochs=50,  # Fewer epochs for ensemble base models
            )

            metrics = self._calculate_metrics(
                model=model,
                X=training_data["X_val"],
                y=training_data["y_val"],
                label_scaler=training_data["label_scaler"],
            )

            models.append(model)
            individual_metrics.append(metrics)
            print(f"   {arch} MAE: {metrics['mae']:.4f}")

        # Create weighted ensemble
        print("\n🔗 Creating weighted ensemble...")
        ensemble = WeightedEnsemble(models)

        # Train ensemble weights
        X_val_device = training_data["X_val"].to(self.device)
        y_val_device = training_data["y_val"].to(self.device)

        ensemble = ensemble.to(self.device)
        ensemble.train_weights(X_val_device, y_val_device, epochs=100)

        # Calculate ensemble metrics
        ensemble_metrics = self._calculate_metrics(
            model=ensemble,
            X=training_data["X_val"],
            y=training_data["y_val"],
            label_scaler=training_data["label_scaler"],
        )

        print(f"\n📊 Ensemble Performance:")
        print(f"   Ensemble MAE: {ensemble_metrics['mae']:.4f}")
        print(f"   Weights: {ensemble.get_weights_dict()}")

        # Save ensemble
        model_id = self._generate_model_id(
            request.user_id, request.metric, "ensemble"
        )

        self._save_ensemble(
            ensemble=ensemble,
            model_id=model_id,
            training_data=training_data,
            architectures=architectures,
            metrics=ensemble_metrics,
        )

        training_time = (datetime.now() - start_time).total_seconds()

        return TrainResponse(
            user_id=request.user_id,
            metric=request.metric,
            model_id=model_id,
            architecture=ModelArchitecture.ENSEMBLE,
            training_samples=len(training_data["X_train"]),
            validation_samples=len(training_data["X_val"]),
            sequence_length=training_data["sequence_length"],
            features_used=num_features,
            training_time_seconds=training_time,
            epochs_completed=50,
            best_epoch=50,
            training_loss=0.0,
            validation_loss=ensemble_metrics["mse"],
            metrics=ModelMetrics(
                mae=ensemble_metrics["mae"],
                rmse=ensemble_metrics["rmse"],
                mape=ensemble_metrics["mape"],
                r2_score=ensemble_metrics["r2"],
            ),
            model_version="v2.0.0-ensemble",
        )

    def _create_model(
        self,
        architecture: str,
        input_dim: int,
        hyperparams: Dict,
    ) -> nn.Module:
        """Create a model instance."""
        if architecture == "tcn":
            return ModelFactory.create(
                model_type="tcn",
                input_dim=input_dim,
                hidden_dim=hyperparams.get("hidden_channels", 64),
                num_layers=hyperparams.get("num_layers", 5),
                dropout=hyperparams.get("dropout", 0.2),
            )
        elif architecture == "lstm_attention":
            return ModelFactory.create(
                model_type="lstm_attention",
                input_dim=input_dim,
                hidden_dim=hyperparams.get("hidden_dim", 128),
                num_layers=hyperparams.get("num_layers", 2),
                dropout=hyperparams.get("dropout", 0.2),
                attention_dim=hyperparams.get("attention_dim", 64),
            )
        elif architecture == "bilstm_residual":
            return ModelFactory.create(
                model_type="bilstm_residual",
                input_dim=input_dim,
                hidden_dim=hyperparams.get("hidden_dim", 128),
                num_layers=hyperparams.get("num_layers", 2),
                dropout=hyperparams.get("dropout", 0.25),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _train_model(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
    ) -> Dict:
        """Train a model with early stopping."""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                X_val_device = X_val.to(self.device)
                y_val_device = y_val.to(self.device)
                val_pred = model(X_val_device)
                val_loss = criterion(val_pred, y_val_device).item()

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"   Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        return {
            "epochs_completed": epoch + 1,
            "best_epoch": best_epoch,
            "final_train_loss": train_loss,
            "best_val_loss": best_val_loss,
        }

    def _calculate_metrics(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        label_scaler: StandardScaler,
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        model.eval()
        with torch.no_grad():
            X_device = X.to(self.device)
            predictions = model(X_device).cpu().numpy()

        # Denormalize
        predictions_real = label_scaler.inverse_transform(predictions).flatten()
        actuals_real = label_scaler.inverse_transform(y.numpy()).flatten()

        mae = mean_absolute_error(actuals_real, predictions_real)
        mse = mean_squared_error(actuals_real, predictions_real)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals_real, predictions_real)

        # MAPE
        mask = actuals_real != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs(
                (actuals_real[mask] - predictions_real[mask]) / actuals_real[mask]
            )) * 100
        else:
            mape = 0.0

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
        }

    def _generate_model_id(
        self,
        user_id: str,
        metric: PredictionMetric,
        architecture: str,
    ) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{user_id}_{metric.value}_{architecture}_{timestamp}"

    def _save_model(
        self,
        model: nn.Module,
        model_id: str,
        training_data: Dict,
        architecture: str,
        hyperparams: Dict,
        metrics: Dict,
        training_result: Dict,
    ) -> None:
        """Save model and artifacts to disk."""
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        # Save PyTorch model
        torch.save(model.state_dict(), model_dir / "model.pt")

        # Save scalers
        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(training_data["scaler"], f)

        with open(model_dir / "label_scaler.pkl", "wb") as f:
            pickle.dump(training_data["label_scaler"], f)

        # Save feature names
        with open(model_dir / "feature_names.pkl", "wb") as f:
            pickle.dump(training_data["feature_names"], f)

        # Save config
        config = {
            "architecture": architecture,
            "hyperparams": hyperparams,
            "input_dim": training_data["num_features"],
            "sequence_length": training_data["sequence_length"],
        }
        with open(model_dir / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        # Save metadata
        metadata = {
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "architecture": architecture,
            "validation_metrics": metrics,
            "training_result": training_result,
            "model_version": "v2.0.0",
        }
        with open(model_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def _save_ensemble(
        self,
        ensemble: WeightedEnsemble,
        model_id: str,
        training_data: Dict,
        architectures: List[str],
        metrics: Dict,
    ) -> None:
        """Save ensemble model and artifacts."""
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        # Save ensemble
        torch.save(ensemble.state_dict(), model_dir / "ensemble.pt")

        # Save individual model states
        for i, model in enumerate(ensemble.models):
            torch.save(model.state_dict(), model_dir / f"base_model_{i}.pt")

        # Save scalers and metadata
        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(training_data["scaler"], f)

        with open(model_dir / "label_scaler.pkl", "wb") as f:
            pickle.dump(training_data["label_scaler"], f)

        with open(model_dir / "feature_names.pkl", "wb") as f:
            pickle.dump(training_data["feature_names"], f)

        config = {
            "architecture": "ensemble",
            "base_architectures": architectures,
            "input_dim": training_data["num_features"],
            "sequence_length": training_data["sequence_length"],
            "weights": ensemble.get_weights_dict(),
        }
        with open(model_dir / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        metadata = {
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "architecture": "ensemble",
            "base_architectures": architectures,
            "validation_metrics": metrics,
            "model_version": "v2.0.0-ensemble",
        }
        with open(model_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)


# Convenience function for quick training
async def train_advanced_model(
    db: AsyncSession,
    user_id: str,
    metric: str,
    architecture: Optional[str] = None,
) -> TrainResponse:
    """
    Quick function to train an advanced model.

    Example:
        response = await train_advanced_model(
            db=session,
            user_id="user123",
            metric="RESTING_HEART_RATE",
            architecture="tcn",  # or None for auto-select
        )
    """
    service = AdvancedModelTrainingService(db)

    request = TrainRequest(
        user_id=user_id,
        metric=PredictionMetric(metric),
    )

    return await service.train_model(request, architecture=architecture)
