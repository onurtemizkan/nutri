"""
PyTorch Model Training Service for Health Metric Prediction

Handles:
1. LSTM model training with early stopping
2. Training/validation monitoring
3. Model checkpointing
4. Performance metrics (MAE, RMSE, R², MAPE)
5. Model persistence and versioning
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml_models.lstm import HealthMetricLSTM, LSTMConfig
from app.schemas.predictions import (
    TrainModelRequest,
    TrainModelResponse,
    TrainingMetrics,
)
from app.services.data_preparation import DataPreparationService


class ModelTrainingService:
    """
    Service for training PyTorch LSTM models for health metric prediction.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_prep_service = DataPreparationService(db)

        # Model storage directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    # ========================================================================
    # Main Training Entry Point
    # ========================================================================

    async def train_model(self, request: TrainModelRequest) -> TrainModelResponse:
        """
        Train a PyTorch LSTM model for health metric prediction.

        Steps:
        1. Prepare training data (sequences, normalization)
        2. Initialize LSTM model
        3. Train with early stopping
        4. Evaluate on validation set
        5. Save model artifacts
        6. Return training metrics

        Args:
            request: Training configuration

        Returns:
            TrainModelResponse with model ID, metrics, and paths
        """
        print(f"\n{'='*70}")
        print(f"🚀 Starting model training for {request.metric.value}")
        print(f"{'='*70}\n")

        # Step 1: Prepare training data
        print("📊 Step 1: Preparing training data...")
        training_data = await self.data_prep_service.prepare_training_data(
            user_id=request.user_id,
            target_metric=request.metric,
            lookback_days=request.lookback_days,
            sequence_length=request.sequence_length,
            validation_split=request.validation_split,
        )

        X_train = training_data["X_train"]
        y_train = training_data["y_train"]
        X_val = training_data["X_val"]
        y_val = training_data["y_val"]
        scaler = training_data["scaler"]
        label_scaler = training_data["label_scaler"]
        feature_names = training_data["feature_names"]
        num_features = training_data["num_features"]

        print("✅ Data prepared:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Validation samples: {len(X_val)}")
        print(f"   - Features: {num_features}")
        print(f"   - Sequence length: {request.sequence_length}")

        # Step 2: Initialize LSTM model
        print("\n🧠 Step 2: Initializing PyTorch LSTM model...")
        config = LSTMConfig(
            input_dim=num_features,
            hidden_dim=request.hidden_dim,
            num_layers=request.num_layers,
            dropout=request.dropout,
            sequence_length=request.sequence_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        model = HealthMetricLSTM(config)
        device = torch.device(config.device)
        model = model.to(device)

        print("✅ Model initialized:")
        print(f"   - Architecture: {request.architecture.value}")
        print(f"   - Hidden dim: {config.hidden_dim}")
        print(f"   - Layers: {config.num_layers}")
        print(f"   - Parameters: {model.count_parameters():,}")
        print(f"   - Device: {device}")

        # Step 3: Train the model
        print(f"\n🏋️ Step 3: Training model ({request.epochs} epochs)...")
        training_result = self._train_loop(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            device=device,
        )

        # Step 4: Evaluate on validation set
        print("\n📈 Step 4: Final evaluation...")
        eval_metrics = self._evaluate_model(
            model=model,
            X_val=X_val,
            y_val=y_val,
            label_scaler=label_scaler,
            device=device,
        )

        print("✅ Evaluation complete:")
        print(f"   - MAE: {eval_metrics['mae']:.4f}")
        print(f"   - RMSE: {eval_metrics['rmse']:.4f}")
        print(f"   - R² Score: {eval_metrics['r2_score']:.4f}")
        print(f"   - MAPE: {eval_metrics['mape']:.2f}%")

        # Step 5: Save model artifacts
        print("\n💾 Step 5: Saving model artifacts...")
        model_id = f"{request.user_id}_{request.metric.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_version = "v1.0.0"

        model_artifacts = self._save_model_artifacts(
            model_id=model_id,
            model=model,
            scaler=scaler,
            label_scaler=label_scaler,
            config=config,
            feature_names=feature_names,
            request=request,
            eval_metrics=eval_metrics,
        )

        print("✅ Model saved:")
        print(f"   - Model ID: {model_id}")
        print(f"   - Path: {model_artifacts['model_path']}")
        print(f"   - Size: {model_artifacts['model_size_mb']:.2f} MB")

        # Step 6: Quality assessment
        is_production_ready, quality_issues = self._assess_model_quality(eval_metrics)

        if is_production_ready:
            print("\n✅ Model is production-ready!")
        else:
            print("\n⚠️ Model has quality issues:")
            for issue in quality_issues:
                print(f"   - {issue}")

        # Step 7: Create response
        training_metrics = TrainingMetrics(
            train_loss=training_result["train_loss"],
            val_loss=training_result["val_loss"],
            best_val_loss=training_result["best_val_loss"],
            mae=eval_metrics["mae"],
            rmse=eval_metrics["rmse"],
            r2_score=eval_metrics["r2_score"],
            mape=eval_metrics["mape"],
            epochs_trained=training_result["epochs_trained"],
            early_stopped=training_result["early_stopped"],
            training_time_seconds=training_result["training_time_seconds"],
        )

        response = TrainModelResponse(
            user_id=request.user_id,
            metric=request.metric,
            architecture=request.architecture,
            model_id=model_id,
            model_version=model_version,
            trained_at=datetime.now(),
            training_metrics=training_metrics,
            total_samples=len(X_train) + len(X_val),
            sequence_length=request.sequence_length,
            num_features=num_features,
            model_path=str(model_artifacts["model_path"]),
            model_size_mb=model_artifacts["model_size_mb"],
            is_production_ready=is_production_ready,
            quality_issues=quality_issues,
        )

        print(f"\n{'='*70}")
        print("✅ Training complete!")
        print(f"{'='*70}\n")

        return response

    # ========================================================================
    # Training Loop
    # ========================================================================

    def _train_loop(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
    ) -> Dict:
        """
        PyTorch training loop with early stopping.

        Returns:
            Dictionary with training metrics
        """
        import time

        start_time = time.time()

        # Move data to device
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler (reduce on plateau)
        # Note: verbose parameter removed in PyTorch 2.x
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Early stopping parameters
        patience = 10
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        early_stopped = False

        # Training history
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i : i + batch_size]
                batch_y = y_train[i : i + batch_size]

                # Forward pass
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val)
                val_loss = criterion(val_predictions, y_val).item()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"   Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Best Val: {best_val_loss:.6f}"
                )

            # Early stopping
            if epochs_without_improvement >= patience:
                print(
                    f"\n⏹️ Early stopping triggered at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
                early_stopped = True
                break

        training_time = time.time() - start_time

        # Replace NaN and infinity values with large numbers for JSON compatibility
        import math

        def sanitize_float(value):
            """Replace NaN and infinity with large finite number."""
            if math.isnan(value) or math.isinf(value):
                return 999999.0
            return value

        train_loss_final = sanitize_float(train_losses[-1])
        val_loss_final = sanitize_float(val_losses[-1])
        best_val_loss_final = sanitize_float(best_val_loss)

        return {
            "train_loss": train_loss_final,
            "val_loss": val_loss_final,
            "best_val_loss": best_val_loss_final,
            "epochs_trained": epoch + 1,
            "early_stopped": early_stopped,
            "training_time_seconds": training_time,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    # ========================================================================
    # Model Evaluation
    # ========================================================================

    def _evaluate_model(
        self,
        model: nn.Module,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        label_scaler,
        device: torch.device,
    ) -> Dict:
        """
        Evaluate model on validation set with comprehensive metrics.

        Returns:
            Dictionary with MAE, RMSE, R², MAPE
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np

        model.eval()
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        with torch.no_grad():
            predictions = model(X_val)

        # Convert to numpy
        y_val_np = y_val.cpu().numpy().flatten()
        predictions_np = predictions.cpu().numpy().flatten()

        # Denormalize for real-world metrics
        y_val_real = label_scaler.inverse_transform(y_val_np.reshape(-1, 1)).flatten()
        predictions_real = label_scaler.inverse_transform(
            predictions_np.reshape(-1, 1)
        ).flatten()

        # Check for NaN values and handle them
        valid_mask = ~(np.isnan(y_val_real) | np.isnan(predictions_real))

        if not np.any(valid_mask):
            # All predictions/targets are NaN - model failed
            # Return very large numbers (JSON-compliant) to indicate failure
            return {
                "mae": 999999.0,
                "rmse": 999999.0,
                "r2_score": -999999.0,
                "mape": 999999.0,
            }

        y_val_real = y_val_real[valid_mask]
        predictions_real = predictions_real[valid_mask]

        # Calculate metrics on denormalized values
        mae = mean_absolute_error(y_val_real, predictions_real)
        mse = mean_squared_error(y_val_real, predictions_real)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_real, predictions_real)

        # MAPE (Mean Absolute Percentage Error) - avoid division by zero
        epsilon = 1e-10  # Small constant to avoid division by zero
        mape = (
            np.mean(np.abs((y_val_real - predictions_real) / (y_val_real + epsilon)))
            * 100
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "mape": mape,
        }

    # ========================================================================
    # Model Persistence
    # ========================================================================

    def _save_model_artifacts(
        self,
        model_id: str,
        model: nn.Module,
        scaler,
        label_scaler,
        config: LSTMConfig,
        feature_names: List[str],
        request: TrainModelRequest,
        eval_metrics: Dict,
    ) -> Dict:
        """
        Save model and all artifacts needed for inference.

        Saves:
        - PyTorch model state dict
        - Feature scaler
        - Label scaler
        - Model config
        - Feature names
        - Training metadata
        """
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        # Save PyTorch model
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save scalers
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        label_scaler_path = model_dir / "label_scaler.pkl"
        with open(label_scaler_path, "wb") as f:
            pickle.dump(label_scaler, f)

        # Save config
        config_path = model_dir / "config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

        # Save feature names
        features_path = model_dir / "feature_names.pkl"
        with open(features_path, "wb") as f:
            pickle.dump(feature_names, f)

        # Save metadata
        metadata = {
            "model_id": model_id,
            "user_id": request.user_id,
            "metric": request.metric.value,
            "architecture": request.architecture.value,
            "trained_at": datetime.now().isoformat(),
            "sequence_length": request.sequence_length,
            "num_features": len(feature_names),
            "lookback_days": request.lookback_days,
            "hyperparameters": {
                "hidden_dim": request.hidden_dim,
                "num_layers": request.num_layers,
                "dropout": request.dropout,
                "learning_rate": request.learning_rate,
                "batch_size": request.batch_size,
                "epochs": request.epochs,
            },
            "validation_metrics": {
                "mae": eval_metrics["mae"],
                "rmse": eval_metrics["rmse"],
                "r2_score": eval_metrics["r2_score"],
                "mape": eval_metrics["mape"],
            },
        }

        metadata_path = model_dir / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        # Calculate model size
        model_size_mb = sum(
            f.stat().st_size for f in model_dir.glob("**/*") if f.is_file()
        ) / (1024 * 1024)

        return {
            "model_path": model_path,
            "model_size_mb": model_size_mb,
        }

    # ========================================================================
    # Quality Assessment
    # ========================================================================

    def _assess_model_quality(self, eval_metrics: Dict) -> tuple[bool, List[str]]:
        """
        Assess if model is production-ready.

        Quality thresholds:
        - R² > 0.5 (explains >50% variance)
        - MAPE < 15% (predictions within 15% on average)
        - MAE reasonable for metric type

        Returns:
            (is_production_ready, quality_issues)
        """
        issues = []

        # R² threshold
        if eval_metrics["r2_score"] < 0.5:
            issues.append(
                f"Low R² score ({eval_metrics['r2_score']:.3f} < 0.5) - "
                "model explains <50% of variance"
            )

        # MAPE threshold
        if eval_metrics["mape"] > 15:
            issues.append(
                f"High prediction error (MAPE {eval_metrics['mape']:.1f}% > 15%) - "
                "predictions off by >15% on average"
            )

        # Negative R² is very bad
        if eval_metrics["r2_score"] < 0:
            issues.append("Negative R² - model performs worse than predicting the mean")

        is_production_ready = len(issues) == 0

        return is_production_ready, issues
