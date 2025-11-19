"""
Prediction API Routes

Endpoints for ML model training and prediction:
- POST /train - Train a new LSTM model
- POST /predict - Make a single prediction
- POST /batch-predict - Predict multiple metrics
- GET /models/{user_id} - List user's trained models
- GET /models/{model_id}/performance - Get model performance history
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.predictions import (
    TrainModelRequest,
    TrainModelResponse,
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    ListModelsResponse,
    ModelInfo,
    PredictionMetric,
)
from app.services.model_training import ModelTrainingService
from app.services.prediction import PredictionService

router = APIRouter()


# ============================================================================
# Model Training
# ============================================================================


@router.post("/train", response_model=TrainModelResponse)
async def train_model(
    request: TrainModelRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Train a PyTorch LSTM model for health metric prediction.

    **Process:**
    1. Fetch historical data (nutrition, activity, health metrics)
    2. Engineer 51 ML features
    3. Create time-series sequences (sliding windows)
    4. Train LSTM neural network with early stopping
    5. Evaluate on validation set
    6. Save model artifacts

    **Quality checks:**
    - R² > 0.5 (explains >50% variance)
    - MAPE < 15% (predictions within 15% on average)

    **Returns:**
    - Model ID and version
    - Training metrics (MAE, RMSE, R², MAPE)
    - Production readiness status
    - Model file path and size

    **Example:**
    ```json
    {
        "user_id": "user_123",
        "metric": "RESTING_HEART_RATE",
        "architecture": "lstm",
        "lookback_days": 90,
        "sequence_length": 30,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2
    }
    ```
    """
    try:
        training_service = ModelTrainingService(db)
        result = await training_service.train_model(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# ============================================================================
# Single Prediction
# ============================================================================


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Make a prediction for a health metric using trained LSTM model.

    **Process:**
    1. Load trained model from disk
    2. Fetch last 30 days of features
    3. Prepare input sequence
    4. Run LSTM inference
    5. Denormalize prediction
    6. Calculate confidence interval
    7. Generate interpretation

    **Returns:**
    - Predicted value with confidence interval
    - Confidence score (0-1)
    - Historical context (30-day average, percentile)
    - Natural language interpretation
    - Actionable recommendations

    **Caching:**
    - Predictions cached in Redis for 24 hours
    - Use `force_recompute_features: true` to bypass cache

    **Example:**
    ```json
    {
        "user_id": "user_123",
        "metric": "RESTING_HEART_RATE",
        "target_date": "2025-01-16"
    }
    ```

    **Example Response:**
    ```json
    {
        "user_id": "user_123",
        "prediction": {
            "predicted_value": 62.5,
            "confidence_interval_lower": 58.3,
            "confidence_interval_upper": 66.7,
            "confidence_score": 0.87,
            "historical_average": 60.2,
            "deviation_from_average": 2.3,
            "percentile": 65.0
        },
        "interpretation": "Your predicted Resting Heart Rate is 62.5, which is 3.8% higher than your 30-day average of 60.2. This prediction has high confidence.",
        "recommendation": "Your resting heart rate may be elevated tomorrow. Consider lighter training and prioritizing recovery.",
        "cached": false
    }
    ```
    """
    try:
        prediction_service = PredictionService(db)
        result = await prediction_service.predict(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Batch Prediction
# ============================================================================


@router.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(
    request: BatchPredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Predict multiple health metrics at once.

    **Use case:** Daily morning forecast (RHR + HRV + Sleep)

    **Process:**
    - Runs prediction for each metric in parallel
    - Returns all successful predictions
    - Lists any failed metrics

    **Example:**
    ```json
    {
        "user_id": "user_123",
        "metrics": [
            "RESTING_HEART_RATE",
            "HEART_RATE_VARIABILITY_SDNN",
            "SLEEP_DURATION"
        ],
        "target_date": "2025-01-16"
    }
    ```
    """
    try:
        prediction_service = PredictionService(db)

        predictions = {}
        failed_metrics = []

        # Predict each metric
        for metric in request.metrics:
            try:
                predict_request = PredictRequest(
                    user_id=request.user_id,
                    metric=metric,
                    target_date=request.target_date,
                    model_version=request.model_version,
                    force_recompute_features=request.force_recompute_features,
                )

                result = await prediction_service.predict(predict_request)
                predictions[metric.value] = result.prediction

            except Exception as e:
                failed_metrics.append(metric.value)
                print(f"⚠️ Failed to predict {metric.value}: {e}")

        # Calculate overall data quality (average of individual predictions)
        if predictions:
            overall_quality = sum(
                0.85 for _ in predictions
            ) / len(predictions)  # TODO: Use actual quality scores
        else:
            overall_quality = 0.0

        response = BatchPredictResponse(
            user_id=request.user_id,
            target_date=request.target_date,
            predicted_at=list(predictions.values())[0].predicted_at
            if predictions
            else None,
            predictions=predictions,
            overall_data_quality=overall_quality,
            all_predictions_successful=len(failed_metrics) == 0,
            failed_metrics=failed_metrics,
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================================================
# Model Management
# ============================================================================


@router.get("/models/{user_id}", response_model=ListModelsResponse)
async def list_models(
    user_id: str,
    metric: Optional[PredictionMetric] = Query(None, description="Filter by metric"),
    db: AsyncSession = Depends(get_db),
):
    """
    List all trained models for a user.

    **Filters:**
    - `metric`: Filter by health metric (optional)

    **Returns:**
    - List of models with metadata
    - Training metrics for each model
    - Active/production-ready status

    **Example:**
    ```
    GET /api/predictions/models/user_123?metric=RESTING_HEART_RATE
    ```
    """
    try:
        from pathlib import Path
        import pickle

        models_dir = Path("models")

        if not models_dir.exists():
            return ListModelsResponse(
                user_id=user_id,
                models=[],
                total_models=0,
            )

        # Find all models for this user
        pattern = f"{user_id}_*"
        if metric:
            pattern = f"{user_id}_{metric.value}_*"

        matching_models = list(models_dir.glob(pattern))

        model_infos = []
        for model_dir in matching_models:
            try:
                # Load metadata
                metadata_path = model_dir / "metadata.pkl"
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)

                # Load training metrics from metadata
                from app.schemas.predictions import TrainingMetrics

                validation_metrics = metadata.get("validation_metrics", {})
                training_metrics = TrainingMetrics(
                    train_loss=metadata.get("train_loss", 0.0),
                    val_loss=metadata.get("val_loss", 0.0),
                    best_val_loss=metadata.get("best_val_loss", 0.0),
                    mae=validation_metrics.get("mae", 0.0),
                    rmse=validation_metrics.get("rmse", 0.0),
                    r2_score=validation_metrics.get("r2_score", 0.0),
                    mape=validation_metrics.get("mape", 0.0),
                    epochs_trained=metadata.get("epochs_trained", 0),
                    early_stopped=metadata.get("early_stopped", False),
                    training_time_seconds=metadata.get("training_time_seconds", 0.0),
                )

                # Calculate model size
                model_size_mb = sum(
                    f.stat().st_size for f in model_dir.glob("**/*") if f.is_file()
                ) / (1024 * 1024)

                model_info = ModelInfo(
                    model_id=metadata["model_id"],
                    user_id=metadata["user_id"],
                    metric=PredictionMetric(metadata["metric"]),
                    architecture=metadata["architecture"],
                    version="v1.0.0",  # TODO: Version management
                    trained_at=metadata["trained_at"],
                    training_metrics=training_metrics,
                    sequence_length=metadata["sequence_length"],
                    num_features=metadata["num_features"],
                    model_size_mb=model_size_mb,
                    is_active=True,  # TODO: Track active models
                    is_production_ready=metadata.get("is_production_ready", True),
                )

                model_infos.append(model_info)

            except Exception as e:
                print(f"⚠️ Error loading model {model_dir.name}: {e}")
                continue

        # Sort by training date (newest first)
        model_infos.sort(key=lambda m: m.trained_at, reverse=True)

        return ListModelsResponse(
            user_id=user_id,
            models=model_infos,
            total_models=len(model_infos),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list models: {str(e)}"
        )


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a trained model.

    **Warning:** This permanently deletes the model and all artifacts.
    Predictions using this model will fail.

    **Example:**
    ```
    DELETE /api/predictions/models/user_123_RESTING_HEART_RATE_20250115_103045
    ```
    """
    try:
        from pathlib import Path
        import shutil

        models_dir = Path("models")
        model_dir = models_dir / model_id

        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        # Delete the model directory
        shutil.rmtree(model_dir)

        return {
            "message": f"Model {model_id} deleted successfully",
            "model_id": model_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete model: {str(e)}"
        )


# ============================================================================
# Convenience Endpoints
# ============================================================================


@router.get("/predict/{user_id}/{metric}/{target_date}", response_model=PredictResponse)
async def predict_get(
    user_id: str,
    metric: PredictionMetric,
    target_date: date,
    db: AsyncSession = Depends(get_db),
):
    """
    Convenience GET endpoint for predictions.

    **Example:**
    ```
    GET /api/predictions/predict/user_123/RESTING_HEART_RATE/2025-01-16
    ```

    Equivalent to:
    ```json
    POST /api/predictions/predict
    {
        "user_id": "user_123",
        "metric": "RESTING_HEART_RATE",
        "target_date": "2025-01-16"
    }
    ```
    """
    request = PredictRequest(
        user_id=user_id,
        metric=metric,
        target_date=target_date,
    )

    return await predict(request, db)
