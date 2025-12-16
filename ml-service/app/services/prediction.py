"""
Prediction Service for Health Metric Forecasting

Handles:
1. Loading trained PyTorch models from disk
2. Preparing input data for prediction
3. Making predictions with confidence intervals
4. Denormalizing predictions to original scale
5. Natural language interpretation
6. Caching predictions in Redis
"""

import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sqlalchemy.ext.asyncio import AsyncSession

from app.redis_client import redis_client
from app.ml_models.lstm import HealthMetricLSTM
from app.schemas.predictions import (
    PredictRequest,
    PredictResponse,
    PredictionResult,
    PredictionMetric,
    ModelArchitecture,
)
from app.services.data_preparation import DataPreparationService


class PredictionService:
    """
    Service for making predictions using trained PyTorch LSTM models.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_prep_service = DataPreparationService(db)
        self.models_dir = Path("models")

        # Cache TTL for predictions (24 hours)
        self.cache_ttl = 86400

    # ========================================================================
    # Main Prediction Entry Point
    # ========================================================================

    async def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Make a prediction for a health metric using trained LSTM model.

        Steps:
        1. Check Redis cache for existing prediction
        2. Load trained model artifacts
        3. Prepare input features
        4. Make prediction
        5. Calculate confidence interval
        6. Denormalize to original scale
        7. Generate interpretation
        8. Cache result

        Args:
            request: Prediction request

        Returns:
            PredictResponse with prediction and interpretation
        """
        print(f"\n{'='*70}")
        print(f"🔮 Making prediction for {request.metric.value}")
        print(f"   User: {request.user_id}")
        print(f"   Target date: {request.target_date}")
        print(f"{'='*70}\n")

        # Step 1: Check cache
        cache_key = self._get_cache_key(
            request.user_id, request.metric, request.target_date
        )
        cached_result = await self._get_cached_prediction(cache_key)

        if cached_result and not request.force_recompute_features:
            print("✅ Using cached prediction")
            return cached_result

        # Step 2: Find and load model
        print("🧠 Step 1: Loading trained model...")
        model_id = self._find_latest_model(
            request.user_id, request.metric, request.architecture
        )

        if not model_id:
            raise ValueError(
                f"No trained model found for user {request.user_id} "
                f"and metric {request.metric.value}"
            )

        model_artifacts = self._load_model_artifacts(model_id)
        print(f"✅ Model loaded: {model_id}")

        # Step 3: Prepare input features
        print("\n📊 Step 2: Preparing input features...")
        X_input = await self.data_prep_service.prepare_prediction_input(
            user_id=request.user_id,
            target_date=request.target_date,
            sequence_length=model_artifacts["config"].sequence_length,
            scaler=model_artifacts["scaler"],
            feature_names=model_artifacts["feature_names"],
        )

        print("✅ Input prepared:")
        print(f"   - Shape: {X_input.shape}")
        print(f"   - Features: {X_input.shape[2]}")
        print(f"   - Sequence length: {X_input.shape[1]}")

        # Step 4: Make prediction
        print("\n🔮 Step 3: Making prediction...")
        model = model_artifacts["model"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        X_input = X_input.to(device)

        model.eval()
        with torch.no_grad():
            normalized_prediction = model(X_input).item()

        # Step 5: Denormalize prediction
        label_scaler = model_artifacts["label_scaler"]
        predicted_value = self.data_prep_service.denormalize_prediction(
            normalized_prediction, label_scaler
        )

        print(f"✅ Prediction: {predicted_value:.2f}")

        # Step 6: Calculate confidence interval
        print("\n📈 Step 4: Calculating confidence interval...")
        confidence_interval = self._calculate_confidence_interval(
            predicted_value, model_artifacts["metadata"]
        )

        print(
            f"✅ 95% CI: [{confidence_interval['lower']:.2f}, {confidence_interval['upper']:.2f}]"
        )

        # Step 7: Get historical context
        print("\n📊 Step 5: Getting historical context...")
        historical_stats = await self._get_historical_stats(
            request.user_id, request.metric
        )

        # Step 8: Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            predicted_value, historical_stats, model_artifacts["metadata"]
        )

        print(f"✅ Confidence score: {confidence_score:.2%}")

        # Step 9: Generate interpretation
        print("\n💬 Step 6: Generating interpretation...")
        interpretation = self._generate_interpretation(
            metric=request.metric,
            predicted_value=predicted_value,
            historical_average=historical_stats["avg"],
            confidence_score=confidence_score,
        )

        recommendation = self._generate_recommendation(
            metric=request.metric,
            predicted_value=predicted_value,
            historical_average=historical_stats["avg"],
        )

        # Step 10: Create response
        prediction_result = PredictionResult(
            metric=request.metric,
            target_date=request.target_date,
            predicted_at=datetime.now(),
            predicted_value=predicted_value,
            confidence_interval_lower=confidence_interval["lower"],
            confidence_interval_upper=confidence_interval["upper"],
            confidence_score=confidence_score,
            historical_average=historical_stats["avg"],
            deviation_from_average=predicted_value - historical_stats["avg"],
            percentile=self._calculate_percentile(
                predicted_value, historical_stats["values"]
            ),
            model_id=model_id,
            model_version=(
                model_artifacts["metadata"]["model_version"]
                if "model_version" in model_artifacts["metadata"]
                else "v1.0.0"
            ),
            architecture=request.architecture or ModelArchitecture.LSTM,
        )

        response = PredictResponse(
            user_id=request.user_id,
            prediction=prediction_result,
            features_used=len(model_artifacts["feature_names"]),
            sequence_length=model_artifacts["config"].sequence_length,
            data_quality_score=0.85,  # TODO: Calculate based on missing data
            interpretation=interpretation,
            recommendation=recommendation,
            cached=False,
        )

        # Step 11: Cache the response
        await self._cache_prediction(cache_key, response)

        print(f"\n{'='*70}")
        print("✅ Prediction complete!")
        print(f"{'='*70}\n")

        return response

    # ========================================================================
    # Model Loading
    # ========================================================================

    def _find_latest_model(
        self,
        user_id: str,
        metric: PredictionMetric,
        architecture: Optional[ModelArchitecture] = None,
    ) -> Optional[str]:
        """
        Find the latest trained model for user and metric.

        Returns:
            Model ID (directory name) or None if not found
        """
        if not self.models_dir.exists():
            return None

        # Find all models for this user and metric
        pattern = f"{user_id}_{metric.value}_*"
        matching_models = list(self.models_dir.glob(pattern))

        if not matching_models:
            return None

        # Sort by timestamp (newest first)
        matching_models.sort(key=lambda p: p.name, reverse=True)

        # Return the newest model
        return matching_models[0].name

    def _load_model_artifacts(self, model_id: str) -> Dict:
        """
        Load all model artifacts from disk.

        Returns:
            Dictionary with model, scalers, config, feature_names, metadata
        """
        model_dir = self.models_dir / model_id

        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        # Load config
        config_path = model_dir / "config.pkl"
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        # Load PyTorch model
        model = HealthMetricLSTM(config)
        model_path = model_dir / "model.pt"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # Load scalers
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        label_scaler_path = model_dir / "label_scaler.pkl"
        with open(label_scaler_path, "rb") as f:
            label_scaler = pickle.load(f)

        # Load feature names
        features_path = model_dir / "feature_names.pkl"
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)

        # Load metadata
        metadata_path = model_dir / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        return {
            "model": model,
            "scaler": scaler,
            "label_scaler": label_scaler,
            "config": config,
            "feature_names": feature_names,
            "metadata": metadata,
        }

    # ========================================================================
    # Confidence Interval Calculation
    # ========================================================================

    def _calculate_confidence_interval(
        self, predicted_value: float, metadata: Dict
    ) -> Dict[str, float]:
        """
        Calculate 95% confidence interval for prediction.

        Uses model's validation MAE to estimate uncertainty.

        Returns:
            {"lower": float, "upper": float}
        """
        # Use 1.96 * MAE as confidence interval width (approximation for 95% CI)
        # Get MAE from validation_metrics (saved during training)
        validation_metrics = metadata.get("validation_metrics", {})
        mae = validation_metrics.get(
            "mae", predicted_value * 0.1
        )  # Fallback: 10% of value

        margin = 1.96 * mae

        return {
            "lower": max(0, predicted_value - margin),  # Don't go negative
            "upper": predicted_value + margin,
        }

    def _calculate_confidence_score(
        self,
        predicted_value: float,
        historical_stats: Dict,
        metadata: Dict,
    ) -> float:
        """
        Calculate confidence score (0-1) for prediction.

        Factors:
        - Model R² score (higher = more confident)
        - Prediction within historical range (in-distribution)
        - Data quality

        Returns:
            Confidence score between 0 and 1
        """
        # Factor 1: Model R² (0.5 to 1.0 maps to 0.5 to 1.0 confidence)
        validation_metrics = metadata.get("validation_metrics", {})
        r2_score = validation_metrics.get("r2_score", 0.5)
        r2_confidence = max(0, min(1, r2_score))

        # Factor 2: In-distribution check
        min_val = historical_stats.get("min", 0)
        max_val = historical_stats.get("max", 100)

        if min_val <= predicted_value <= max_val:
            distribution_confidence = 1.0
        else:
            # Penalize extrapolation
            distribution_confidence = 0.7

        # Factor 3: Data quality (TODO: calculate based on missing features)
        data_quality_confidence = 0.85

        # Weighted average
        confidence = (
            0.5 * r2_confidence
            + 0.3 * distribution_confidence
            + 0.2 * data_quality_confidence
        )

        return confidence

    # ========================================================================
    # Historical Statistics
    # ========================================================================

    async def _get_historical_stats(
        self, user_id: str, metric: PredictionMetric
    ) -> Dict:
        """
        Get historical statistics for the metric (30-day window).

        Returns:
            Dictionary with avg, min, max, std, values
        """
        from app.models.health_metric import HealthMetric
        from sqlalchemy import select, and_
        from datetime import timedelta

        # Fetch last 30 days of data
        start_date = date.today() - timedelta(days=30)

        result = await self.db.execute(
            select(HealthMetric).where(
                and_(
                    HealthMetric.user_id == user_id,
                    HealthMetric.metric_type == metric.value,
                    HealthMetric.recorded_at >= start_date,
                )
            )
        )
        metrics = result.scalars().all()

        if not metrics:
            # Return sensible defaults
            return {
                "avg": 60.0,
                "min": 40.0,
                "max": 100.0,
                "std": 10.0,
                "values": [60.0],
            }

        values = [m.value for m in metrics]

        return {
            "avg": np.mean(values),
            "min": np.min(values),
            "max": np.max(values),
            "std": np.std(values),
            "values": values,
        }

    def _calculate_percentile(
        self, predicted_value: float, historical_values: List[float]
    ) -> float:
        """
        Calculate what percentile the prediction is in user's history.

        Returns:
            Percentile (0-100)
        """
        if not historical_values:
            return 50.0

        percentile = (
            np.sum(np.array(historical_values) <= predicted_value)
            / len(historical_values)
            * 100
        )

        return float(percentile)

    # ========================================================================
    # Interpretation and Recommendations
    # ========================================================================

    def _generate_interpretation(
        self,
        metric: PredictionMetric,
        predicted_value: float,
        historical_average: float,
        confidence_score: float,
    ) -> str:
        """
        Generate natural language interpretation of prediction.
        """
        metric_name = self._get_metric_display_name(metric)
        deviation = predicted_value - historical_average
        deviation_pct = (deviation / historical_average) * 100

        confidence_text = (
            "high confidence"
            if confidence_score > 0.8
            else "moderate confidence"
            if confidence_score > 0.6
            else "low confidence"
        )

        if abs(deviation_pct) < 5:
            trend = "similar to"
        elif deviation > 0:
            trend = f"{abs(deviation_pct):.1f}% higher than"
        else:
            trend = f"{abs(deviation_pct):.1f}% lower than"

        return (
            f"Your predicted {metric_name} is {predicted_value:.1f}, "
            f"which is {trend} your 30-day average of {historical_average:.1f}. "
            f"This prediction has {confidence_text}."
        )

    def _generate_recommendation(
        self,
        metric: PredictionMetric,
        predicted_value: float,
        historical_average: float,
    ) -> Optional[str]:
        """
        Generate actionable recommendation based on prediction.
        """
        deviation = predicted_value - historical_average

        if metric == PredictionMetric.RHR:
            if deviation > 5:
                return (
                    "Your resting heart rate may be elevated tomorrow. "
                    "Consider lighter training, prioritizing recovery, and ensuring "
                    "adequate hydration."
                )
            elif deviation < -5:
                return (
                    "Your resting heart rate is predicted to be lower than average, "
                    "indicating good recovery. This may be a good day for higher "
                    "intensity training."
                )

        elif metric in [PredictionMetric.HRV_SDNN, PredictionMetric.HRV_RMSSD]:
            if deviation < -10:
                return (
                    "Your HRV may be lower tomorrow, suggesting reduced recovery. "
                    "Prioritize sleep, reduce stress, and consider active recovery "
                    "instead of intense training."
                )
            elif deviation > 10:
                return (
                    "Your HRV is predicted to be higher, indicating strong recovery. "
                    "This is a good time for challenging workouts."
                )

        return None

    def _get_metric_display_name(self, metric: PredictionMetric) -> str:
        """Get human-readable metric name."""
        metric_names = {
            PredictionMetric.RHR: "Resting Heart Rate",
            PredictionMetric.HRV_SDNN: "Heart Rate Variability (SDNN)",
            PredictionMetric.HRV_RMSSD: "Heart Rate Variability (RMSSD)",
            PredictionMetric.SLEEP_DURATION: "Sleep Duration",
            PredictionMetric.RECOVERY_SCORE: "Recovery Score",
        }
        return metric_names.get(metric, metric.value)

    # ========================================================================
    # Redis Caching
    # ========================================================================

    def _get_cache_key(
        self, user_id: str, metric: PredictionMetric, target_date: date
    ) -> str:
        """Generate Redis cache key for prediction."""
        return f"prediction:{user_id}:{metric.value}:{target_date.isoformat()}"

    async def _get_cached_prediction(self, cache_key: str) -> Optional[PredictResponse]:
        """Retrieve cached prediction from Redis."""
        cached = await redis_client.get(cache_key)

        if cached:
            # redis_client.get() already deserializes JSON to dict
            # Set cached=True to indicate this came from cache
            cached["cached"] = True
            return PredictResponse(**cached)

        return None

    async def _cache_prediction(
        self, cache_key: str, response: PredictResponse
    ) -> None:
        """Cache prediction in Redis."""
        # NOTE: Do NOT set response.cached = True here!
        # The cached flag should remain False for fresh predictions.
        # It's only True when retrieving from cache.

        # Convert to dict (redis_client.set will handle JSON encoding)
        data = response.model_dump()

        await redis_client.set(cache_key, data, ttl=self.cache_ttl)
