"""
Correlation Engine Service

Analyzes statistical correlations between nutrition/activity features and health metrics.
Implements Pearson, Spearman, Kendall correlations with lag analysis for time-delayed effects.
"""

from datetime import date, datetime, timedelta, UTC
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession
from statsmodels.tsa.stattools import grangercausalitytests

from app.schemas.correlations import (
    CorrelationMethod,
    HealthMetricTarget,
    CorrelationRequest,
    CorrelationResult,
    CorrelationResponse,
    LagAnalysisRequest,
    LagAnalysisResult,
    LagAnalysisResponse,
    interpret_correlation_strength,
    interpret_correlation_direction,
)
from app.services.feature_engineering import FeatureEngineeringService
from app.schemas.features import FeatureCategory


class CorrelationEngineService:
    """
    Service for analyzing correlations between features and health metrics.

    Supports multiple correlation methods and lag analysis for detecting
    time-delayed effects (e.g., "today's protein affects tomorrow's HRV").
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.feature_service = FeatureEngineeringService(db)

    # ========================================================================
    # Main Entry Point - Correlation Analysis
    # ========================================================================

    async def analyze_correlations(
        self,
        request: CorrelationRequest,
    ) -> CorrelationResponse:
        """
        Analyze correlations between all features and a target health metric.

        This is the primary analysis function that:
        1. Fetches historical features for lookback period
        2. Fetches target health metric data
        3. Computes correlations using requested methods
        4. Filters by significance and minimum correlation
        5. Returns top K results

        Args:
            request: CorrelationRequest with user_id, target_metric, methods, etc.

        Returns:
            CorrelationResponse with sorted correlation results
        """
        user_id = request.user_id
        lookback_days = request.lookback_days
        target_metric = request.target_metric

        # Step 1: Build feature matrix (one row per day)
        feature_matrix, dates = await self._build_feature_matrix(
            user_id, lookback_days
        )

        if feature_matrix.empty:
            return self._empty_correlation_response(request)

        # Step 2: Fetch target metric values
        target_values = await self._fetch_target_metric(
            user_id, target_metric, dates
        )

        if target_values.empty:
            return self._empty_correlation_response(request)

        # Step 3: Align data (remove days with missing target values)
        aligned_features, aligned_target = self._align_data(
            feature_matrix, target_values, dates
        )

        if len(aligned_features) < 7:
            return CorrelationResponse(
                user_id=user_id,
                target_metric=target_metric,
                analyzed_at=datetime.now(UTC),
                lookback_days=lookback_days,
                correlations=[],
                total_features_analyzed=0,
                significant_correlations=0,
                data_quality_score=0.0,
                missing_days=lookback_days - len(aligned_features),
                warning="Insufficient data: need at least 7 days with both features and target metric",
            )

        # Step 4: Compute correlations for each feature
        all_correlations = []

        for feature_name in aligned_features.columns:
            feature_values = aligned_features[feature_name].dropna()
            target_matched = aligned_target[feature_values.index]

            # Skip if too few valid pairs
            if len(feature_values) < 7:
                continue

            # Compute correlations for each requested method
            for method in request.methods:
                result = self._compute_correlation(
                    feature_name,
                    feature_values.values,
                    target_matched.values,
                    method,
                    request.significance_threshold,
                )

                if result:
                    all_correlations.append(result)

        # Step 5: Filter and sort results
        # Filter by significance and minimum correlation
        significant = [
            r for r in all_correlations
            if r.is_significant and abs(r.correlation) >= request.min_correlation
        ]

        # Sort by absolute correlation (strongest first)
        significant.sort(key=lambda x: abs(x.correlation), reverse=True)

        # Take top K
        top_correlations = significant[:request.top_k]

        # Step 6: Calculate summary statistics
        strongest_positive = None
        strongest_negative = None

        for r in top_correlations:
            if r.direction == "positive" and (
                not strongest_positive or r.correlation > strongest_positive.correlation
            ):
                strongest_positive = r
            elif r.direction == "negative" and (
                not strongest_negative or r.correlation < strongest_negative.correlation
            ):
                strongest_negative = r

        # Data quality
        missing_days = lookback_days - len(aligned_features)
        # Cap quality_score at 1.0 (can exceed 1.0 if more data than lookback_days)
        quality_score = min(1.0, len(aligned_features) / lookback_days)

        return CorrelationResponse(
            user_id=user_id,
            target_metric=target_metric,
            analyzed_at=datetime.now(UTC),
            lookback_days=lookback_days,
            correlations=top_correlations,
            total_features_analyzed=len(aligned_features.columns),
            significant_correlations=len(significant),
            strongest_positive=strongest_positive,
            strongest_negative=strongest_negative,
            data_quality_score=quality_score,
            missing_days=missing_days,
            warning=None if quality_score > 0.7 else "Data quality is low (< 70% complete)",
        )

    def _compute_correlation(
        self,
        feature_name: str,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        method: CorrelationMethod,
        significance_threshold: float,
    ) -> Optional[CorrelationResult]:
        """
        Compute a single correlation between feature and target.

        Args:
            feature_name: Name of the feature
            feature_values: Array of feature values
            target_values: Array of target metric values
            method: Correlation method (pearson, spearman, etc.)
            significance_threshold: P-value threshold

        Returns:
            CorrelationResult or None if computation fails
        """
        try:
            correlation = 0.0
            p_value = 1.0

            if method == CorrelationMethod.PEARSON:
                correlation, p_value = stats.pearsonr(feature_values, target_values)

            elif method == CorrelationMethod.SPEARMAN:
                correlation, p_value = stats.spearmanr(feature_values, target_values)

            elif method == CorrelationMethod.KENDALL:
                correlation, p_value = stats.kendalltau(feature_values, target_values)

            elif method == CorrelationMethod.GRANGER:
                # Granger causality is different - test if feature "causes" target
                # Requires at least 15 samples for reliable results
                if len(feature_values) < 15:
                    return None

                # Prepare data for Granger test (2D array)
                data = np.column_stack([target_values, feature_values])

                # Run Granger test with max lag of 3 days
                max_lag = min(3, len(feature_values) // 5)
                if max_lag < 1:
                    return None

                granger_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)

                # Extract minimum p-value across all lags
                p_values = [
                    granger_result[lag][0]["ssr_ftest"][1]
                    for lag in range(1, max_lag + 1)
                ]
                p_value = min(p_values)

                # Granger doesn't give correlation coefficient, use F-statistic as proxy
                f_stats = [
                    granger_result[lag][0]["ssr_ftest"][0]
                    for lag in range(1, max_lag + 1)
                ]
                max_f = max(f_stats)
                # Normalize F-statistic to [-1, 1] range (very rough approximation)
                correlation = min(1.0, max_f / 10)  # Rough heuristic

            # Check if correlation is NaN (occurs with constant arrays or insufficient variance)
            if np.isnan(correlation) or np.isnan(p_value):
                return None

            # Determine feature category from name
            feature_category = self._infer_feature_category(feature_name)

            # Calculate explained variance (R²) for Pearson
            explained_variance = None
            if method == CorrelationMethod.PEARSON:
                explained_variance = correlation ** 2

            return CorrelationResult(
                feature_name=feature_name,
                feature_category=feature_category,
                correlation=float(correlation),
                p_value=float(p_value),
                sample_size=len(feature_values),
                method=method,
                is_significant=p_value < significance_threshold,
                strength=interpret_correlation_strength(correlation),
                direction=interpret_correlation_direction(correlation),
                explained_variance=explained_variance,
            )

        except Exception as e:
            # Log error but don't fail entire analysis
            print(f"Error computing correlation for {feature_name} with {method}: {e}")
            return None

    def _infer_feature_category(self, feature_name: str) -> str:
        """Infer feature category from feature name."""
        name_lower = feature_name.lower()

        if any(x in name_lower for x in ["protein", "carbs", "fat", "calories", "fiber", "meal", "eating"]):
            return "nutrition"
        elif any(x in name_lower for x in ["steps", "active", "workout", "recovery", "cardio", "strength"]):
            return "activity"
        elif any(x in name_lower for x in ["rhr", "hrv", "sleep", "recovery_score"]):
            return "health"
        elif any(x in name_lower for x in ["day_of_week", "weekend", "week", "month", "cycle"]):
            return "temporal"
        elif any(x in name_lower for x in ["per_kg", "per_minute", "per_workout", "to_recovery", "to_intensity"]):
            return "interaction"
        else:
            return "other"

    # ========================================================================
    # Lag Analysis (Time-Delayed Effects)
    # ========================================================================

    async def analyze_lag(
        self,
        request: LagAnalysisRequest,
    ) -> LagAnalysisResponse:
        """
        Analyze lagged correlations to detect time-delayed effects.

        Example: "Does today's protein intake affect tomorrow's HRV?"

        Tests correlation at multiple time lags (0h, 6h, 12h, 24h, 48h, etc.)
        to find the optimal lag where correlation is strongest.

        Args:
            request: LagAnalysisRequest with user_id, target_metric, feature_name, etc.

        Returns:
            LagAnalysisResponse with correlations at each lag
        """
        user_id = request.user_id
        lookback_days = request.lookback_days
        target_metric = request.target_metric
        feature_name = request.feature_name

        # Build feature matrix and fetch target
        feature_matrix, dates = await self._build_feature_matrix(
            user_id, lookback_days
        )

        if feature_matrix.empty or feature_name not in feature_matrix.columns:
            return self._empty_lag_response(request)

        target_values = await self._fetch_target_metric(
            user_id, target_metric, dates
        )

        if target_values.empty:
            return self._empty_lag_response(request)

        # Align data
        aligned_features, aligned_target = self._align_data(
            feature_matrix, target_values, dates
        )

        if len(aligned_features) < 14:
            return LagAnalysisResponse(
                user_id=user_id,
                target_metric=target_metric,
                feature_name=feature_name,
                analyzed_at=datetime.now(UTC),
                lag_results=[],
                optimal_lag_hours=None,
                optimal_correlation=None,
                immediate_effect=False,
                delayed_effect=False,
                effect_duration_hours=None,
                interpretation="Insufficient data for lag analysis (need at least 14 days)",
            )

        # Extract feature and target arrays
        feature_values = aligned_features[feature_name].values
        target_values_array = aligned_target.values

        # Compute correlations at different lags
        lag_results = []
        max_lag_hours = request.max_lag_hours
        lag_step = request.lag_step_hours

        for lag_hours in range(0, max_lag_hours + 1, lag_step):
            lag_days = lag_hours / 24

            # Shift feature backward by lag_days
            # (feature at day X correlates with target at day X + lag_days)
            shifted_feature, shifted_target = self._apply_lag(
                feature_values, target_values_array, lag_days
            )

            if len(shifted_feature) < 7:
                continue

            # Compute correlation at this lag
            try:
                if request.method == CorrelationMethod.PEARSON:
                    correlation, p_value = stats.pearsonr(shifted_feature, shifted_target)
                elif request.method == CorrelationMethod.SPEARMAN:
                    correlation, p_value = stats.spearmanr(shifted_feature, shifted_target)
                else:
                    correlation, p_value = stats.kendalltau(shifted_feature, shifted_target)

                lag_results.append(
                    LagAnalysisResult(
                        lag_hours=lag_hours,
                        correlation=float(correlation),
                        p_value=float(p_value),
                        is_significant=p_value < 0.05,
                    )
                )

            except Exception as e:
                print(f"Error computing correlation at lag {lag_hours}h: {e}")
                continue

        if not lag_results:
            return self._empty_lag_response(request)

        # Find optimal lag (strongest correlation)
        lag_results.sort(key=lambda x: x.lag_hours)
        optimal = max(lag_results, key=lambda x: abs(x.correlation))

        optimal_lag_hours = optimal.lag_hours
        optimal_correlation = optimal.correlation

        # Determine immediate vs delayed effect
        immediate_effect = lag_results[0].is_significant and abs(lag_results[0].correlation) >= 0.3
        delayed_effect = optimal_lag_hours > 0

        # Calculate effect duration (consecutive significant lags)
        effect_duration = self._calculate_effect_duration(lag_results, lag_step)

        # Generate interpretation
        interpretation = self._generate_lag_interpretation(
            feature_name,
            target_metric,
            optimal_lag_hours,
            optimal_correlation,
            immediate_effect,
            delayed_effect,
            effect_duration,
        )

        return LagAnalysisResponse(
            user_id=user_id,
            target_metric=target_metric,
            feature_name=feature_name,
            analyzed_at=datetime.now(UTC),
            lag_results=lag_results,
            optimal_lag_hours=optimal_lag_hours,
            optimal_correlation=optimal_correlation,
            immediate_effect=immediate_effect,
            delayed_effect=delayed_effect,
            effect_duration_hours=effect_duration,
            interpretation=interpretation,
        )

    def _apply_lag(
        self, feature: np.ndarray, target: np.ndarray, lag_days: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply lag to feature array.

        Shift feature backward by lag_days so that feature[i] aligns with target[i + lag_days].
        Remove elements that don't have matching pairs after shifting.

        Args:
            feature: Feature values
            target: Target values
            lag_days: Number of days to lag

        Returns:
            (shifted_feature, shifted_target) with aligned pairs
        """
        lag_indices = int(lag_days)

        if lag_indices == 0:
            return feature, target

        # Shift feature backward (remove first lag_indices elements)
        shifted_feature = feature[:-lag_indices] if lag_indices > 0 else feature
        # Shift target forward (remove last lag_indices elements)
        shifted_target = target[lag_indices:] if lag_indices > 0 else target

        # Ensure same length
        min_length = min(len(shifted_feature), len(shifted_target))
        return shifted_feature[:min_length], shifted_target[:min_length]

    def _calculate_effect_duration(
        self, lag_results: List[LagAnalysisResult], lag_step: int
    ) -> Optional[int]:
        """
        Calculate duration of significant correlation effect.

        Returns: Number of hours where correlation remains significant
        """
        if not lag_results:
            return None

        # Find consecutive significant lags starting from 0
        duration = 0
        for result in lag_results:
            if result.is_significant:
                duration = result.lag_hours + lag_step
            else:
                break  # Stop at first non-significant lag

        return duration if duration > 0 else None

    def _generate_lag_interpretation(
        self,
        feature_name: str,
        target_metric: HealthMetricTarget,
        optimal_lag: int,
        correlation: float,
        immediate: bool,
        delayed: bool,
        duration: Optional[int],
    ) -> str:
        """Generate natural language interpretation of lag analysis."""
        metric_name = target_metric.value.replace("_", " ").title()
        direction = "increases" if correlation > 0 else "decreases"
        strength = interpret_correlation_strength(correlation)

        if immediate and not delayed:
            return (
                f"{feature_name} has an immediate effect on {metric_name}. "
                f"Changes in {feature_name} {direction} {metric_name} within the same day. "
                f"Correlation strength: {strength}."
            )

        elif delayed:
            lag_desc = f"{optimal_lag} hours" if optimal_lag < 48 else f"{optimal_lag // 24} days"
            duration_desc = f"The effect lasts for {duration} hours." if duration else ""

            return (
                f"{feature_name} has a delayed effect on {metric_name}. "
                f"Changes in {feature_name} {direction} {metric_name} after {lag_desc}. "
                f"Correlation strength: {strength}. {duration_desc}"
            )

        else:
            return (
                f"No significant correlation found between {feature_name} and {metric_name} "
                f"at any tested lag (0-{optimal_lag} hours)."
            )

    # ========================================================================
    # Helper Functions
    # ========================================================================

    async def _build_feature_matrix(
        self, user_id: str, lookback_days: int
    ) -> Tuple[pd.DataFrame, List[date]]:
        """
        Build a feature matrix (one row per day) for the lookback period.

        Returns:
            (feature_df, dates) where feature_df has features as columns, dates as rows
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        # Collect features for each day
        features_by_date = {}
        dates = []

        current_date = start_date
        while current_date <= end_date:
            try:
                # Engineer features for this date
                response = await self.feature_service.engineer_features(
                    user_id=user_id,
                    target_date=current_date,
                    categories=[FeatureCategory.ALL],
                    lookback_days=30,
                    force_recompute=False,
                )

                # Flatten all features into a single dict
                flat_features = {}

                if response.nutrition:
                    flat_features.update({
                        f"nutrition_{k}": v
                        for k, v in response.nutrition.model_dump().items()
                        if v is not None
                    })

                if response.activity:
                    flat_features.update({
                        f"activity_{k}": v
                        for k, v in response.activity.model_dump().items()
                        if v is not None
                    })

                if response.health:
                    flat_features.update({
                        f"health_{k}": v
                        for k, v in response.health.model_dump().items()
                        if v is not None
                    })

                if response.temporal:
                    flat_features.update({
                        f"temporal_{k}": v
                        for k, v in response.temporal.model_dump().items()
                        if v is not None
                    })

                if response.interaction:
                    flat_features.update({
                        f"interaction_{k}": v
                        for k, v in response.interaction.model_dump().items()
                        if v is not None
                    })

                features_by_date[current_date] = flat_features
                dates.append(current_date)

            except Exception as e:
                print(f"Error engineering features for {current_date}: {e}")

            current_date += timedelta(days=1)

        # Convert to DataFrame
        if not features_by_date:
            return pd.DataFrame(), []

        feature_df = pd.DataFrame.from_dict(features_by_date, orient="index")
        return feature_df, dates

    async def _fetch_target_metric(
        self, user_id: str, target_metric: HealthMetricTarget, dates: List[date]
    ) -> pd.Series:
        """
        Fetch target health metric values for the given dates.

        Returns:
            pd.Series with date as index and metric values
        """
        from app.models.health_metric import HealthMetric
        from sqlalchemy import select, and_

        start_date = min(dates)
        end_date = max(dates)

        result = await self.db.execute(
            select(HealthMetric).where(
                and_(
                    HealthMetric.user_id == user_id,
                    HealthMetric.metric_type == target_metric.value,
                    HealthMetric.recorded_at >= datetime.combine(start_date, datetime.min.time()),
                    HealthMetric.recorded_at <= datetime.combine(end_date, datetime.max.time()),
                )
            )
        )
        metrics = result.scalars().all()

        if not metrics:
            return pd.Series(dtype=float)

        # Group by date and take average (in case multiple readings per day)
        data = {}
        for metric in metrics:
            metric_date = metric.recorded_at.date()
            if metric_date not in data:
                data[metric_date] = []
            data[metric_date].append(metric.value)

        # Average multiple readings per day
        averaged = {d: np.mean(values) for d, values in data.items()}

        return pd.Series(averaged)

    def _align_data(
        self, feature_matrix: pd.DataFrame, target_values: pd.Series, dates: List[date]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align feature matrix and target values.

        Remove days where target metric is missing.

        Returns:
            (aligned_features, aligned_target)
        """
        # Find dates with both features and target
        valid_dates = [d for d in dates if d in target_values.index]

        # Filter to valid dates
        aligned_features = feature_matrix.loc[valid_dates]
        aligned_target = target_values.loc[valid_dates]

        return aligned_features, aligned_target

    def _empty_correlation_response(
        self, request: CorrelationRequest
    ) -> CorrelationResponse:
        """Return empty response when no data available."""
        return CorrelationResponse(
            user_id=request.user_id,
            target_metric=request.target_metric,
            analyzed_at=datetime.now(UTC),
            lookback_days=request.lookback_days,
            correlations=[],
            total_features_analyzed=0,
            significant_correlations=0,
            data_quality_score=0.0,
            missing_days=request.lookback_days,
            warning="No data available for correlation analysis",
        )

    def _empty_lag_response(self, request: LagAnalysisRequest) -> LagAnalysisResponse:
        """Return empty response when no data available for lag analysis."""
        return LagAnalysisResponse(
            user_id=request.user_id,
            target_metric=request.target_metric,
            feature_name=request.feature_name,
            analyzed_at=datetime.now(UTC),
            lag_results=[],
            optimal_lag_hours=None,
            optimal_correlation=None,
            immediate_effect=False,
            delayed_effect=False,
            effect_duration_hours=None,
            interpretation="No data available for lag analysis",
        )
