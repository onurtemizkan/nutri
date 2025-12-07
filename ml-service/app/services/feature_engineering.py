"""
Feature Engineering Service

Transforms raw data (meals, activities, health metrics) into ML-ready features.
Implements 50+ features across 5 categories: nutrition, activity, health, temporal, interaction.
"""

from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meal import Meal
from app.models.activity import Activity
from app.models.health_metric import HealthMetric
from app.models.user import User
from app.schemas.features import (
    NutritionFeatures,
    ActivityFeatures,
    HealthFeatures,
    TemporalFeatures,
    InteractionFeatures,
    EngineerFeaturesResponse,
    FeatureCategory,
)
from app.redis_client import redis_client


class FeatureEngineeringService:
    """
    Service for engineering ML features from raw data.

    Features are computed for a specific user and target date, with configurable
    lookback period for historical data.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    async def engineer_features(
        self,
        user_id: str,
        target_date: date,
        categories: List[FeatureCategory],
        lookback_days: int = 30,
        force_recompute: bool = False,
    ) -> EngineerFeaturesResponse:
        """
        Engineer all requested feature categories for a user and date.

        Args:
            user_id: User ID
            target_date: Date to engineer features for
            categories: List of feature categories to compute
            lookback_days: Days of historical data to use (7-90)
            force_recompute: If True, ignore cache and recompute

        Returns:
            EngineerFeaturesResponse with all computed features
        """
        # Check cache first
        if not force_recompute:
            cached = await self._get_cached_features(user_id, target_date)
            if cached:
                return cached

        # Determine which categories to compute
        compute_all = FeatureCategory.ALL in categories
        compute_nutrition = compute_all or FeatureCategory.NUTRITION in categories
        compute_activity = compute_all or FeatureCategory.ACTIVITY in categories
        compute_health = compute_all or FeatureCategory.HEALTH in categories
        compute_temporal = compute_all or FeatureCategory.TEMPORAL in categories
        compute_interaction = compute_all or FeatureCategory.INTERACTION in categories

        # Fetch user data (needed for some features like calorie_deficit)
        user = await self._get_user(user_id)

        # Fetch raw data
        meals_df = await self._fetch_meals(user_id, target_date, lookback_days)
        activities_df = await self._fetch_activities(
            user_id, target_date, lookback_days
        )
        health_df = await self._fetch_health_metrics(
            user_id, target_date, lookback_days
        )

        # Engineer features by category
        nutrition = None
        activity = None
        health = None
        temporal = None
        interaction = None

        if compute_nutrition:
            nutrition = await self._engineer_nutrition_features(
                meals_df, target_date, user
            )

        if compute_activity:
            activity = await self._engineer_activity_features(
                activities_df, target_date
            )

        if compute_health:
            health = await self._engineer_health_features(health_df, target_date)

        if compute_temporal:
            temporal = self._engineer_temporal_features(target_date)

        if compute_interaction:
            interaction = self._engineer_interaction_features(
                nutrition, activity, health, user
            )

        # Calculate data quality metrics
        (
            feature_count,
            missing_features,
            quality_score,
        ) = self._calculate_quality_metrics(
            nutrition, activity, health, temporal, interaction
        )

        response = EngineerFeaturesResponse(
            user_id=user_id,
            target_date=target_date,
            computed_at=datetime.now(timezone.utc),
            cached=False,
            nutrition=nutrition,
            activity=activity,
            health=health,
            temporal=temporal,
            interaction=interaction,
            feature_count=feature_count,
            missing_features=missing_features,
            data_quality_score=quality_score,
        )

        # Cache the result
        await self._cache_features(user_id, target_date, response)

        return response

    # ========================================================================
    # Nutrition Features (16 features)
    # ========================================================================

    async def _engineer_nutrition_features(
        self,
        meals_df: pd.DataFrame,
        target_date: date,
        user: Optional[User],
    ) -> NutritionFeatures:
        """
        Engineer nutrition features from meal data.

        Features:
        - Daily totals (5): calories, protein, carbs, fat, fiber
        - Rolling averages (4): 7-day avg for main macros
        - Macro ratios (3): protein/carbs/fat as % of total calories
        - Meal timing (4): first meal, last meal, eating window, meal count
        - Late night eating (2): carbs and calories after 8pm
        - Regularity (1): consistency of meal timing
        - Calorie balance (2): deficit/surplus vs TDEE
        """
        if meals_df.empty:
            # Return zeros if no meal data
            return self._empty_nutrition_features()

        # Filter for target date
        today_meals = meals_df[meals_df["date"] == target_date]

        # Daily totals
        calories_daily = today_meals["calories"].sum()
        protein_daily = today_meals["protein"].sum()
        carbs_daily = today_meals["carbs"].sum()
        fat_daily = today_meals["fat"].sum()
        fiber_daily = today_meals["fiber"].sum()

        # Rolling averages (7 days)
        last_7_days = meals_df[meals_df["date"] > (target_date - timedelta(days=7))]
        daily_sums = last_7_days.groupby("date").agg(
            {
                "calories": "sum",
                "protein": "sum",
                "carbs": "sum",
                "fat": "sum",
            }
        )

        calories_7d_avg = daily_sums["calories"].mean() if not daily_sums.empty else 0
        protein_7d_avg = daily_sums["protein"].mean() if not daily_sums.empty else 0
        carbs_7d_avg = daily_sums["carbs"].mean() if not daily_sums.empty else 0
        fat_7d_avg = daily_sums["fat"].mean() if not daily_sums.empty else 0

        # Macro ratios
        total_cal = (
            calories_daily if calories_daily > 0 else 1
        )  # Avoid division by zero
        protein_ratio = (protein_daily * 4) / total_cal
        carbs_ratio = (carbs_daily * 4) / total_cal
        fat_ratio = (fat_daily * 9) / total_cal

        # Meal timing
        meal_count = len(today_meals)
        first_meal_time = None
        last_meal_time = None
        eating_window = None

        if not today_meals.empty:
            meal_times = today_meals["consumed_at"].dt.hour + (
                today_meals["consumed_at"].dt.minute / 60
            )
            first_meal_time = float(meal_times.min())
            last_meal_time = float(meal_times.max())
            eating_window = last_meal_time - first_meal_time if meal_count > 1 else 0

        # Late night eating (after 8pm)
        late_meals = today_meals[today_meals["consumed_at"].dt.hour >= 20]
        late_night_carbs = late_meals["carbs"].sum()
        late_night_calories = late_meals["calories"].sum()

        # Meal regularity (consistency of meal timing over 7 days)
        meal_regularity = self._calculate_meal_regularity(last_7_days)

        # Calorie balance (estimate user TDEE from goal calories)
        calorie_deficit = None
        calorie_deficit_7d = None
        user_tdee = self._estimate_tdee(user) if user else None
        if user_tdee:
            calorie_deficit = user_tdee - calories_daily
            if not daily_sums.empty:
                calorie_deficit_7d = user_tdee - daily_sums["calories"].mean()

        return NutritionFeatures(
            calories_daily=calories_daily,
            protein_daily=protein_daily,
            carbs_daily=carbs_daily,
            fat_daily=fat_daily,
            fiber_daily=fiber_daily,
            calories_7d_avg=calories_7d_avg,
            protein_7d_avg=protein_7d_avg,
            carbs_7d_avg=carbs_7d_avg,
            fat_7d_avg=fat_7d_avg,
            protein_ratio=min(protein_ratio, 1.0),
            carbs_ratio=min(carbs_ratio, 1.0),
            fat_ratio=min(fat_ratio, 1.0),
            first_meal_time=first_meal_time,
            last_meal_time=last_meal_time,
            eating_window=eating_window,
            meal_count=meal_count,
            late_night_carbs=late_night_carbs,
            late_night_calories=late_night_calories,
            meal_regularity=meal_regularity,
            calorie_deficit=calorie_deficit,
            calorie_deficit_7d=calorie_deficit_7d,
        )

    def _calculate_meal_regularity(self, meals_df: pd.DataFrame) -> float:
        """
        Calculate meal timing regularity (0-1).

        Higher value = more consistent meal times across days.
        Uses standard deviation of first meal time.
        """
        if meals_df.empty:
            return 0.0

        daily_first_meals = meals_df.groupby("date")["consumed_at"].min()
        if len(daily_first_meals) < 2:
            return 1.0  # Only 1 day, assume regular

        first_meal_hours = daily_first_meals.dt.hour + (
            daily_first_meals.dt.minute / 60
        )
        std = first_meal_hours.std()

        # Map std to 0-1 (lower std = higher regularity)
        # std of 0-1 hour = 1.0, std of 3+ hours = 0.0
        regularity = max(0, 1.0 - (std / 3.0))
        return regularity

    def _estimate_tdee(self, user: User) -> Optional[float]:
        """
        Estimate user's Total Daily Energy Expenditure (TDEE).

        Uses goal_calories as baseline (users typically set goals near their TDEE).
        Adjusts based on activity_level if available.

        Returns:
            Estimated TDEE in calories, or None if insufficient data
        """
        if not user:
            return None

        # Use goal_calories as baseline estimate
        # Most users set their calorie goal based on their TDEE +/- deficit
        base_calories: float = (
            float(user.goal_calories) if user.goal_calories else 2000.0
        )

        # Activity level multipliers (rough estimates)
        activity_multipliers = {
            "sedentary": 0.9,  # Less than goal (deficit for weight loss)
            "light": 0.95,  # Slightly less
            "moderate": 1.0,  # At goal
            "active": 1.05,  # Slightly more (maintenance for active people)
            "very_active": 1.1,  # More (maintenance for very active people)
        }

        # Get activity level (default to moderate)
        activity_level: str = (
            str(user.activity_level) if hasattr(user, "activity_level") else "moderate"
        )
        multiplier = activity_multipliers.get(activity_level, 1.0)

        estimated_tdee = base_calories * multiplier

        return estimated_tdee

    def _empty_nutrition_features(self) -> NutritionFeatures:
        """Return nutrition features with all zeros (no data available)."""
        return NutritionFeatures(
            calories_daily=0,
            protein_daily=0,
            carbs_daily=0,
            fat_daily=0,
            fiber_daily=0,
            calories_7d_avg=0,
            protein_7d_avg=0,
            carbs_7d_avg=0,
            fat_7d_avg=0,
            protein_ratio=0,
            carbs_ratio=0,
            fat_ratio=0,
            first_meal_time=None,
            last_meal_time=None,
            eating_window=None,
            meal_count=0,
            late_night_carbs=0,
            late_night_calories=0,
            meal_regularity=0,
            calorie_deficit=None,
            calorie_deficit_7d=None,
        )

    # ========================================================================
    # Activity Features (12 features)
    # ========================================================================

    async def _engineer_activity_features(
        self,
        activities_df: pd.DataFrame,
        target_date: date,
    ) -> ActivityFeatures:
        """
        Engineer activity features from workout and step data.

        Features:
        - Daily activity (3): steps, active minutes, calories burned
        - Rolling averages (3): 7-day avg for daily metrics
        - Workout intensity (3): count, avg intensity, high-intensity minutes
        - Recovery (2): hours since last workout, days since rest
        - Activity type distribution (3): cardio/strength/flexibility minutes (7d)
        """
        if activities_df.empty:
            return self._empty_activity_features()

        # Filter for target date
        today_activities = activities_df[activities_df["date"] == target_date]

        # Daily activity
        steps_daily = int(
            today_activities[today_activities["activity_type"] == "WALKING"][
                "duration_minutes"
            ].sum()
            * 100
        )  # Rough estimate: 100 steps/min

        active_minutes_daily = today_activities["duration_minutes"].sum()
        calories_burned_daily = today_activities["calories_burned"].sum()

        # Rolling averages (7 days)
        last_7_days = activities_df[
            activities_df["date"] > (target_date - timedelta(days=7))
        ]
        daily_sums = last_7_days.groupby("date").agg(
            {
                "duration_minutes": "sum",
                "calories_burned": "sum",
            }
        )

        steps_7d_avg = 0  # TODO: Implement proper step tracking
        active_minutes_7d_avg = (
            daily_sums["duration_minutes"].mean() if not daily_sums.empty else 0
        )
        calories_burned_7d_avg = (
            daily_sums["calories_burned"].mean() if not daily_sums.empty else 0
        )

        # Workout intensity
        workout_count_daily = len(today_activities)
        if not today_activities.empty:
            intensity_values = today_activities["intensity"].map(
                {"LOW": 0.33, "MODERATE": 0.66, "HIGH": 1.0}
            )
            workout_intensity_avg = intensity_values.mean()
            # Handle NaN (occurs when all values are unmapped or no valid intensities)
            if pd.isna(workout_intensity_avg):
                workout_intensity_avg = None
        else:
            workout_intensity_avg = None
        high_intensity_minutes = today_activities[
            today_activities["intensity"] == "HIGH"
        ]["duration_minutes"].sum()

        # Recovery time (hours since last high-intensity workout)
        recovery_time = self._calculate_recovery_time(activities_df, target_date)

        # Days since rest (no workouts)
        days_since_rest = self._calculate_days_since_rest(activities_df, target_date)

        # Activity type distribution (7 days)
        cardio_types = ["RUNNING", "CYCLING", "SWIMMING", "WALKING", "ROWING"]
        strength_types = ["WEIGHT_TRAINING", "BODYWEIGHT", "CROSSFIT", "POWERLIFTING"]
        flexibility_types = ["YOGA", "STRETCHING", "PILATES"]

        cardio_minutes_7d = last_7_days[
            last_7_days["activity_type"].isin(cardio_types)
        ]["duration_minutes"].sum()

        strength_minutes_7d = last_7_days[
            last_7_days["activity_type"].isin(strength_types)
        ]["duration_minutes"].sum()

        flexibility_minutes_7d = last_7_days[
            last_7_days["activity_type"].isin(flexibility_types)
        ]["duration_minutes"].sum()

        return ActivityFeatures(
            steps_daily=steps_daily,
            active_minutes_daily=active_minutes_daily,
            calories_burned_daily=calories_burned_daily,
            steps_7d_avg=steps_7d_avg,
            active_minutes_7d_avg=active_minutes_7d_avg,
            calories_burned_7d_avg=calories_burned_7d_avg,
            workout_count_daily=workout_count_daily,
            workout_intensity_avg=workout_intensity_avg,
            high_intensity_minutes=high_intensity_minutes,
            recovery_time=recovery_time,
            days_since_rest=days_since_rest,
            cardio_minutes_7d=cardio_minutes_7d,
            strength_minutes_7d=strength_minutes_7d,
            flexibility_minutes_7d=flexibility_minutes_7d,
        )

    def _calculate_recovery_time(
        self, activities_df: pd.DataFrame, target_date: date
    ) -> float:
        """Calculate hours since last high-intensity workout."""
        high_intensity = activities_df[activities_df["intensity"] == "HIGH"]
        if high_intensity.empty:
            return 72.0  # Default: 3 days

        last_high = high_intensity[high_intensity["date"] <= target_date]["date"].max()
        if pd.isna(last_high):
            return 72.0

        hours = (target_date - last_high).days * 24
        return float(hours)

    def _calculate_days_since_rest(
        self, activities_df: pd.DataFrame, target_date: date
    ) -> int:
        """Calculate days since last rest day (no workouts)."""
        # Get dates with workouts
        workout_dates = set(activities_df[activities_df["date"] <= target_date]["date"])

        days_since_rest = 0
        check_date = target_date
        while check_date in workout_dates and days_since_rest < 30:
            days_since_rest += 1
            check_date -= timedelta(days=1)

        return days_since_rest

    def _empty_activity_features(self) -> ActivityFeatures:
        """Return activity features with all zeros (no data available)."""
        return ActivityFeatures(
            steps_daily=0,
            active_minutes_daily=0,
            calories_burned_daily=0,
            steps_7d_avg=0,
            active_minutes_7d_avg=0,
            calories_burned_7d_avg=0,
            workout_count_daily=0,
            workout_intensity_avg=None,
            high_intensity_minutes=0,
            recovery_time=72.0,
            days_since_rest=7,
            cardio_minutes_7d=0,
            strength_minutes_7d=0,
            flexibility_minutes_7d=0,
        )

    # ========================================================================
    # Health Features (12 features)
    # ========================================================================

    async def _engineer_health_features(
        self,
        health_df: pd.DataFrame,
        target_date: date,
    ) -> HealthFeatures:
        """
        Engineer health features (lagged values and trends).

        Features:
        - RHR (6): yesterday, 7d avg/std, trend, baseline, deviation
        - HRV (6): yesterday, 7d avg/std, trend, baseline, deviation
        - Sleep (3): duration last night, quality, 7d avg
        - Recovery (2): score yesterday, 7d avg
        """
        if health_df.empty:
            return self._empty_health_features()

        # RHR features
        rhr_df = health_df[health_df["metric_type"] == "RESTING_HEART_RATE"]
        (
            rhr_yesterday,
            rhr_7d_avg,
            rhr_7d_std,
            rhr_trend,
            rhr_baseline,
            rhr_deviation,
        ) = self._calculate_metric_features(rhr_df, target_date)

        # HRV features (using RMSSD as primary)
        hrv_df = health_df[health_df["metric_type"] == "HEART_RATE_VARIABILITY_RMSSD"]
        (
            hrv_yesterday,
            hrv_7d_avg,
            hrv_7d_std,
            hrv_trend,
            hrv_baseline,
            hrv_deviation,
        ) = self._calculate_metric_features(hrv_df, target_date)

        # Sleep features
        sleep_duration_df = health_df[health_df["metric_type"] == "SLEEP_DURATION"]
        sleep_quality_df = health_df[health_df["metric_type"] == "SLEEP_SCORE"]

        sleep_duration_last = self._get_yesterday_value(sleep_duration_df, target_date)
        sleep_quality_last = self._get_yesterday_value(sleep_quality_df, target_date)
        sleep_duration_7d_avg = self._calculate_7d_avg(sleep_duration_df, target_date)

        # Recovery score features
        recovery_df = health_df[health_df["metric_type"] == "RECOVERY_SCORE"]
        recovery_score_yesterday = self._get_yesterday_value(recovery_df, target_date)
        recovery_score_7d_avg = self._calculate_7d_avg(recovery_df, target_date)

        return HealthFeatures(
            rhr_yesterday=rhr_yesterday,
            rhr_7d_avg=rhr_7d_avg,
            rhr_7d_std=rhr_7d_std,
            rhr_trend=rhr_trend,
            rhr_baseline=rhr_baseline,
            rhr_deviation=rhr_deviation,
            hrv_yesterday=hrv_yesterday,
            hrv_7d_avg=hrv_7d_avg,
            hrv_7d_std=hrv_7d_std,
            hrv_trend=hrv_trend,
            hrv_baseline=hrv_baseline,
            hrv_deviation=hrv_deviation,
            sleep_duration_last=sleep_duration_last,
            sleep_quality_last=sleep_quality_last,
            sleep_duration_7d_avg=sleep_duration_7d_avg,
            recovery_score_yesterday=recovery_score_yesterday,
            recovery_score_7d_avg=recovery_score_7d_avg,
        )

    def _calculate_metric_features(
        self, metric_df: pd.DataFrame, target_date: date
    ) -> Tuple[Optional[float], ...]:
        """
        Calculate comprehensive features for a health metric.

        Returns: (yesterday, 7d_avg, 7d_std, trend, baseline, deviation)
        """
        if metric_df.empty:
            return None, None, None, None, None, None

        # Yesterday's value
        yesterday = self._get_yesterday_value(metric_df, target_date)

        # 7-day statistics
        last_7_days = metric_df[
            (metric_df["date"] > (target_date - timedelta(days=7)))
            & (metric_df["date"] <= target_date)
        ]
        avg_7d = last_7_days["value"].mean() if not last_7_days.empty else None
        std_7d = last_7_days["value"].std() if not last_7_days.empty else None

        # Trend (linear slope over 7 days)
        trend = None
        if not last_7_days.empty and len(last_7_days) >= 3:
            x = np.arange(len(last_7_days))
            y = last_7_days["value"].values
            trend = float(np.polyfit(x, y, 1)[0])  # Slope of linear regression

        # 30-day baseline
        last_30_days = metric_df[
            (metric_df["date"] > (target_date - timedelta(days=30)))
            & (metric_df["date"] <= target_date)
        ]
        baseline = last_30_days["value"].mean() if not last_30_days.empty else None

        # Deviation from baseline
        deviation = None
        if yesterday is not None and baseline is not None:
            deviation = yesterday - baseline

        return yesterday, avg_7d, std_7d, trend, baseline, deviation

    def _get_yesterday_value(
        self, metric_df: pd.DataFrame, target_date: date
    ) -> Optional[float]:
        """Get metric value from yesterday."""
        yesterday = target_date - timedelta(days=1)
        yesterday_data = metric_df[metric_df["date"] == yesterday]
        if not yesterday_data.empty:
            return float(yesterday_data["value"].iloc[0])
        return None

    def _calculate_7d_avg(
        self, metric_df: pd.DataFrame, target_date: date
    ) -> Optional[float]:
        """Calculate 7-day average for a metric."""
        last_7_days = metric_df[
            (metric_df["date"] > (target_date - timedelta(days=7)))
            & (metric_df["date"] <= target_date)
        ]
        if not last_7_days.empty:
            return float(last_7_days["value"].mean())
        return None

    def _empty_health_features(self) -> HealthFeatures:
        """Return health features with all None (no data available)."""
        return HealthFeatures(
            rhr_yesterday=None,
            rhr_7d_avg=None,
            rhr_7d_std=None,
            rhr_trend=None,
            rhr_baseline=None,
            rhr_deviation=None,
            hrv_yesterday=None,
            hrv_7d_avg=None,
            hrv_7d_std=None,
            hrv_trend=None,
            hrv_baseline=None,
            hrv_deviation=None,
            sleep_duration_last=None,
            sleep_quality_last=None,
            sleep_duration_7d_avg=None,
            recovery_score_yesterday=None,
            recovery_score_7d_avg=None,
        )

    # ========================================================================
    # Temporal Features (5 features)
    # ========================================================================

    def _engineer_temporal_features(self, target_date: date) -> TemporalFeatures:
        """
        Engineer temporal features (time-based patterns).

        Features:
        - Basic temporal (4): day_of_week, is_weekend, week_of_year, month
        - Physiological cycles (2): menstrual cycle day and phase (if tracked)
        """
        dt = datetime.combine(target_date, datetime.min.time())

        return TemporalFeatures(
            day_of_week=dt.weekday(),  # 0=Monday, 6=Sunday
            is_weekend=dt.weekday() >= 5,
            week_of_year=dt.isocalendar()[1],
            month=dt.month,
            menstrual_cycle_day=None,  # TODO: Implement cycle tracking
            menstrual_cycle_phase=None,
        )

    # ========================================================================
    # Interaction Features (6 features)
    # ========================================================================

    def _engineer_interaction_features(
        self,
        nutrition: Optional[NutritionFeatures],
        activity: Optional[ActivityFeatures],
        health: Optional[HealthFeatures],
        user: Optional[User],
    ) -> InteractionFeatures:
        """
        Engineer interaction features (combinations of other features).

        Features:
        - Nutrition per body weight (2): protein/kg, calories/kg
        - Nutrition per activity (2): carbs per active minute, protein per workout
        - Recovery-adjusted (2): protein to recovery time, carbs to intensity
        """
        # Nutrition per body weight
        protein_per_kg = None
        calories_per_kg = None
        user_weight = user.current_weight if (user and user.current_weight) else None
        if user_weight and nutrition:
            protein_per_kg = nutrition.protein_daily / user_weight
            calories_per_kg = nutrition.calories_daily / user_weight

        # Nutrition per activity
        carbs_per_active_minute = None
        protein_per_workout = None
        if nutrition and activity:
            if activity.active_minutes_daily > 0:
                carbs_per_active_minute = (
                    nutrition.carbs_daily / activity.active_minutes_daily
                )
            if activity.workout_count_daily > 0:
                protein_per_workout = (
                    nutrition.protein_daily / activity.workout_count_daily
                )

        # Recovery-adjusted nutrition
        protein_to_recovery = None
        carbs_to_intensity = None
        if nutrition and activity:
            if activity.recovery_time > 0:
                protein_to_recovery = nutrition.protein_daily / activity.recovery_time
            if activity.workout_intensity_avg and activity.workout_intensity_avg > 0:
                carbs_to_intensity = (
                    nutrition.carbs_daily / activity.workout_intensity_avg
                )

        return InteractionFeatures(
            protein_per_kg=protein_per_kg,
            calories_per_kg=calories_per_kg,
            carbs_per_active_minute=carbs_per_active_minute,
            protein_per_workout=protein_per_workout,
            protein_to_recovery=protein_to_recovery,
            carbs_to_intensity=carbs_to_intensity,
        )

    # ========================================================================
    # Data Fetching
    # ========================================================================

    async def _get_user(self, user_id: str) -> Optional[User]:
        """Fetch user from database."""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def _fetch_meals(
        self, user_id: str, target_date: date, lookback_days: int
    ) -> pd.DataFrame:
        """Fetch meals for the lookback period and convert to DataFrame."""
        start_date = target_date - timedelta(days=lookback_days)

        result = await self.db.execute(
            select(Meal).where(
                and_(
                    Meal.user_id == user_id,
                    Meal.consumed_at
                    >= datetime.combine(start_date, datetime.min.time()),
                    Meal.consumed_at
                    <= datetime.combine(target_date, datetime.max.time()),
                )
            )
        )
        meals = result.scalars().all()

        if not meals:
            return pd.DataFrame()

        data = [
            {
                "date": meal.consumed_at.date(),
                "consumed_at": meal.consumed_at,
                "calories": meal.calories,
                "protein": meal.protein,
                "carbs": meal.carbs,
                "fat": meal.fat,
                "fiber": meal.fiber or 0,
            }
            for meal in meals
        ]

        return pd.DataFrame(data)

    async def _fetch_activities(
        self, user_id: str, target_date: date, lookback_days: int
    ) -> pd.DataFrame:
        """Fetch activities for the lookback period and convert to DataFrame."""
        start_date = target_date - timedelta(days=lookback_days)

        result = await self.db.execute(
            select(Activity).where(
                and_(
                    Activity.user_id == user_id,
                    Activity.started_at
                    >= datetime.combine(start_date, datetime.min.time()),
                    Activity.started_at
                    <= datetime.combine(target_date, datetime.max.time()),
                )
            )
        )
        activities = result.scalars().all()

        if not activities:
            return pd.DataFrame()

        data = [
            {
                "date": activity.started_at.date(),
                "activity_type": activity.activity_type,
                "intensity": activity.intensity,
                "duration_minutes": activity.duration,
                "calories_burned": activity.calories_burned or 0,
            }
            for activity in activities
        ]

        return pd.DataFrame(data)

    async def _fetch_health_metrics(
        self, user_id: str, target_date: date, lookback_days: int
    ) -> pd.DataFrame:
        """Fetch health metrics for the lookback period and convert to DataFrame."""
        start_date = target_date - timedelta(days=lookback_days)

        result = await self.db.execute(
            select(HealthMetric).where(
                and_(
                    HealthMetric.user_id == user_id,
                    HealthMetric.recorded_at
                    >= datetime.combine(start_date, datetime.min.time()),
                    HealthMetric.recorded_at
                    <= datetime.combine(target_date, datetime.max.time()),
                )
            )
        )
        metrics = result.scalars().all()

        if not metrics:
            return pd.DataFrame()

        data = [
            {
                "date": metric.recorded_at.date(),
                "metric_type": metric.metric_type,
                "value": metric.value,
            }
            for metric in metrics
        ]

        return pd.DataFrame(data)

    # ========================================================================
    # Quality Metrics
    # ========================================================================

    def _calculate_quality_metrics(
        self,
        nutrition: Optional[NutritionFeatures],
        activity: Optional[ActivityFeatures],
        health: Optional[HealthFeatures],
        temporal: Optional[TemporalFeatures],
        interaction: Optional[InteractionFeatures],
    ) -> Tuple[int, int, float]:
        """
        Calculate feature count, missing features, and quality score.

        Returns: (feature_count, missing_features, quality_score)
        """
        total_features = 0
        missing_features = 0

        # Count features in each category
        for feature_set in [nutrition, activity, health, temporal, interaction]:
            if feature_set:
                fields = feature_set.model_dump()
                total_features += len(fields)
                missing_features += sum(1 for v in fields.values() if v is None)

        # Quality score: 1.0 - (missing / total)
        quality_score = (
            1.0 - (missing_features / total_features) if total_features > 0 else 0.0
        )

        return total_features, missing_features, quality_score

    # ========================================================================
    # Caching
    # ========================================================================

    async def _get_cached_features(
        self, user_id: str, target_date: date
    ) -> Optional[EngineerFeaturesResponse]:
        """Retrieve cached features from Redis."""
        key = f"features:{user_id}:{target_date}:all"
        cached_json = await redis_client.get(key)

        if cached_json:
            data = cached_json
            data["cached"] = True
            data["computed_at"] = datetime.fromisoformat(data["computed_at"])
            data["target_date"] = date.fromisoformat(data["target_date"])
            return EngineerFeaturesResponse(**data)

        return None

    async def _cache_features(
        self, user_id: str, target_date: date, response: EngineerFeaturesResponse
    ) -> None:
        """Cache features in Redis."""
        # Note: key format is f"features:{user_id}:{target_date}:all"
        await redis_client.cache_features(
            user_id, str(target_date), "all", response.model_dump(mode="json")
        )
