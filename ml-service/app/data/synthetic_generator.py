"""
Comprehensive Synthetic Health Data Generator

Generates realistic, diverse synthetic datasets for training and evaluating
LSTM models for health metric prediction. Based on research from:
- NeuroKit2 for HRV simulation patterns
- Synthea synthetic patient generation principles
- Real-world health metric distributions from wearable data studies

Features:
- 5 diverse user personas with different lifestyles
- Realistic correlations between nutrition and health metrics
- Temporal patterns (weekday/weekend, seasonal)
- Circadian rhythm effects on health metrics
- Activity-recovery relationships
"""
# mypy: ignore-errors

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import random


class UserPersona(Enum):
    """Five diverse user personas with different lifestyle patterns."""

    ATHLETE = "athlete"  # High activity, strict nutrition, excellent recovery
    OFFICE_WORKER = "office_worker"  # Sedentary, irregular meals, moderate stress
    HEALTH_ENTHUSIAST = "health_enthusiast"  # Balanced, tracks everything, improving
    SHIFT_WORKER = "shift_worker"  # Irregular sleep, variable meal times
    STUDENT = "student"  # Late nights, irregular patterns, stress peaks


@dataclass
class PersonaConfig:
    """Configuration for each user persona."""

    name: str
    age: int
    weight_kg: float
    height_cm: float
    gender: str
    activity_level: str

    # Health baselines
    rhr_baseline: float
    rhr_std: float
    hrv_baseline: float  # RMSSD
    hrv_std: float
    sleep_baseline: float  # hours
    sleep_std: float

    # Nutrition patterns
    calories_target: int
    protein_target: float  # g/kg bodyweight
    meal_regularity: float  # 0-1 (1 = very regular)
    late_eating_prob: float  # probability of eating after 8pm

    # Activity patterns
    workouts_per_week: float
    avg_workout_duration: int  # minutes
    high_intensity_ratio: float  # 0-1

    # Lifestyle factors
    stress_level: float  # 0-1
    sleep_regularity: float  # 0-1
    alcohol_frequency: float  # drinks per week

    # Correlation strengths (how much nutrition affects health)
    nutrition_health_correlation: float  # 0-1


# Define the 5 diverse user personas
PERSONA_CONFIGS: Dict[UserPersona, PersonaConfig] = {
    UserPersona.ATHLETE: PersonaConfig(
        name="Athletic Alex",
        age=28,
        weight_kg=75.0,
        height_cm=180.0,
        gender="male",
        activity_level="very_active",
        rhr_baseline=52.0,
        rhr_std=3.0,
        hrv_baseline=65.0,
        hrv_std=12.0,
        sleep_baseline=8.0,
        sleep_std=0.5,
        calories_target=3000,
        protein_target=2.0,
        meal_regularity=0.9,
        late_eating_prob=0.1,
        workouts_per_week=6.0,
        avg_workout_duration=75,
        high_intensity_ratio=0.4,
        stress_level=0.2,
        sleep_regularity=0.9,
        alcohol_frequency=1.0,
        nutrition_health_correlation=0.7,
    ),
    UserPersona.OFFICE_WORKER: PersonaConfig(
        name="Office Oliver",
        age=35,
        weight_kg=85.0,
        height_cm=175.0,
        gender="male",
        activity_level="sedentary",
        rhr_baseline=72.0,
        rhr_std=5.0,
        hrv_baseline=35.0,
        hrv_std=8.0,
        sleep_baseline=6.5,
        sleep_std=1.0,
        calories_target=2200,
        protein_target=1.0,
        meal_regularity=0.5,
        late_eating_prob=0.4,
        workouts_per_week=1.5,
        avg_workout_duration=30,
        high_intensity_ratio=0.1,
        stress_level=0.6,
        sleep_regularity=0.5,
        alcohol_frequency=4.0,
        nutrition_health_correlation=0.5,
    ),
    UserPersona.HEALTH_ENTHUSIAST: PersonaConfig(
        name="Healthy Hannah",
        age=32,
        weight_kg=62.0,
        height_cm=165.0,
        gender="female",
        activity_level="active",
        rhr_baseline=60.0,
        rhr_std=4.0,
        hrv_baseline=50.0,
        hrv_std=10.0,
        sleep_baseline=7.5,
        sleep_std=0.7,
        calories_target=1900,
        protein_target=1.5,
        meal_regularity=0.8,
        late_eating_prob=0.15,
        workouts_per_week=4.0,
        avg_workout_duration=50,
        high_intensity_ratio=0.25,
        stress_level=0.3,
        sleep_regularity=0.8,
        alcohol_frequency=2.0,
        nutrition_health_correlation=0.65,
    ),
    UserPersona.SHIFT_WORKER: PersonaConfig(
        name="Shift-work Sam",
        age=40,
        weight_kg=78.0,
        height_cm=172.0,
        gender="male",
        activity_level="light",
        rhr_baseline=68.0,
        rhr_std=6.0,
        hrv_baseline=38.0,
        hrv_std=10.0,
        sleep_baseline=6.0,
        sleep_std=1.5,
        calories_target=2400,
        protein_target=1.2,
        meal_regularity=0.3,
        late_eating_prob=0.6,
        workouts_per_week=2.0,
        avg_workout_duration=40,
        high_intensity_ratio=0.15,
        stress_level=0.5,
        sleep_regularity=0.3,
        alcohol_frequency=3.0,
        nutrition_health_correlation=0.4,
    ),
    UserPersona.STUDENT: PersonaConfig(
        name="Student Sofia",
        age=22,
        weight_kg=58.0,
        height_cm=168.0,
        gender="female",
        activity_level="moderate",
        rhr_baseline=65.0,
        rhr_std=5.0,
        hrv_baseline=55.0,
        hrv_std=15.0,
        sleep_baseline=6.5,
        sleep_std=1.2,
        calories_target=1800,
        protein_target=1.0,
        meal_regularity=0.4,
        late_eating_prob=0.5,
        workouts_per_week=2.5,
        avg_workout_duration=35,
        high_intensity_ratio=0.2,
        stress_level=0.55,
        sleep_regularity=0.4,
        alcohol_frequency=3.5,
        nutrition_health_correlation=0.55,
    ),
}


class SyntheticDataGenerator:
    """
    Generates realistic synthetic health data for model training.

    Based on research findings:
    - RHR typically ranges 40-100 bpm, with athletes lower
    - HRV (RMSSD) typically ranges 20-100ms, highly individual
    - Nutrition affects next-day HRV and RHR with 12-48 hour lag
    - High protein intake correlates with better HRV (+0.3-0.5 correlation)
    - Late night eating negatively affects sleep and next-day HRV
    - High-intensity exercise temporarily increases RHR, then decreases
    - Sleep quality strongly affects next-day HRV
    """

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_user_data(
        self,
        persona: UserPersona,
        user_id: str,
        num_days: int = 180,
        start_date: Optional[date] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate complete synthetic data for a user persona.

        Args:
            persona: User persona type
            user_id: Unique user identifier
            num_days: Number of days of data to generate
            start_date: Starting date (defaults to 180 days ago)

        Returns:
            Dictionary with DataFrames for meals, activities, health_metrics
        """
        config = PERSONA_CONFIGS[persona]

        if start_date is None:
            start_date = date.today() - timedelta(days=num_days)

        # Generate data for each day
        all_meals = []
        all_activities = []
        all_health_metrics = []

        # Track rolling state for temporal correlations
        state = {
            "cumulative_fatigue": 0.0,
            "sleep_debt": 0.0,
            "nutrition_quality_7d": [],
            "activity_load_7d": [],
            "stress_events": [],
        }

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)

            # Generate daily data
            daily_meals = self._generate_daily_meals(
                config, user_id, current_date, state
            )
            daily_activities = self._generate_daily_activities(
                config, user_id, current_date, state
            )
            daily_health = self._generate_daily_health_metrics(
                config, user_id, current_date, state, daily_meals, daily_activities
            )

            all_meals.extend(daily_meals)
            all_activities.extend(daily_activities)
            all_health_metrics.extend(daily_health)

            # Update rolling state
            self._update_state(
                state, daily_meals, daily_activities, daily_health, config
            )

        return {
            "meals": pd.DataFrame(all_meals),
            "activities": pd.DataFrame(all_activities),
            "health_metrics": pd.DataFrame(all_health_metrics),
            "user": self._generate_user_record(config, user_id),
        }

    def _generate_daily_meals(
        self,
        config: PersonaConfig,
        user_id: str,
        current_date: date,
        state: Dict,
    ) -> List[Dict]:
        """Generate meals for a single day."""
        meals = []
        is_weekend = current_date.weekday() >= 5

        # Determine number of meals (3-5 typically)
        _ = 3
        snack_prob = 0.4 if config.meal_regularity > 0.7 else 0.6
        num_snacks = np.random.binomial(2, snack_prob)

        # Weekend patterns differ
        if is_weekend:
            breakfast_time_mean = 9.5 if config.sleep_regularity < 0.6 else 8.5
        else:
            breakfast_time_mean = 7.5

        # Generate main meals
        meal_times = {
            "breakfast": breakfast_time_mean
            + np.random.normal(0, 0.5 * (1 - config.meal_regularity)),
            "lunch": 12.5 + np.random.normal(0, 0.5 * (1 - config.meal_regularity)),
            "dinner": 18.5 + np.random.normal(0, 1.0 * (1 - config.meal_regularity)),
        }

        # Daily calorie distribution
        daily_calories = config.calories_target * np.random.normal(1.0, 0.15)
        if is_weekend:
            daily_calories *= np.random.uniform(1.0, 1.2)  # Weekend eating

        # Macro distribution (protein-focused for athletes, carb-focused for others)
        protein_g = (
            config.protein_target * config.weight_kg * np.random.normal(1.0, 0.1)
        )
        fat_ratio = np.random.uniform(0.25, 0.35)

        # Calculate macros
        protein_calories = protein_g * 4
        fat_calories = daily_calories * fat_ratio
        carb_calories = daily_calories - protein_calories - fat_calories

        carbs_g = carb_calories / 4
        fat_g = fat_calories / 9

        # Distribute across meals
        meal_distributions = {
            "breakfast": (0.25, 0.3),  # (mean, std)
            "lunch": (0.35, 0.05),
            "dinner": (0.35, 0.08),
        }

        for meal_type, time_hour in meal_times.items():
            dist_mean, dist_std = meal_distributions[meal_type]
            portion = max(0.1, np.random.normal(dist_mean, dist_std))

            # Add some variation
            meal_time = datetime.combine(current_date, datetime.min.time())
            meal_time = meal_time.replace(
                hour=int(time_hour),
                minute=int((time_hour % 1) * 60),
            )

            meals.append(
                {
                    "user_id": user_id,
                    "name": f"{meal_type.capitalize()}",
                    "meal_type": meal_type.upper(),
                    "calories": int(daily_calories * portion),
                    "protein": round(protein_g * portion, 1),
                    "carbs": round(carbs_g * portion, 1),
                    "fat": round(fat_g * portion, 1),
                    "fiber": round(np.random.uniform(5, 15) * portion, 1),
                    "sugar": round(np.random.uniform(10, 40) * portion, 1),
                    "consumed_at": meal_time,
                }
            )

        # Add snacks
        for i in range(num_snacks):
            snack_time = np.random.choice([10.5, 15.5, 21.0])

            # Late night snack check
            if snack_time >= 20 and np.random.random() > config.late_eating_prob:
                continue

            snack_calories = np.random.uniform(100, 300)
            snack_time_dt = datetime.combine(current_date, datetime.min.time())
            snack_time_dt = snack_time_dt.replace(
                hour=int(snack_time),
                minute=int((snack_time % 1) * 60),
            )

            meals.append(
                {
                    "user_id": user_id,
                    "name": f"Snack {i+1}",
                    "meal_type": "SNACK",
                    "calories": int(snack_calories),
                    "protein": round(snack_calories * 0.1 / 4, 1),
                    "carbs": round(snack_calories * 0.5 / 4, 1),
                    "fat": round(snack_calories * 0.4 / 9, 1),
                    "fiber": round(np.random.uniform(1, 5), 1),
                    "sugar": round(np.random.uniform(5, 20), 1),
                    "consumed_at": snack_time_dt,
                }
            )

        return meals

    def _generate_daily_activities(
        self,
        config: PersonaConfig,
        user_id: str,
        current_date: date,
        state: Dict,
    ) -> List[Dict]:
        """Generate activities for a single day."""
        activities = []
        is_weekend = current_date.weekday() >= 5
        day_of_week = current_date.weekday()

        # Calculate workout probability for this day
        workouts_per_day = config.workouts_per_week / 7
        workout_prob = min(1.0, workouts_per_day * (1.3 if is_weekend else 1.0))

        # Rest day pattern (every 3-4 days for athletes, random for others)
        if config.activity_level == "very_active":
            # Athletes typically rest every 3-4 days
            rest_day = day_of_week % 4 == 3
            if rest_day:
                workout_prob *= 0.2

        if np.random.random() < workout_prob:
            # Determine workout type
            activity_types = self._get_activity_types_for_persona(config)
            activity_type = np.random.choice(activity_types)

            # Workout timing
            if is_weekend:
                workout_hour = np.random.choice([9, 10, 11, 16, 17])
            else:
                workout_hour = np.random.choice([6, 7, 17, 18, 19])

            # Duration with variation
            duration = int(config.avg_workout_duration * np.random.normal(1.0, 0.2))
            duration = max(15, min(120, duration))

            # Intensity
            if np.random.random() < config.high_intensity_ratio:
                intensity = "HIGH"
                calories_per_min = np.random.uniform(10, 14)
            elif np.random.random() < 0.5:
                intensity = "MODERATE"
                calories_per_min = np.random.uniform(6, 10)
            else:
                intensity = "LOW"
                calories_per_min = np.random.uniform(3, 6)

            # Cumulative fatigue affects performance
            fatigue_factor = 1.0 - (state["cumulative_fatigue"] * 0.1)
            duration = int(duration * fatigue_factor)

            started_at = datetime.combine(current_date, datetime.min.time())
            started_at = started_at.replace(hour=workout_hour, minute=0)

            activities.append(
                {
                    "user_id": user_id,
                    "activity_type": activity_type,
                    "intensity": intensity,
                    "duration": duration,
                    "calories_burned": int(duration * calories_per_min),
                    "started_at": started_at,
                    "notes": None,
                }
            )

        # Add daily steps as walking activity
        base_steps = {
            "sedentary": 4000,
            "light": 6000,
            "moderate": 8000,
            "active": 10000,
            "very_active": 12000,
        }

        daily_steps = base_steps.get(config.activity_level, 7000)
        daily_steps = int(daily_steps * np.random.normal(1.0, 0.2))

        if is_weekend:
            daily_steps = int(daily_steps * np.random.uniform(0.7, 1.3))

        # Convert steps to walking activity (rough: 100 steps/minute)
        walking_minutes = daily_steps / 100

        activities.append(
            {
                "user_id": user_id,
                "activity_type": "WALKING",
                "intensity": "LOW",
                "duration": int(walking_minutes),
                "calories_burned": int(walking_minutes * 4),
                "started_at": datetime.combine(
                    current_date, datetime.min.time()
                ).replace(hour=12),
                "notes": f"Daily steps: {daily_steps}",
            }
        )

        return activities

    def _generate_daily_health_metrics(
        self,
        config: PersonaConfig,
        user_id: str,
        current_date: date,
        state: Dict,
        daily_meals: List[Dict],
        daily_activities: List[Dict],
    ) -> List[Dict]:
        """
        Generate health metrics for a single day.

        Implements realistic correlations:
        - High protein → better HRV (lag 1 day)
        - Late eating → worse sleep → worse HRV
        - High intensity exercise → elevated RHR next day
        - Poor sleep → elevated RHR, lower HRV
        - Stress → elevated RHR, lower HRV
        """
        metrics = []
        is_weekend = current_date.weekday() >= 5

        # Calculate nutrition quality score (0-1)
        total_calories = sum(m["calories"] for m in daily_meals)
        total_protein = sum(m["protein"] for m in daily_meals)

        protein_adequacy = min(
            1.0, total_protein / (config.protein_target * config.weight_kg)
        )
        calorie_adequacy = (
            1.0 - abs(total_calories - config.calories_target) / config.calories_target
        )
        calorie_adequacy = max(0, min(1, calorie_adequacy))

        # Check late eating
        late_meals = [m for m in daily_meals if m["consumed_at"].hour >= 20]
        late_eating_impact = len(late_meals) * 0.15

        # Calculate activity load
        high_intensity_mins = sum(
            a["duration"] for a in daily_activities if a.get("intensity") == "HIGH"
        )
        _ = sum(a["duration"] for a in daily_activities)

        # Get 7-day rolling averages from state
        avg_nutrition_quality = (
            np.mean(state["nutrition_quality_7d"][-7:])
            if state["nutrition_quality_7d"]
            else 0.5
        )
        avg_activity_load = (
            np.mean(state["activity_load_7d"][-7:]) if state["activity_load_7d"] else 0
        )

        # ========== RHR Calculation ==========
        # Base RHR with circadian variation
        rhr_base = config.rhr_baseline

        # Factors affecting RHR
        rhr_modifiers = []

        # 1. Sleep debt increases RHR (+1-5 bpm)
        rhr_modifiers.append(state["sleep_debt"] * 2)

        # 2. High activity yesterday increases RHR temporarily
        if state["activity_load_7d"]:
            yesterday_load = (
                state["activity_load_7d"][-1] if state["activity_load_7d"] else 0
            )
            rhr_modifiers.append(yesterday_load * 0.05)

        # 3. Poor nutrition increases RHR
        rhr_modifiers.append((1 - avg_nutrition_quality) * 3)

        # 4. Cumulative fatigue increases RHR
        rhr_modifiers.append(state["cumulative_fatigue"] * 2)

        # 5. Stress increases RHR
        rhr_modifiers.append(config.stress_level * 5)

        # 6. Weekend relaxation decreases RHR slightly
        if is_weekend:
            rhr_modifiers.append(-2)

        # 7. Alcohol increases RHR
        if config.alcohol_frequency > 0 and np.random.random() < (
            config.alcohol_frequency / 7
        ):
            rhr_modifiers.append(np.random.uniform(2, 6))

        # Apply correlation strength
        total_modifier = sum(rhr_modifiers) * config.nutrition_health_correlation

        rhr_value = rhr_base + total_modifier + np.random.normal(0, config.rhr_std)
        rhr_value = max(40, min(100, rhr_value))  # Physiological bounds

        # ========== HRV Calculation ==========
        # Base HRV (inversely related to RHR in general)
        hrv_base = config.hrv_baseline

        hrv_modifiers = []

        # 1. Good protein intake improves HRV (research shows +0.3-0.5 correlation)
        hrv_modifiers.append(protein_adequacy * 8)

        # 2. Sleep debt decreases HRV
        hrv_modifiers.append(-state["sleep_debt"] * 5)

        # 3. Late eating decreases HRV (through sleep disruption)
        hrv_modifiers.append(-late_eating_impact * 10)

        # 4. Moderate activity improves HRV long-term
        fitness_effect = min(5, avg_activity_load * 0.02)
        hrv_modifiers.append(fitness_effect)

        # 5. Excessive activity (overtraining) decreases HRV
        if state["cumulative_fatigue"] > 0.5:
            hrv_modifiers.append(-(state["cumulative_fatigue"] - 0.5) * 15)

        # 6. Stress decreases HRV significantly
        hrv_modifiers.append(-config.stress_level * 12)

        # 7. Weekend recovery improves HRV
        if is_weekend:
            hrv_modifiers.append(3)

        # Apply correlation strength
        total_hrv_modifier = sum(hrv_modifiers) * config.nutrition_health_correlation

        hrv_value = hrv_base + total_hrv_modifier + np.random.normal(0, config.hrv_std)
        hrv_value = max(10, min(150, hrv_value))  # Physiological bounds

        # ========== Sleep Calculation ==========
        sleep_base = config.sleep_baseline

        sleep_modifiers = []

        # 1. Late eating disrupts sleep
        sleep_modifiers.append(-late_eating_impact * 0.5)

        # 2. High activity can improve or disrupt sleep
        if high_intensity_mins > 60:
            sleep_modifiers.append(-0.3)  # Too much
        elif high_intensity_mins > 0:
            sleep_modifiers.append(0.2)  # Just right

        # 3. Weekend sleep-in
        if is_weekend and config.sleep_regularity < 0.7:
            sleep_modifiers.append(np.random.uniform(0.5, 1.5))

        # 4. Stress disrupts sleep
        sleep_modifiers.append(-config.stress_level * 1.0)

        # 5. Alcohol disrupts sleep quality
        if config.alcohol_frequency > 0 and np.random.random() < (
            config.alcohol_frequency / 7
        ):
            sleep_modifiers.append(-0.5)

        sleep_duration = (
            sleep_base + sum(sleep_modifiers) + np.random.normal(0, config.sleep_std)
        )
        sleep_duration = max(3, min(12, sleep_duration))

        # Sleep quality (0-100)
        sleep_quality = (
            70
            + (sleep_duration - 6) * 5
            - late_eating_impact * 10
            - config.stress_level * 15
        )
        sleep_quality = max(20, min(100, sleep_quality + np.random.normal(0, 8)))

        # ========== Recovery Score ==========
        # Composite metric based on HRV, RHR, and sleep
        rhr_component = (100 - rhr_value) / 60 * 30  # Lower RHR = better
        hrv_component = min(
            40, hrv_value / config.hrv_baseline * 30
        )  # Higher HRV = better
        sleep_component = sleep_quality * 0.3

        recovery_score = rhr_component + hrv_component + sleep_component
        recovery_score = max(0, min(100, recovery_score + np.random.normal(0, 5)))

        # Record timestamp (morning measurement)
        recorded_at = datetime.combine(current_date, datetime.min.time()).replace(
            hour=7, minute=0
        )

        # Add all health metrics
        metrics.append(
            {
                "user_id": user_id,
                "metric_type": "RESTING_HEART_RATE",
                "value": round(rhr_value, 1),
                "source": "SYNTHETIC",
                "recorded_at": recorded_at,
            }
        )

        metrics.append(
            {
                "user_id": user_id,
                "metric_type": "HEART_RATE_VARIABILITY_RMSSD",
                "value": round(hrv_value, 1),
                "source": "SYNTHETIC",
                "recorded_at": recorded_at,
            }
        )

        metrics.append(
            {
                "user_id": user_id,
                "metric_type": "HEART_RATE_VARIABILITY_SDNN",
                "value": round(
                    hrv_value * 1.2, 1
                ),  # SDNN typically ~20% higher than RMSSD
                "source": "SYNTHETIC",
                "recorded_at": recorded_at,
            }
        )

        metrics.append(
            {
                "user_id": user_id,
                "metric_type": "SLEEP_DURATION",
                "value": round(sleep_duration, 2),
                "source": "SYNTHETIC",
                "recorded_at": recorded_at,
            }
        )

        metrics.append(
            {
                "user_id": user_id,
                "metric_type": "SLEEP_SCORE",
                "value": round(sleep_quality, 1),
                "source": "SYNTHETIC",
                "recorded_at": recorded_at,
            }
        )

        metrics.append(
            {
                "user_id": user_id,
                "metric_type": "RECOVERY_SCORE",
                "value": round(recovery_score, 1),
                "source": "SYNTHETIC",
                "recorded_at": recorded_at,
            }
        )

        return metrics

    def _update_state(
        self,
        state: Dict,
        daily_meals: List[Dict],
        daily_activities: List[Dict],
        daily_health: List[Dict],
        config: PersonaConfig,
    ) -> None:
        """Update rolling state for next day's calculations."""
        # Calculate nutrition quality
        total_calories = sum(m["calories"] for m in daily_meals)
        total_protein = sum(m["protein"] for m in daily_meals)

        protein_adequacy = min(
            1.0, total_protein / (config.protein_target * config.weight_kg)
        )
        calorie_adequacy = (
            1.0 - abs(total_calories - config.calories_target) / config.calories_target
        )
        nutrition_quality = (protein_adequacy + max(0, calorie_adequacy)) / 2

        state["nutrition_quality_7d"].append(nutrition_quality)
        if len(state["nutrition_quality_7d"]) > 14:
            state["nutrition_quality_7d"] = state["nutrition_quality_7d"][-14:]

        # Calculate activity load
        total_activity = sum(a["duration"] for a in daily_activities)
        high_intensity = sum(
            a["duration"] for a in daily_activities if a.get("intensity") == "HIGH"
        )
        activity_load = total_activity + high_intensity * 0.5

        state["activity_load_7d"].append(activity_load)
        if len(state["activity_load_7d"]) > 14:
            state["activity_load_7d"] = state["activity_load_7d"][-14:]

        # Update cumulative fatigue (decays, increases with high activity)
        fatigue_increase = high_intensity / 60 * 0.1
        fatigue_decay = 0.15
        state["cumulative_fatigue"] = max(
            0, min(1, state["cumulative_fatigue"] + fatigue_increase - fatigue_decay)
        )

        # Update sleep debt
        sleep_metric = next(
            (m for m in daily_health if m["metric_type"] == "SLEEP_DURATION"), None
        )
        if sleep_metric:
            sleep_deficit = config.sleep_baseline - sleep_metric["value"]
            state["sleep_debt"] = max(
                0, min(3, state["sleep_debt"] + sleep_deficit * 0.3)
            )

    def _get_activity_types_for_persona(self, config: PersonaConfig) -> List[str]:
        """Get relevant activity types for a persona."""
        if config.activity_level == "very_active":
            return [
                "RUNNING",
                "CYCLING",
                "SWIMMING",
                "WEIGHT_TRAINING",
                "CROSSFIT",
                "ROWING",
            ]
        elif config.activity_level in ["active", "moderate"]:
            return ["RUNNING", "CYCLING", "YOGA", "WEIGHT_TRAINING", "PILATES"]
        else:
            return ["WALKING", "YOGA", "STRETCHING", "CYCLING"]

    def _generate_user_record(self, config: PersonaConfig, user_id: str) -> Dict:
        """Generate user profile record."""
        return {
            "id": user_id,
            "email": f"{config.name.lower().replace(' ', '.')}@example.com",
            "name": config.name,
            "current_weight": config.weight_kg,
            "goal_weight": config.weight_kg
            * (0.95 if config.activity_level == "sedentary" else 1.0),
            "height": config.height_cm,
            "activity_level": config.activity_level.upper(),
            "goal_calories": config.calories_target,
            "goal_protein": int(config.protein_target * config.weight_kg),
            "goal_carbs": int(config.calories_target * 0.45 / 4),
            "goal_fat": int(config.calories_target * 0.30 / 9),
        }

    def generate_all_users(
        self,
        num_days: int = 180,
        start_date: Optional[date] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate data for all 5 user personas.

        Returns:
            Dictionary mapping user_id to their data dictionaries
        """
        all_data = {}

        for i, persona in enumerate(UserPersona):
            user_id = f"synthetic_user_{i+1}"
            print(f"Generating data for {persona.value} ({user_id})...")
            all_data[user_id] = self.generate_user_data(
                persona=persona,
                user_id=user_id,
                num_days=num_days,
                start_date=start_date,
            )

        return all_data

    def export_to_csv(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        output_dir: str = "synthetic_data",
    ) -> None:
        """Export generated data to CSV files."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Combine all users' data
        all_meals = []
        all_activities = []
        all_health_metrics = []
        all_users = []

        for user_id, user_data in data.items():
            all_meals.append(user_data["meals"])
            all_activities.append(user_data["activities"])
            all_health_metrics.append(user_data["health_metrics"])
            all_users.append(user_data["user"])

        # Save combined DataFrames
        pd.concat(all_meals, ignore_index=True).to_csv(
            f"{output_dir}/meals.csv", index=False
        )
        pd.concat(all_activities, ignore_index=True).to_csv(
            f"{output_dir}/activities.csv", index=False
        )
        pd.concat(all_health_metrics, ignore_index=True).to_csv(
            f"{output_dir}/health_metrics.csv", index=False
        )
        pd.DataFrame(all_users).to_csv(f"{output_dir}/users.csv", index=False)

        print(f"Data exported to {output_dir}/")


# Convenience function for quick generation
def generate_synthetic_dataset(
    num_days: int = 180,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate a complete synthetic dataset for all 5 user personas.

    Args:
        num_days: Number of days of data per user
        seed: Random seed for reproducibility
        output_dir: If provided, export to CSV files

    Returns:
        Dictionary mapping user_id to their data
    """
    generator = SyntheticDataGenerator(seed=seed)
    data = generator.generate_all_users(num_days=num_days)

    if output_dir:
        generator.export_to_csv(data, output_dir)

    return data


if __name__ == "__main__":
    # Generate and export synthetic data
    print("Generating synthetic health data for 5 diverse users...")
    data = generate_synthetic_dataset(
        num_days=180,
        seed=42,
        output_dir="ml-service/synthetic_data",
    )

    # Print summary statistics
    for user_id, user_data in data.items():
        print(f"\n{user_id}:")
        print(f"  Meals: {len(user_data['meals'])} records")
        print(f"  Activities: {len(user_data['activities'])} records")
        print(f"  Health Metrics: {len(user_data['health_metrics'])} records")

        # Show health metric ranges
        health_df = user_data["health_metrics"]
        for metric_type in ["RESTING_HEART_RATE", "HEART_RATE_VARIABILITY_RMSSD"]:
            metric_data = health_df[health_df["metric_type"] == metric_type]["value"]
            print(
                f"  {metric_type}: mean={metric_data.mean():.1f}, std={metric_data.std():.1f}"
            )
