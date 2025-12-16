#!/usr/bin/env python3
"""
Body Stress Level Prediction Experiment

Research-Based Stress Estimation using:
- HRV metrics (RMSSD, SDNN, LF/HF ratio approximation, pNN50)
- Resting Heart Rate and deviation from baseline
- Sleep metrics (duration, quality, efficiency, deep sleep)
- Activity load and recovery balance
- Nutrition timing and quality

Based on scientific research from:
- Whoop Recovery Score methodology
- Oura Ring Readiness Score
- Garmin Stress Score (Firstbeat Analytics)
- Academic stress prediction literature

Stress Score Components (similar to commercial systems):
1. HRV component (40%): Lower HRV = Higher stress
2. RHR component (25%): Elevated RHR = Higher stress
3. Sleep component (20%): Poor sleep = Higher stress
4. Activity Balance (10%): Overtraining/underrecovery = Higher stress
5. Other factors (5%): Nutrition timing, regularity

Categories:
- LOW STRESS (0-33): Green zone, well recovered
- MODERATE STRESS (34-66): Yellow zone, maintaining
- HIGH STRESS (67-100): Red zone, needs recovery
"""

import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# ENHANCED STRESS-AWARE PERSONAS
# =============================================================================


class StressPersona(Enum):
    """12 diverse personas with varying stress profiles."""

    ELITE_ATHLETE = "elite_athlete"
    RECREATIONAL_RUNNER = "recreational_runner"
    SEDENTARY_OFFICE = "sedentary_office"
    STRESSED_EXECUTIVE = "stressed_executive"
    HEALTH_ENTHUSIAST = "health_enthusiast"
    NIGHT_SHIFT_NURSE = "night_shift_nurse"
    COLLEGE_STUDENT_EXAMS = "college_student_exams"
    NEW_PARENT = "new_parent"
    RETIRED_ACTIVE = "retired_active"
    YOGA_PRACTITIONER = "yoga_practitioner"
    CHRONIC_STRESS = "chronic_stress"
    RECOVERING_BURNOUT = "recovering_burnout"


@dataclass
class StressPersonaConfig:
    """Configuration for stress-aware persona."""

    name: str
    age: int

    # Baseline physiological markers
    rhr_baseline: float  # Resting heart rate baseline
    rhr_variability: float  # Day-to-day RHR variance
    hrv_rmssd_baseline: float  # RMSSD baseline (ms)
    hrv_sdnn_baseline: float  # SDNN baseline (ms)
    hrv_variability: float  # HRV day-to-day variance

    # Sleep characteristics
    sleep_duration_baseline: float  # Hours
    sleep_quality_baseline: float  # 0-100
    deep_sleep_pct_baseline: float  # 0-1
    sleep_regularity: float  # 0-1 (1 = very regular)

    # Stress characteristics
    baseline_stress_level: float  # 0-1 chronic stress
    stress_reactivity: float  # How much acute stress affects physiology
    stress_recovery_rate: float  # How quickly stress resolves
    stress_event_frequency: float  # Probability of stress event per day

    # Activity patterns
    activity_level: str
    workout_frequency: float  # Per week
    overtraining_risk: float  # 0-1

    # Lifestyle factors
    nutrition_regularity: float  # 0-1
    late_eating_frequency: float  # 0-1
    alcohol_frequency: float  # Drinks per week
    caffeine_sensitivity: float  # 0-1


# Define 12 stress personas with realistic parameters
STRESS_PERSONAS: Dict[StressPersona, StressPersonaConfig] = {
    StressPersona.ELITE_ATHLETE: StressPersonaConfig(
        name="Elite Athlete",
        age=26,
        rhr_baseline=48.0,
        rhr_variability=2.5,
        hrv_rmssd_baseline=75.0,
        hrv_sdnn_baseline=90.0,
        hrv_variability=10.0,
        sleep_duration_baseline=8.5,
        sleep_quality_baseline=85.0,
        deep_sleep_pct_baseline=0.22,
        sleep_regularity=0.9,
        baseline_stress_level=0.15,
        stress_reactivity=0.3,
        stress_recovery_rate=0.8,
        stress_event_frequency=0.1,
        activity_level="elite",
        workout_frequency=10.0,
        overtraining_risk=0.3,
        nutrition_regularity=0.95,
        late_eating_frequency=0.05,
        alcohol_frequency=0.5,
        caffeine_sensitivity=0.3,
    ),
    StressPersona.RECREATIONAL_RUNNER: StressPersonaConfig(
        name="Recreational Runner",
        age=34,
        rhr_baseline=58.0,
        rhr_variability=3.5,
        hrv_rmssd_baseline=55.0,
        hrv_sdnn_baseline=65.0,
        hrv_variability=12.0,
        sleep_duration_baseline=7.5,
        sleep_quality_baseline=75.0,
        deep_sleep_pct_baseline=0.18,
        sleep_regularity=0.75,
        baseline_stress_level=0.25,
        stress_reactivity=0.4,
        stress_recovery_rate=0.6,
        stress_event_frequency=0.15,
        activity_level="active",
        workout_frequency=4.0,
        overtraining_risk=0.15,
        nutrition_regularity=0.7,
        late_eating_frequency=0.15,
        alcohol_frequency=2.0,
        caffeine_sensitivity=0.5,
    ),
    StressPersona.SEDENTARY_OFFICE: StressPersonaConfig(
        name="Sedentary Office Worker",
        age=42,
        rhr_baseline=72.0,
        rhr_variability=5.0,
        hrv_rmssd_baseline=32.0,
        hrv_sdnn_baseline=40.0,
        hrv_variability=8.0,
        sleep_duration_baseline=6.5,
        sleep_quality_baseline=60.0,
        deep_sleep_pct_baseline=0.12,
        sleep_regularity=0.5,
        baseline_stress_level=0.5,
        stress_reactivity=0.6,
        stress_recovery_rate=0.4,
        stress_event_frequency=0.25,
        activity_level="sedentary",
        workout_frequency=0.5,
        overtraining_risk=0.0,
        nutrition_regularity=0.4,
        late_eating_frequency=0.4,
        alcohol_frequency=4.0,
        caffeine_sensitivity=0.7,
    ),
    StressPersona.STRESSED_EXECUTIVE: StressPersonaConfig(
        name="Stressed Executive",
        age=48,
        rhr_baseline=78.0,
        rhr_variability=6.0,
        hrv_rmssd_baseline=25.0,
        hrv_sdnn_baseline=32.0,
        hrv_variability=7.0,
        sleep_duration_baseline=5.5,
        sleep_quality_baseline=50.0,
        deep_sleep_pct_baseline=0.10,
        sleep_regularity=0.3,
        baseline_stress_level=0.75,
        stress_reactivity=0.8,
        stress_recovery_rate=0.25,
        stress_event_frequency=0.5,
        activity_level="low",
        workout_frequency=1.0,
        overtraining_risk=0.0,
        nutrition_regularity=0.3,
        late_eating_frequency=0.6,
        alcohol_frequency=6.0,
        caffeine_sensitivity=0.8,
    ),
    StressPersona.HEALTH_ENTHUSIAST: StressPersonaConfig(
        name="Health Enthusiast",
        age=31,
        rhr_baseline=60.0,
        rhr_variability=3.0,
        hrv_rmssd_baseline=52.0,
        hrv_sdnn_baseline=62.0,
        hrv_variability=10.0,
        sleep_duration_baseline=7.8,
        sleep_quality_baseline=78.0,
        deep_sleep_pct_baseline=0.19,
        sleep_regularity=0.85,
        baseline_stress_level=0.2,
        stress_reactivity=0.35,
        stress_recovery_rate=0.7,
        stress_event_frequency=0.12,
        activity_level="active",
        workout_frequency=5.0,
        overtraining_risk=0.1,
        nutrition_regularity=0.9,
        late_eating_frequency=0.1,
        alcohol_frequency=1.5,
        caffeine_sensitivity=0.4,
    ),
    StressPersona.NIGHT_SHIFT_NURSE: StressPersonaConfig(
        name="Night Shift Nurse",
        age=35,
        rhr_baseline=68.0,
        rhr_variability=7.0,
        hrv_rmssd_baseline=35.0,
        hrv_sdnn_baseline=42.0,
        hrv_variability=12.0,
        sleep_duration_baseline=6.0,
        sleep_quality_baseline=55.0,
        deep_sleep_pct_baseline=0.11,
        sleep_regularity=0.2,  # Very irregular due to shifts
        baseline_stress_level=0.55,
        stress_reactivity=0.65,
        stress_recovery_rate=0.35,
        stress_event_frequency=0.4,
        activity_level="moderate",
        workout_frequency=2.0,
        overtraining_risk=0.05,
        nutrition_regularity=0.25,
        late_eating_frequency=0.7,
        alcohol_frequency=2.0,
        caffeine_sensitivity=0.6,
    ),
    StressPersona.COLLEGE_STUDENT_EXAMS: StressPersonaConfig(
        name="College Student (Exam Period)",
        age=21,
        rhr_baseline=65.0,
        rhr_variability=5.5,
        hrv_rmssd_baseline=48.0,
        hrv_sdnn_baseline=55.0,
        hrv_variability=15.0,
        sleep_duration_baseline=6.0,
        sleep_quality_baseline=58.0,
        deep_sleep_pct_baseline=0.14,
        sleep_regularity=0.35,
        baseline_stress_level=0.45,
        stress_reactivity=0.7,
        stress_recovery_rate=0.5,
        stress_event_frequency=0.35,
        activity_level="light",
        workout_frequency=2.0,
        overtraining_risk=0.05,
        nutrition_regularity=0.35,
        late_eating_frequency=0.5,
        alcohol_frequency=4.0,
        caffeine_sensitivity=0.5,
    ),
    StressPersona.NEW_PARENT: StressPersonaConfig(
        name="New Parent",
        age=32,
        rhr_baseline=70.0,
        rhr_variability=6.5,
        hrv_rmssd_baseline=38.0,
        hrv_sdnn_baseline=45.0,
        hrv_variability=14.0,
        sleep_duration_baseline=5.5,
        sleep_quality_baseline=45.0,
        deep_sleep_pct_baseline=0.08,
        sleep_regularity=0.15,  # Very fragmented sleep
        baseline_stress_level=0.6,
        stress_reactivity=0.7,
        stress_recovery_rate=0.3,
        stress_event_frequency=0.6,  # Baby wake-ups, etc.
        activity_level="low",
        workout_frequency=1.0,
        overtraining_risk=0.0,
        nutrition_regularity=0.3,
        late_eating_frequency=0.5,
        alcohol_frequency=1.0,
        caffeine_sensitivity=0.7,
    ),
    StressPersona.RETIRED_ACTIVE: StressPersonaConfig(
        name="Retired Active",
        age=67,
        rhr_baseline=62.0,
        rhr_variability=4.0,
        hrv_rmssd_baseline=35.0,
        hrv_sdnn_baseline=45.0,
        hrv_variability=8.0,
        sleep_duration_baseline=7.0,
        sleep_quality_baseline=70.0,
        deep_sleep_pct_baseline=0.12,
        sleep_regularity=0.85,
        baseline_stress_level=0.2,
        stress_reactivity=0.3,
        stress_recovery_rate=0.5,
        stress_event_frequency=0.08,
        activity_level="moderate",
        workout_frequency=4.0,
        overtraining_risk=0.1,
        nutrition_regularity=0.85,
        late_eating_frequency=0.1,
        alcohol_frequency=2.0,
        caffeine_sensitivity=0.4,
    ),
    StressPersona.YOGA_PRACTITIONER: StressPersonaConfig(
        name="Yoga Practitioner",
        age=38,
        rhr_baseline=56.0,
        rhr_variability=2.5,
        hrv_rmssd_baseline=62.0,
        hrv_sdnn_baseline=72.0,
        hrv_variability=8.0,
        sleep_duration_baseline=7.8,
        sleep_quality_baseline=82.0,
        deep_sleep_pct_baseline=0.20,
        sleep_regularity=0.9,
        baseline_stress_level=0.15,
        stress_reactivity=0.25,
        stress_recovery_rate=0.85,
        stress_event_frequency=0.08,
        activity_level="moderate",
        workout_frequency=5.0,
        overtraining_risk=0.02,
        nutrition_regularity=0.9,
        late_eating_frequency=0.05,
        alcohol_frequency=0.5,
        caffeine_sensitivity=0.3,
    ),
    StressPersona.CHRONIC_STRESS: StressPersonaConfig(
        name="Chronic High Stress",
        age=45,
        rhr_baseline=82.0,
        rhr_variability=7.0,
        hrv_rmssd_baseline=22.0,
        hrv_sdnn_baseline=28.0,
        hrv_variability=6.0,
        sleep_duration_baseline=5.0,
        sleep_quality_baseline=40.0,
        deep_sleep_pct_baseline=0.07,
        sleep_regularity=0.25,
        baseline_stress_level=0.85,
        stress_reactivity=0.9,
        stress_recovery_rate=0.15,
        stress_event_frequency=0.7,
        activity_level="sedentary",
        workout_frequency=0.0,
        overtraining_risk=0.0,
        nutrition_regularity=0.2,
        late_eating_frequency=0.7,
        alcohol_frequency=8.0,
        caffeine_sensitivity=0.9,
    ),
    StressPersona.RECOVERING_BURNOUT: StressPersonaConfig(
        name="Recovering from Burnout",
        age=40,
        rhr_baseline=70.0,
        rhr_variability=5.5,
        hrv_rmssd_baseline=30.0,
        hrv_sdnn_baseline=38.0,
        hrv_variability=10.0,
        sleep_duration_baseline=7.5,
        sleep_quality_baseline=55.0,
        deep_sleep_pct_baseline=0.11,
        sleep_regularity=0.6,
        baseline_stress_level=0.55,
        stress_reactivity=0.6,
        stress_recovery_rate=0.4,
        stress_event_frequency=0.2,
        activity_level="light",
        workout_frequency=2.0,
        overtraining_risk=0.05,
        nutrition_regularity=0.6,
        late_eating_frequency=0.25,
        alcohol_frequency=2.0,
        caffeine_sensitivity=0.6,
    ),
}


# =============================================================================
# STRESS DATA GENERATOR
# =============================================================================


class StressDataGenerator:
    """
    Generates synthetic data with explicit stress labels and features.

    Stress Score Calculation (0-100, higher = more stressed):
    Based on research from Whoop, Oura, Garmin, and academic literature.

    Components:
    1. HRV Component (40%): RMSSD deviation from baseline
       - Low RMSSD = High sympathetic activity = High stress
    2. RHR Component (25%): RHR elevation above baseline
       - Elevated RHR = High stress
    3. Sleep Component (20%): Sleep quality, duration, efficiency
       - Poor sleep = High stress
    4. Activity Balance (10%): Training load vs recovery
       - Overtraining = High stress
    5. Contextual (5%): Late eating, alcohol, irregularity
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def generate_persona_data(
        self,
        persona: StressPersona,
        num_days: int = 730,
        start_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Generate comprehensive stress data for a persona."""
        config = STRESS_PERSONAS[persona]

        if start_date is None:
            start_date = date.today() - timedelta(days=num_days)

        records = []

        # Rolling state for temporal dynamics
        state = {
            "cumulative_stress": config.baseline_stress_level,
            "sleep_debt": 0.0,
            "training_load_7d": [],
            "hrv_7d": [],
            "rhr_7d": [],
            "stress_events_active": [],
        }

        for day_idx in range(num_days):
            current_date = start_date + timedelta(days=day_idx)
            day_of_week = current_date.weekday()
            is_weekend = day_of_week >= 5

            # Generate stress events
            stress_event = self._generate_stress_events(config, day_idx, state)

            # Calculate acute stress level for this day
            acute_stress = self._calculate_acute_stress(
                config, state, stress_event, is_weekend
            )

            # Generate physiological responses to stress
            hrv_rmssd, hrv_sdnn, hrv_lf_hf, pnn50 = self._generate_hrv_metrics(
                config, acute_stress, state
            )
            rhr = self._generate_rhr(config, acute_stress, state)

            # Generate sleep metrics (affected by previous day's stress)
            (
                sleep_duration,
                sleep_quality,
                sleep_efficiency,
                deep_sleep_pct,
            ) = self._generate_sleep_metrics(config, state, is_weekend)

            # Generate activity metrics
            (
                activity_load,
                workout_duration,
                workout_intensity,
            ) = self._generate_activity_metrics(config, day_of_week, state)

            # Generate nutrition metrics
            (
                nutrition_score,
                late_eating,
                meal_regularity,
            ) = self._generate_nutrition_metrics(config, is_weekend)

            # Calculate composite stress score (0-100)
            stress_score = self._calculate_stress_score(
                config=config,
                hrv_rmssd=hrv_rmssd,
                hrv_sdnn=hrv_sdnn,
                hrv_lf_hf=hrv_lf_hf,
                rhr=rhr,
                sleep_duration=sleep_duration,
                sleep_quality=sleep_quality,
                deep_sleep_pct=deep_sleep_pct,
                activity_load=activity_load,
                state=state,
                late_eating=late_eating,
            )

            # Recovery score (inverse of stress)
            recovery_score = 100 - stress_score

            # Stress category
            if stress_score <= 33:
                stress_category = "LOW"
            elif stress_score <= 66:
                stress_category = "MODERATE"
            else:
                stress_category = "HIGH"

            # Record data
            records.append(
                {
                    "date": current_date,
                    "persona": persona.value,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    # HRV Metrics
                    "hrv_rmssd": hrv_rmssd,
                    "hrv_sdnn": hrv_sdnn,
                    "hrv_lf_hf_ratio": hrv_lf_hf,
                    "pnn50": pnn50,
                    # Heart Rate
                    "rhr": rhr,
                    "rhr_deviation": rhr - config.rhr_baseline,
                    # Sleep Metrics
                    "sleep_duration": sleep_duration,
                    "sleep_quality": sleep_quality,
                    "sleep_efficiency": sleep_efficiency,
                    "deep_sleep_pct": deep_sleep_pct,
                    "sleep_debt": state["sleep_debt"],
                    # Activity Metrics
                    "activity_load": activity_load,
                    "workout_duration": workout_duration,
                    "workout_intensity": workout_intensity,
                    "training_load_7d": (
                        np.mean(state["training_load_7d"][-7:])
                        if state["training_load_7d"]
                        else 0
                    ),
                    # Nutrition Metrics
                    "nutrition_score": nutrition_score,
                    "late_eating": late_eating,
                    "meal_regularity": meal_regularity,
                    # Stress Context
                    "stress_event": 1 if stress_event else 0,
                    "acute_stress": acute_stress,
                    "cumulative_stress": state["cumulative_stress"],
                    # Target Variables
                    "stress_score": stress_score,
                    "recovery_score": recovery_score,
                    "stress_category": stress_category,
                    # Rolling averages
                    "hrv_rmssd_7d_avg": (
                        np.mean(state["hrv_7d"][-7:]) if state["hrv_7d"] else hrv_rmssd
                    ),
                    "rhr_7d_avg": (
                        np.mean(state["rhr_7d"][-7:]) if state["rhr_7d"] else rhr
                    ),
                }
            )

            # Update state
            self._update_state(
                state,
                config,
                hrv_rmssd,
                rhr,
                sleep_duration,
                activity_load,
                stress_event,
            )

        return pd.DataFrame(records)

    def _generate_stress_events(
        self, config: StressPersonaConfig, day_idx: int, state: Dict
    ) -> bool:
        """Generate random stress events based on persona."""
        # Base probability
        prob = config.stress_event_frequency

        # Decay existing stress events
        state["stress_events_active"] = [
            e for e in state["stress_events_active"] if e["end_day"] > day_idx
        ]

        # Generate new event
        if self.rng.random() < prob:
            # Event duration (1-7 days typically)
            duration = self.rng.integers(1, 8)
            state["stress_events_active"].append(
                {
                    "start_day": day_idx,
                    "end_day": day_idx + duration,
                    "intensity": self.rng.uniform(0.3, 1.0),
                }
            )
            return True

        return len(state["stress_events_active"]) > 0

    def _calculate_acute_stress(
        self,
        config: StressPersonaConfig,
        state: Dict,
        stress_event: bool,
        is_weekend: bool,
    ) -> float:
        """Calculate acute stress level for the day."""
        # Base stress
        acute = config.baseline_stress_level

        # Add active stress events
        for event in state["stress_events_active"]:
            acute += event["intensity"] * config.stress_reactivity * 0.3

        # Sleep debt increases stress
        acute += state["sleep_debt"] * 0.1

        # Cumulative stress effect
        acute += (state["cumulative_stress"] - config.baseline_stress_level) * 0.2

        # Weekend relaxation
        if is_weekend:
            acute *= 0.85

        # Add random variation
        acute += self.rng.normal(0, 0.08)

        return np.clip(acute, 0, 1)

    def _generate_hrv_metrics(
        self, config: StressPersonaConfig, acute_stress: float, state: Dict
    ) -> Tuple[float, float, float, float]:
        """
        Generate HRV metrics responsive to stress.

        Higher stress → Lower HRV (sympathetic dominance)
        Lower stress → Higher HRV (parasympathetic dominance)
        """
        # Stress impact on HRV (inverse relationship)
        stress_impact = 1 - (acute_stress * config.stress_reactivity * 0.4)

        # RMSSD (primary parasympathetic indicator)
        rmssd = config.hrv_rmssd_baseline * stress_impact
        rmssd += self.rng.normal(0, config.hrv_variability)
        rmssd = max(10, min(150, rmssd))

        # SDNN (overall HRV)
        sdnn = config.hrv_sdnn_baseline * stress_impact * 0.95
        sdnn += self.rng.normal(0, config.hrv_variability * 1.1)
        sdnn = max(15, min(180, sdnn))

        # LF/HF ratio (sympathetic/parasympathetic balance)
        # Higher stress → Higher LF/HF ratio
        base_lf_hf = 1.5  # Normal balance ~1.0-2.0
        lf_hf = base_lf_hf + (acute_stress * 2.5)  # Stress increases ratio
        lf_hf += self.rng.normal(0, 0.5)
        lf_hf = max(0.5, min(6.0, lf_hf))

        # pNN50 (parasympathetic indicator)
        # Lower stress → Higher pNN50
        pnn50_base = 25 + (1 - acute_stress) * 25
        pnn50 = pnn50_base + self.rng.normal(0, 8)
        pnn50 = max(0, min(70, pnn50))

        return rmssd, sdnn, lf_hf, pnn50

    def _generate_rhr(
        self, config: StressPersonaConfig, acute_stress: float, state: Dict
    ) -> float:
        """
        Generate RHR responsive to stress.

        Higher stress → Higher RHR
        """
        # Stress elevates RHR
        stress_elevation = acute_stress * config.stress_reactivity * 12

        # Sleep debt elevates RHR
        sleep_debt_effect = state["sleep_debt"] * 2

        # Training load effect (temporary elevation then adaptation)
        training_effect = 0
        if state["training_load_7d"]:
            avg_load = np.mean(state["training_load_7d"][-3:])
            if avg_load > 60:  # Overreaching
                training_effect = (avg_load - 60) * 0.1

        rhr = (
            config.rhr_baseline + stress_elevation + sleep_debt_effect + training_effect
        )
        rhr += self.rng.normal(0, config.rhr_variability)
        rhr = max(40, min(110, rhr))

        return rhr

    def _generate_sleep_metrics(
        self, config: StressPersonaConfig, state: Dict, is_weekend: bool
    ) -> Tuple[float, float, float, float]:
        """Generate sleep metrics affected by stress."""
        # High cumulative stress impairs sleep
        stress_impact = state["cumulative_stress"] * 0.3

        # Duration
        duration = config.sleep_duration_baseline - stress_impact * 1.5
        if is_weekend and config.sleep_regularity < 0.7:
            duration += self.rng.uniform(0.5, 1.5)  # Sleep-in
        duration += self.rng.normal(0, 0.8 * (1 - config.sleep_regularity))
        duration = max(3, min(12, duration))

        # Quality (0-100)
        quality = config.sleep_quality_baseline - stress_impact * 20
        quality += self.rng.normal(0, 10)
        quality = max(10, min(100, quality))

        # Efficiency (time asleep / time in bed)
        efficiency = 0.85 - stress_impact * 0.15
        efficiency += self.rng.normal(0, 0.05)
        efficiency = max(0.5, min(0.98, efficiency))

        # Deep sleep percentage
        deep_pct = config.deep_sleep_pct_baseline - stress_impact * 0.08
        deep_pct += self.rng.normal(0, 0.03)
        deep_pct = max(0.03, min(0.35, deep_pct))

        return duration, quality, efficiency, deep_pct

    def _generate_activity_metrics(
        self, config: StressPersonaConfig, day_of_week: int, state: Dict
    ) -> Tuple[float, float, str]:
        """Generate activity/exercise metrics."""
        # Workout probability
        workout_prob = config.workout_frequency / 7

        if self.rng.random() < workout_prob:
            duration = self.rng.normal(45, 15)
            duration = max(15, min(120, duration))

            # Intensity based on persona
            if config.activity_level == "elite":
                intensity = self.rng.choice(["HIGH", "MODERATE"], p=[0.5, 0.5])
            elif config.activity_level == "active":
                intensity = self.rng.choice(
                    ["HIGH", "MODERATE", "LOW"], p=[0.3, 0.5, 0.2]
                )
            else:
                intensity = self.rng.choice(["MODERATE", "LOW"], p=[0.4, 0.6])

            # Activity load score
            intensity_multiplier = {"HIGH": 2.0, "MODERATE": 1.0, "LOW": 0.5}
            load = duration * intensity_multiplier[intensity]
        else:
            duration = 0
            intensity = "NONE"
            load = self.rng.uniform(10, 30)  # Daily movement

        return load, duration, intensity

    def _generate_nutrition_metrics(
        self, config: StressPersonaConfig, is_weekend: bool
    ) -> Tuple[float, bool, float]:
        """Generate nutrition-related metrics."""
        # Nutrition quality score (0-100)
        score = config.nutrition_regularity * 80 + self.rng.uniform(0, 20)
        if is_weekend:
            score *= self.rng.uniform(0.85, 1.1)
        score = max(20, min(100, score))

        # Late eating
        late_eating = self.rng.random() < config.late_eating_frequency

        # Meal regularity for the day
        regularity = config.nutrition_regularity + self.rng.normal(0, 0.1)
        regularity = max(0, min(1, regularity))

        return score, late_eating, regularity

    def _calculate_stress_score(
        self,
        config: StressPersonaConfig,
        hrv_rmssd: float,
        hrv_sdnn: float,
        hrv_lf_hf: float,
        rhr: float,
        sleep_duration: float,
        sleep_quality: float,
        deep_sleep_pct: float,
        activity_load: float,
        state: Dict,
        late_eating: bool,
    ) -> float:
        """
        Calculate composite stress score (0-100, higher = more stressed).

        Based on Whoop/Oura/Garmin methodologies.
        """
        # 1. HRV Component (40%) - Lower HRV = Higher stress
        hrv_pct_of_baseline = hrv_rmssd / config.hrv_rmssd_baseline
        # Transform: 1.2+ baseline → 0 stress, 0.6 baseline → 100 stress
        hrv_stress = (1.2 - hrv_pct_of_baseline) / 0.6 * 100
        hrv_stress = np.clip(hrv_stress, 0, 100)

        # LF/HF ratio contribution (higher = more stress)
        lf_hf_stress = (hrv_lf_hf - 1.0) / 4.0 * 50  # 1.0 → 0, 5.0 → 50
        lf_hf_stress = np.clip(lf_hf_stress, 0, 50)

        hrv_component = hrv_stress * 0.7 + lf_hf_stress * 0.3

        # 2. RHR Component (25%) - Higher RHR = Higher stress
        rhr_pct_above_baseline = (rhr - config.rhr_baseline) / config.rhr_baseline
        # Transform: 0% above → 0 stress, 20% above → 100 stress
        rhr_stress = rhr_pct_above_baseline / 0.2 * 100
        rhr_stress = np.clip(rhr_stress, 0, 100)

        # 3. Sleep Component (20%) - Poor sleep = Higher stress
        # Sleep duration: <6h high stress, >8h low stress
        duration_stress = (8 - sleep_duration) / 4 * 60
        duration_stress = np.clip(duration_stress, 0, 60)

        # Sleep quality
        quality_stress = (100 - sleep_quality) * 0.4

        sleep_component = duration_stress * 0.6 + quality_stress * 0.4

        # 4. Activity Balance (10%) - Overtraining = Higher stress
        avg_load = (
            np.mean(state["training_load_7d"][-7:]) if state["training_load_7d"] else 30
        )
        # Optimal load ~40-60, too low or too high = stress
        if avg_load < 20:
            activity_stress = 30  # Too sedentary
        elif avg_load > 80:
            activity_stress = min(60, (avg_load - 80) * 2)  # Overtraining
        else:
            activity_stress = 0  # Balanced

        # 5. Contextual (5%)
        contextual_stress = 0
        if late_eating:
            contextual_stress += 30
        contextual_stress += state["sleep_debt"] * 10
        contextual_stress = np.clip(contextual_stress, 0, 100)

        # Weighted combination
        stress_score = (
            hrv_component * 0.40
            + rhr_stress * 0.25
            + sleep_component * 0.20
            + activity_stress * 0.10
            + contextual_stress * 0.05
        )

        # Add small random variation
        stress_score += self.rng.normal(0, 3)

        return np.clip(stress_score, 0, 100)

    def _update_state(
        self,
        state: Dict,
        config: StressPersonaConfig,
        hrv_rmssd: float,
        rhr: float,
        sleep_duration: float,
        activity_load: float,
        stress_event: bool,
    ):
        """Update rolling state for temporal dynamics."""
        # Update HRV history
        state["hrv_7d"].append(hrv_rmssd)
        if len(state["hrv_7d"]) > 14:
            state["hrv_7d"] = state["hrv_7d"][-14:]

        # Update RHR history
        state["rhr_7d"].append(rhr)
        if len(state["rhr_7d"]) > 14:
            state["rhr_7d"] = state["rhr_7d"][-14:]

        # Update training load
        state["training_load_7d"].append(activity_load)
        if len(state["training_load_7d"]) > 14:
            state["training_load_7d"] = state["training_load_7d"][-14:]

        # Update sleep debt
        sleep_need = config.sleep_duration_baseline
        sleep_deficit = sleep_need - sleep_duration
        state["sleep_debt"] = max(0, min(5, state["sleep_debt"] + sleep_deficit * 0.3))

        # Update cumulative stress (with recovery)
        if stress_event:
            state["cumulative_stress"] += 0.1 * config.stress_reactivity
        else:
            # Recovery
            recovery_target = config.baseline_stress_level
            recovery_rate = config.stress_recovery_rate * 0.1
            state["cumulative_stress"] = (
                state["cumulative_stress"] * (1 - recovery_rate)
                + recovery_target * recovery_rate
            )

        state["cumulative_stress"] = np.clip(state["cumulative_stress"], 0, 1)


# =============================================================================
# STRESS PREDICTION MODELS
# =============================================================================


class StressLSTM(nn.Module):
    """LSTM model for stress score prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_type: str = "regression",  # "regression" or "classification"
        num_classes: int = 3,
    ):
        super().__init__()
        self.output_type = output_type

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        if output_type == "regression":
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Output 0-1, scale to 0-100
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)

        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Output
        out = self.fc(context)

        if self.output_type == "regression":
            out = out * 100  # Scale to 0-100

        return out.squeeze()


class StressTransformer(nn.Module):
    """Transformer model for stress prediction."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_type: str = "regression",
        num_classes: int = 3,
    ):
        super().__init__()
        self.output_type = output_type

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if output_type == "regression":
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        out = self.fc(x)

        if self.output_type == "regression":
            out = out * 100

        return out.squeeze()


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================


class StressExperiment:
    """Comprehensive stress prediction experiment."""

    def __init__(
        self,
        output_dir: str = "experiments/stress_prediction",
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)

    def run_full_experiment(
        self,
        num_days: int = 730,
        sequence_length: int = 14,
        epochs: int = 100,
    ):
        """Run the complete stress prediction experiment."""
        print("=" * 80)
        print(" BODY STRESS LEVEL PREDICTION EXPERIMENT")
        print("=" * 80)
        print(f"\nDevice: {self.device}")
        print(f"Days per persona: {num_days}")
        print(f"Sequence length: {sequence_length}")
        print(f"Personas: {len(STRESS_PERSONAS)}")

        results = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "device": str(self.device),
                "num_personas": len(STRESS_PERSONAS),
                "days_per_persona": num_days,
                "sequence_length": sequence_length,
            },
            "persona_stats": {},
            "model_results": {},
            "feature_importance": {},
            "per_persona_results": {},
        }

        # Phase 1: Generate Data
        print("\n" + "=" * 80)
        print(" PHASE 1: STRESS DATA GENERATION")
        print("=" * 80)

        generator = StressDataGenerator(seed=self.seed)
        all_data = []

        for persona in tqdm(STRESS_PERSONAS, desc="Generating personas"):
            df = generator.generate_persona_data(persona, num_days=num_days)
            all_data.append(df)

            # Record persona stats
            results["persona_stats"][persona.value] = {
                "stress_score_mean": float(df["stress_score"].mean()),
                "stress_score_std": float(df["stress_score"].std()),
                "hrv_rmssd_mean": float(df["hrv_rmssd"].mean()),
                "rhr_mean": float(df["rhr"].mean()),
                "sleep_duration_mean": float(df["sleep_duration"].mean()),
                "low_stress_pct": float((df["stress_category"] == "LOW").mean() * 100),
                "moderate_stress_pct": float(
                    (df["stress_category"] == "MODERATE").mean() * 100
                ),
                "high_stress_pct": float(
                    (df["stress_category"] == "HIGH").mean() * 100
                ),
            }

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records: {len(combined_df):,}")
        print(f"Stress distribution:")
        print(combined_df["stress_category"].value_counts(normalize=True).round(3))

        # Prepare sequences
        print("\n" + "=" * 80)
        print(" PHASE 2: FEATURE ENGINEERING & SEQUENCE PREPARATION")
        print("=" * 80)

        (
            X_train,
            X_test,
            y_reg_train,
            y_reg_test,
            y_cls_train,
            y_cls_test,
            feature_names,
        ) = self._prepare_sequences(combined_df, sequence_length)

        print(f"Training sequences: {len(X_train)}")
        print(f"Test sequences: {len(X_test)}")
        print(f"Features: {len(feature_names)}")

        # Phase 3: Train Traditional ML Models
        print("\n" + "=" * 80)
        print(" PHASE 3: TRADITIONAL ML MODELS")
        print("=" * 80)

        # Flatten sequences for traditional ML
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Random Forest Regression
        print("\n--- Random Forest Regressor ---")
        rf_reg = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=self.seed, n_jobs=-1
        )
        rf_reg.fit(X_train_flat, y_reg_train)
        rf_pred = rf_reg.predict(X_test_flat)

        rf_reg_results = {
            "mae": float(mean_absolute_error(y_reg_test, rf_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, rf_pred))),
            "r2": float(r2_score(y_reg_test, rf_pred)),
        }
        print(f"  MAE: {rf_reg_results['mae']:.2f}")
        print(f"  RMSE: {rf_reg_results['rmse']:.2f}")
        print(f"  R²: {rf_reg_results['r2']:.3f}")
        results["model_results"]["random_forest_regression"] = rf_reg_results

        # Random Forest Classification
        print("\n--- Random Forest Classifier ---")
        rf_cls = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=self.seed, n_jobs=-1
        )
        rf_cls.fit(X_train_flat, y_cls_train)
        rf_cls_pred = rf_cls.predict(X_test_flat)

        rf_cls_results = {
            "accuracy": float(accuracy_score(y_cls_test, rf_cls_pred)),
            "f1_macro": float(f1_score(y_cls_test, rf_cls_pred, average="macro")),
            "f1_weighted": float(f1_score(y_cls_test, rf_cls_pred, average="weighted")),
        }
        print(f"  Accuracy: {rf_cls_results['accuracy']:.3f}")
        print(f"  F1 (macro): {rf_cls_results['f1_macro']:.3f}")
        print(
            classification_report(
                y_cls_test, rf_cls_pred, target_names=["LOW", "MODERATE", "HIGH"]
            )
        )
        results["model_results"]["random_forest_classification"] = rf_cls_results

        # Gradient Boosting
        print("\n--- Gradient Boosting Regressor ---")
        gb_reg = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=self.seed
        )
        gb_reg.fit(X_train_flat, y_reg_train)
        gb_pred = gb_reg.predict(X_test_flat)

        gb_reg_results = {
            "mae": float(mean_absolute_error(y_reg_test, gb_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, gb_pred))),
            "r2": float(r2_score(y_reg_test, gb_pred)),
        }
        print(f"  MAE: {gb_reg_results['mae']:.2f}")
        print(f"  RMSE: {gb_reg_results['rmse']:.2f}")
        print(f"  R²: {gb_reg_results['r2']:.3f}")
        results["model_results"]["gradient_boosting_regression"] = gb_reg_results

        # Feature Importance Analysis
        print("\n--- Feature Importance (Random Forest) ---")
        # Get importance for last timestep features (most relevant)
        importance_per_timestep = rf_reg.feature_importances_.reshape(
            sequence_length, -1
        )
        avg_importance = importance_per_timestep.mean(axis=0)

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": avg_importance}
        ).sort_values("importance", ascending=False)

        print("\nTop 15 Most Important Features:")
        for _, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        results["feature_importance"] = {
            row["feature"]: float(row["importance"])
            for _, row in importance_df.iterrows()
        }

        # Phase 4: Train Deep Learning Models
        print("\n" + "=" * 80)
        print(" PHASE 4: DEEP LEARNING MODELS")
        print("=" * 80)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_reg_train_tensor = torch.FloatTensor(y_reg_train).to(self.device)
        y_reg_test_tensor = torch.FloatTensor(y_reg_test).to(self.device)
        y_cls_train_tensor = torch.LongTensor(y_cls_train).to(self.device)
        y_cls_test_tensor = torch.LongTensor(y_cls_test).to(self.device)

        train_dataset = TensorDataset(
            X_train_tensor, y_reg_train_tensor, y_cls_train_tensor
        )
        test_dataset = TensorDataset(
            X_test_tensor, y_reg_test_tensor, y_cls_test_tensor
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        input_dim = X_train.shape[2]

        # LSTM Regression
        print("\n--- BiLSTM + Attention (Regression) ---")
        lstm_reg = StressLSTM(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            output_type="regression",
        ).to(self.device)

        lstm_reg_results = self._train_model(
            lstm_reg, train_loader, test_loader, epochs=epochs, task="regression"
        )
        print(f"  MAE: {lstm_reg_results['mae']:.2f}")
        print(f"  RMSE: {lstm_reg_results['rmse']:.2f}")
        print(f"  R²: {lstm_reg_results['r2']:.3f}")
        results["model_results"]["lstm_regression"] = lstm_reg_results

        # LSTM Classification
        print("\n--- BiLSTM + Attention (Classification) ---")
        lstm_cls = StressLSTM(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            output_type="classification",
            num_classes=3,
        ).to(self.device)

        lstm_cls_results = self._train_model(
            lstm_cls, train_loader, test_loader, epochs=epochs, task="classification"
        )
        print(f"  Accuracy: {lstm_cls_results['accuracy']:.3f}")
        print(f"  F1 (macro): {lstm_cls_results['f1_macro']:.3f}")
        results["model_results"]["lstm_classification"] = lstm_cls_results

        # Transformer Regression
        print("\n--- Transformer (Regression) ---")
        transformer_reg = StressTransformer(
            input_dim=input_dim,
            d_model=64,
            nhead=4,
            num_layers=3,
            dropout=0.1,
            output_type="regression",
        ).to(self.device)

        transformer_reg_results = self._train_model(
            transformer_reg, train_loader, test_loader, epochs=epochs, task="regression"
        )
        print(f"  MAE: {transformer_reg_results['mae']:.2f}")
        print(f"  RMSE: {transformer_reg_results['rmse']:.2f}")
        print(f"  R²: {transformer_reg_results['r2']:.3f}")
        results["model_results"]["transformer_regression"] = transformer_reg_results

        # Transformer Classification
        print("\n--- Transformer (Classification) ---")
        transformer_cls = StressTransformer(
            input_dim=input_dim,
            d_model=64,
            nhead=4,
            num_layers=3,
            dropout=0.1,
            output_type="classification",
            num_classes=3,
        ).to(self.device)

        transformer_cls_results = self._train_model(
            transformer_cls,
            train_loader,
            test_loader,
            epochs=epochs,
            task="classification",
        )
        print(f"  Accuracy: {transformer_cls_results['accuracy']:.3f}")
        print(f"  F1 (macro): {transformer_cls_results['f1_macro']:.3f}")
        results["model_results"]["transformer_classification"] = transformer_cls_results

        # Phase 5: Per-Persona Analysis
        print("\n" + "=" * 80)
        print(" PHASE 5: PER-PERSONA ANALYSIS")
        print("=" * 80)

        for persona in STRESS_PERSONAS:
            persona_df = combined_df[combined_df["persona"] == persona.value]
            X_p, _, y_reg_p, _, _, _, _ = self._prepare_sequences(
                persona_df, sequence_length, test_size=0.0
            )

            if len(X_p) > 0:
                X_p_flat = X_p.reshape(X_p.shape[0], -1)
                pred = rf_reg.predict(X_p_flat)

                results["per_persona_results"][persona.value] = {
                    "mae": float(mean_absolute_error(y_reg_p, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_reg_p, pred))),
                    "r2": float(r2_score(y_reg_p, pred)),
                    "actual_mean": float(y_reg_p.mean()),
                    "predicted_mean": float(pred.mean()),
                }

                print(f"\n{persona.value}:")
                print(f"  Actual stress: {y_reg_p.mean():.1f} ± {y_reg_p.std():.1f}")
                print(f"  Predicted: {pred.mean():.1f} ± {pred.std():.1f}")
                print(
                    f"  MAE: {results['per_persona_results'][persona.value]['mae']:.2f}"
                )

        # Save results
        results["metadata"]["end_time"] = datetime.now().isoformat()

        results_path = self.output_dir / "stress_experiment_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 80)
        print(" EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {results_path}")

        # Summary
        print("\n" + "=" * 80)
        print(" SUMMARY: BEST MODELS")
        print("=" * 80)

        print("\n### Regression (Stress Score Prediction) ###")
        reg_models = {
            k: v for k, v in results["model_results"].items() if "regression" in k
        }
        best_reg = min(reg_models.items(), key=lambda x: x[1]["mae"])
        print(
            f"Best: {best_reg[0]} (MAE: {best_reg[1]['mae']:.2f}, R²: {best_reg[1]['r2']:.3f})"
        )

        print("\n### Classification (Stress Category) ###")
        cls_models = {
            k: v for k, v in results["model_results"].items() if "classification" in k
        }
        best_cls = max(cls_models.items(), key=lambda x: x[1]["accuracy"])
        print(
            f"Best: {best_cls[0]} (Accuracy: {best_cls[1]['accuracy']:.3f}, F1: {best_cls[1]['f1_macro']:.3f})"
        )

        print("\n### Top 5 Stress Predictors ###")
        for i, (feat, imp) in enumerate(
            list(results["feature_importance"].items())[:5], 1
        ):
            print(f"  {i}. {feat}: {imp:.4f}")

        return results

    def _prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        test_size: float = 0.15,
    ) -> Tuple[np.ndarray, ...]:
        """Prepare sequences for training."""

        # Feature columns
        feature_cols = [
            # HRV features
            "hrv_rmssd",
            "hrv_sdnn",
            "hrv_lf_hf_ratio",
            "pnn50",
            "hrv_rmssd_7d_avg",
            # RHR features
            "rhr",
            "rhr_deviation",
            "rhr_7d_avg",
            # Sleep features
            "sleep_duration",
            "sleep_quality",
            "sleep_efficiency",
            "deep_sleep_pct",
            "sleep_debt",
            # Activity features
            "activity_load",
            "workout_duration",
            "training_load_7d",
            # Nutrition features
            "nutrition_score",
            "late_eating",
            "meal_regularity",
            # Context features
            "is_weekend",
            "stress_event",
            "cumulative_stress",
        ]

        # Create sequences per persona
        all_sequences = []
        all_targets_reg = []
        all_targets_cls = []

        for persona in df["persona"].unique():
            persona_df = (
                df[df["persona"] == persona].sort_values("date").reset_index(drop=True)
            )

            X = persona_df[feature_cols].values
            y_reg = persona_df["stress_score"].values
            y_cls = (
                persona_df["stress_category"]
                .map({"LOW": 0, "MODERATE": 1, "HIGH": 2})
                .values
            )

            # Normalize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Create sequences
            for i in range(len(X) - sequence_length):
                seq = X[i : i + sequence_length]
                target_reg = y_reg[i + sequence_length]  # Predict next day
                target_cls = y_cls[i + sequence_length]

                all_sequences.append(seq)
                all_targets_reg.append(target_reg)
                all_targets_cls.append(target_cls)

        X = np.array(all_sequences)
        y_reg = np.array(all_targets_reg)
        y_cls = np.array(all_targets_cls)

        if test_size > 0:
            (
                X_train,
                X_test,
                y_reg_train,
                y_reg_test,
                y_cls_train,
                y_cls_test,
            ) = train_test_split(
                X, y_reg, y_cls, test_size=test_size, random_state=self.seed
            )
            return (
                X_train,
                X_test,
                y_reg_train,
                y_reg_test,
                y_cls_train,
                y_cls_test,
                feature_cols,
            )
        else:
            return (
                X,
                np.array([]),
                y_reg,
                np.array([]),
                y_cls,
                np.array([]),
                feature_cols,
            )

    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        task: str = "regression",
    ) -> Dict[str, Any]:
        """Train a PyTorch model."""
        if task == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for batch in train_loader:
                X_batch = batch[0]
                if task == "regression":
                    y_batch = batch[1]
                else:
                    y_batch = batch[2]

                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    X_batch = batch[0]
                    if task == "regression":
                        y_batch = batch[1]
                    else:
                        y_batch = batch[2]

                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()

            val_loss /= len(test_loader)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

        # Final evaluation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                X_batch = batch[0]
                if task == "regression":
                    y_batch = batch[1]
                else:
                    y_batch = batch[2]

                output = model(X_batch)

                if task == "classification":
                    output = output.argmax(dim=1)

                all_preds.extend(output.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        preds = np.array(all_preds)
        targets = np.array(all_targets)

        if task == "regression":
            return {
                "mae": float(mean_absolute_error(targets, preds)),
                "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
                "r2": float(r2_score(targets, preds)),
            }
        else:
            return {
                "accuracy": float(accuracy_score(targets, preds)),
                "f1_macro": float(f1_score(targets, preds, average="macro")),
                "f1_weighted": float(f1_score(targets, preds, average="weighted")),
            }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stress Prediction Experiment")
    parser.add_argument("--days", type=int, default=730, help="Days per persona")
    parser.add_argument("--seq-length", type=int, default=14, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")

    args = parser.parse_args()

    if args.quick:
        args.days = 180
        args.epochs = 20

    experiment = StressExperiment(
        output_dir="experiments/stress_prediction",
        seed=42,
    )

    results = experiment.run_full_experiment(
        num_days=args.days,
        sequence_length=args.seq_length,
        epochs=args.epochs,
    )
