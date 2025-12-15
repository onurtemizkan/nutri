# Acute Nutrition-HRV Research & Proposed Beta Testing Models

## Executive Summary

This document presents research findings on how specific nutritional factors acutely affect Heart Rate Variability (HRV) within 1-5 hours of consumption, and proposes predictive models for validation during beta testing with real user data.

**Key Finding**: Nutrition-HRV relationships show consistent, measurable acute effects that can be predicted with moderate-to-high confidence, particularly for alcohol, meal timing, glycemic load, and hydration status.

---

## Part 1: Research Findings

### 1.1 High-Confidence Acute Effects

These factors show consistent, well-documented effects across multiple studies:

#### Alcohol
| Metric | Effect | Timing | Recovery |
|--------|--------|--------|----------|
| RMSSD | ↓ 22.7% (2+ drinks) | During consumption | 4-5 days full recovery |
| HF Power | ↓ 30-40% | 0-3h post | 24-48h partial |
| LF/HF Ratio | ↑ 25-35% | During sleep | 2-3 days |
| RHR | ↑ 5-15 bpm | Overnight | 24-48h |

**Dose-Response**: Linear up to ~4 drinks, then saturates. Each additional drink adds ~5% RMSSD reduction.

**Mechanism**: Alcohol suppresses vagal tone, increases sympathetic activity, disrupts sleep architecture (REM/deep sleep), and causes dehydration.

#### Late Eating (Within 3h of Sleep)
| Metric | Effect | Timing |
|--------|--------|--------|
| Overnight RMSSD | ↓ 8-15% | All night |
| Sleep HRV Recovery | ↓ 20-30% | First 3h of sleep |
| Morning Readiness | ↓ 10-15% | Upon waking |

**Mechanism**: Elevated metabolic activity during sleep, circadian phase disruption, insulin/glucose response interference with parasympathetic recovery.

#### High Glycemic Index Carbohydrates
| Metric | Effect | Onset | Peak | Duration |
|--------|--------|-------|------|----------|
| LF/HF Ratio | ↑ 40% | 40-60 min | 60-120 min | 2-3h |
| RMSSD | ↓ 10-20% | 30-60 min | 60-90 min | 2-3h |
| HF Power | ↓ 15-25% | 45-90 min | 90-120 min | 2-3h |

**Dose-Response**: Effects scale with glycemic load. High-GI (>70) meals show strongest effects.

**Mechanism**: Rapid insulin spike → sympathetic activation → vagal withdrawal.

#### Dehydration
| Hydration Loss | RMSSD Effect | Onset |
|----------------|--------------|-------|
| -1% body mass | ↓ 5-10% | 1-2h |
| -2% body mass | ↓ 15-25% | 2-4h |
| -3% body mass | ↓ 30-40% | 3-6h |

**Mechanism**: Reduced blood volume → increased heart rate → reduced vagal tone → sympathetic compensation.

#### Large Meals (>1000 kcal)
| Metric | Effect | Timing |
|--------|--------|--------|
| RMSSD | ↓ 15-25% | 30 min - 3h post-meal |
| LF/HF | ↑ 20-40% | 45 min - 2h post-meal |
| RHR | ↑ 5-10 bpm | 15 min - 2h post-meal |

**Mechanism**: "Postprandial dip" - blood flow diverted to digestion, thermic effect of food, insulin response.

---

### 1.2 Moderate-Confidence Acute Effects

These factors show consistent effects but with higher individual variability:

#### Caffeine
| CYP1A2 Genotype | Effect | Population |
|-----------------|--------|------------|
| Fast metabolizers (AA) | Minimal/positive | ~40% |
| Slow metabolizers (AC/CC) | ↓ RMSSD 10-20% | ~60% |

**Key Insight**: Caffeine effects are highly genotype-dependent. Without genetic data, predict based on:
- Habitual consumption (tolerance develops)
- Time of day (circadian interaction)
- Dose (>400mg shows consistent negative effects)

**Timing**: Onset 30-45 min, peak 45-60 min, half-life 3-7 hours (genotype-dependent).

#### Energy Drinks (Caffeine + Taurine + Sugar)
| Metric | Effect | Timing |
|--------|--------|--------|
| RMSSD | ↓ 15-25% | 30-120 min |
| LF/HF | ↑ 30-50% | 45-90 min |
| RHR | ↑ 8-15 bpm | 30-90 min |

**Note**: Stronger than caffeine alone due to synergistic effects.

#### Capsaicin (Spicy Food)
| Sex | Effect | Timing |
|-----|--------|--------|
| Male | ↓ HRV 10-15% | 30-90 min |
| Female | Minimal | - |

**Mechanism**: Sex-hormone interaction with TRPV1 receptor activation.

---

### 1.3 Protective/Positive Acute Effects

These factors may improve HRV acutely:

| Factor | Effect | Timing | Confidence |
|--------|--------|--------|------------|
| Omega-3 rich meal | ↑ RMSSD 5-10% | 2-4h | Moderate |
| Adequate hydration | ↑ RMSSD 10-15% | 1-2h | High |
| Light meal (<400 kcal) | Neutral to ↑ 5% | 1-2h | Moderate |
| Low-GI carbohydrates | Neutral | - | High |
| Magnesium-rich foods | ↑ RMSSD 5-8% | 2-4h | Low-Moderate |

---

### 1.4 Individual Variation Factors

These factors modify the magnitude of acute effects:

1. **Age**: Older adults show larger negative effects (↓ baseline HRV)
2. **Fitness Level**: Higher fitness = faster recovery
3. **Habitual Consumption**: Tolerance develops (especially caffeine)
4. **Circadian Phase**: Morning HRV more sensitive than evening
5. **Sleep Debt**: Amplifies negative effects
6. **Chronic Stress**: Reduces adaptive capacity
7. **Genetics**: CYP1A2 (caffeine), ADH1B (alcohol)

---

## Part 2: Proposed Beta Testing Models

### 2.1 Model Architecture Overview

We propose a **three-tier model system** for beta testing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 1: RULE-BASED BASELINE                  │
│  Simple heuristics for immediate feedback (no training needed)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 TIER 2: PERSONALIZED ML MODEL                   │
│   User-specific model trained on 30+ days of their data         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  TIER 3: POPULATION ENSEMBLE                    │
│    Transfer learning from aggregate anonymized user data        │
└─────────────────────────────────────────────────────────────────┘
```

---

### 2.2 Tier 1: Rule-Based Baseline Model

**Purpose**: Provide immediate predictions without training data

**Implementation**:
```python
class NutritionHRVRuleEngine:
    """
    Rule-based HRV prediction from nutrition events.
    Uses research-backed effect sizes and timing.
    """

    EFFECT_RULES = {
        'alcohol': {
            'threshold': 1,  # drinks
            'rmssd_effect_per_unit': -0.05,  # -5% per drink
            'max_effect': -0.35,  # cap at 35% reduction
            'onset_hours': 0.5,
            'peak_hours': 3.0,
            'recovery_days': 4,
        },
        'late_eating': {
            'threshold_hours_before_sleep': 3,
            'rmssd_effect': -0.12,  # -12%
            'applies_to': 'overnight_hrv',
        },
        'high_gi_meal': {
            'threshold_gi': 70,
            'rmssd_effect': -0.15,
            'onset_hours': 0.5,
            'peak_hours': 1.5,
            'duration_hours': 3,
        },
        'large_meal': {
            'threshold_kcal': 1000,
            'rmssd_effect': -0.20,
            'onset_hours': 0.5,
            'peak_hours': 1.5,
            'duration_hours': 3,
        },
        'caffeine_high': {
            'threshold_mg': 400,
            'rmssd_effect': -0.10,  # Conservative for unknown genotype
            'onset_hours': 0.5,
            'peak_hours': 1.0,
            'duration_hours': 6,
        },
        'dehydration': {
            'threshold_deficit_pct': 1,
            'rmssd_effect_per_pct': -0.12,
            'max_effect': -0.40,
        },
    }

    def predict_hrv_impact(
        self,
        baseline_rmssd: float,
        nutrition_events: List[NutritionEvent],
        prediction_time: datetime,
    ) -> HRVPrediction:
        """
        Predict HRV at a future time given nutrition events.

        Returns:
            HRVPrediction with expected_rmssd, confidence_interval,
            contributing_factors
        """
        total_effect = 0.0
        factors = []

        for event in nutrition_events:
            effect = self._calculate_event_effect(event, prediction_time)
            if effect != 0:
                total_effect += effect
                factors.append({
                    'event': event.type,
                    'effect': effect,
                    'confidence': self._get_confidence(event.type)
                })

        # Cap total negative effect at 50%
        total_effect = max(total_effect, -0.50)

        predicted_rmssd = baseline_rmssd * (1 + total_effect)

        return HRVPrediction(
            expected_rmssd=predicted_rmssd,
            confidence_low=predicted_rmssd * 0.85,
            confidence_high=predicted_rmssd * 1.15,
            contributing_factors=factors,
            model_tier='rule_based'
        )
```

**Validation Metrics for Beta**:
- Mean Absolute Percentage Error (MAPE) < 20%
- Directional Accuracy > 70% (predicts increase/decrease correctly)
- Calibrated Confidence Intervals (70% of actuals within CI)

---

### 2.3 Tier 2: Personalized ML Model

**Purpose**: Learn individual response patterns from user's own data

**Minimum Data Requirements**:
- 30 days of HRV data (ideally continuous wearable)
- 30+ logged meals with timing
- At least 5 instances of each target behavior (alcohol, late eating, etc.)

**Architecture**: Lightweight Gradient Boosting with Temporal Features

```python
class PersonalizedNutritionHRVModel:
    """
    User-specific model that learns individual nutrition-HRV patterns.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.model = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            min_child_samples=5,  # Small for individual data
        )
        self.feature_scaler = StandardScaler()
        self.baseline_model = NutritionHRVRuleEngine()

    def extract_features(
        self,
        nutrition_log: pd.DataFrame,
        hrv_data: pd.DataFrame,
        target_time: datetime,
    ) -> np.ndarray:
        """
        Extract features for HRV prediction at target_time.
        """
        features = {}

        # 1. Nutrition features (past 24h)
        recent_meals = nutrition_log[
            nutrition_log['consumed_at'] > target_time - timedelta(hours=24)
        ]

        features['total_calories_24h'] = recent_meals['calories'].sum()
        features['total_alcohol_24h'] = recent_meals['alcohol_drinks'].sum()
        features['avg_glycemic_index'] = recent_meals['glycemic_index'].mean()
        features['last_meal_hours_ago'] = self._hours_since_last_meal(
            recent_meals, target_time
        )
        features['caffeine_mg_12h'] = self._caffeine_last_n_hours(
            recent_meals, target_time, 12
        )

        # 2. Temporal features
        features['hour_of_day'] = target_time.hour
        features['is_weekend'] = target_time.weekday() >= 5
        features['days_since_last_alcohol'] = self._days_since_alcohol(
            nutrition_log, target_time
        )

        # 3. User baseline features
        features['baseline_rmssd_7d'] = hrv_data['rmssd'].rolling(7).mean().iloc[-1]
        features['rmssd_trend_7d'] = self._calculate_trend(
            hrv_data['rmssd'].tail(7)
        )

        # 4. Interaction features
        features['late_eating_flag'] = int(
            features['last_meal_hours_ago'] < 3 and target_time.hour >= 22
        )
        features['alcohol_x_sleep_debt'] = (
            features['total_alcohol_24h'] *
            features.get('sleep_debt_hours', 0)
        )

        return self._to_feature_vector(features)

    def train(
        self,
        nutrition_log: pd.DataFrame,
        hrv_data: pd.DataFrame,
        min_samples: int = 30,
    ) -> Dict[str, float]:
        """
        Train personalized model on user's data.

        Returns training metrics.
        """
        if len(hrv_data) < min_samples:
            raise InsufficientDataError(
                f"Need {min_samples} days, have {len(hrv_data)}"
            )

        X, y = self._prepare_training_data(nutrition_log, hrv_data)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Temporal split
        )

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[early_stopping(10)],
        )

        # Calculate validation metrics
        y_pred = self.model.predict(X_val_scaled)

        return {
            'mae': mean_absolute_error(y_val, y_pred),
            'mape': mean_absolute_percentage_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred),
            'directional_accuracy': self._directional_accuracy(y_val, y_pred),
        }

    def predict(
        self,
        nutrition_log: pd.DataFrame,
        hrv_data: pd.DataFrame,
        target_time: datetime,
    ) -> HRVPrediction:
        """
        Predict HRV, falling back to rule-based if insufficient data.
        """
        if not self._is_trained():
            return self.baseline_model.predict_hrv_impact(
                baseline_rmssd=hrv_data['rmssd'].mean(),
                nutrition_events=self._extract_events(nutrition_log),
                prediction_time=target_time,
            )

        features = self.extract_features(nutrition_log, hrv_data, target_time)
        features_scaled = self.feature_scaler.transform([features])

        predicted_rmssd = self.model.predict(features_scaled)[0]

        # Estimate uncertainty from model
        confidence_interval = self._estimate_uncertainty(features_scaled)

        return HRVPrediction(
            expected_rmssd=predicted_rmssd,
            confidence_low=confidence_interval[0],
            confidence_high=confidence_interval[1],
            contributing_factors=self._get_feature_contributions(features),
            model_tier='personalized',
        )
```

**Key Features to Track**:
1. `total_alcohol_24h` - Number of alcoholic drinks
2. `last_meal_hours_ago` - Hours since last meal
3. `late_eating_flag` - Binary: meal within 3h of sleep
4. `avg_glycemic_index` - Weighted GI of recent meals
5. `total_calories_24h` - Caloric load
6. `caffeine_mg_12h` - Caffeine in last 12 hours
7. `hydration_score` - Estimated hydration status
8. `baseline_rmssd_7d` - Personal 7-day baseline
9. `hour_of_day` - Circadian effects
10. `days_since_last_alcohol` - Recovery tracking

---

### 2.4 Tier 3: Population Ensemble Model

**Purpose**: Leverage aggregate patterns from all beta users (privacy-preserving)

**Architecture**: Federated Learning or Differential Privacy Aggregation

```python
class PopulationNutritionHRVModel:
    """
    Aggregate model trained on anonymized population data.
    Uses federated learning to preserve privacy.
    """

    def __init__(self):
        # Pre-trained on synthetic data, fine-tuned on population
        self.base_model = self._load_pretrained_model()
        self.population_effects = {}

    def update_from_user(
        self,
        user_model: PersonalizedNutritionHRVModel,
        differential_privacy_epsilon: float = 1.0,
    ):
        """
        Update population model with user's learned effects.
        Applies differential privacy to protect individual data.
        """
        # Extract effect sizes from user model
        user_effects = user_model.get_learned_effects()

        # Add noise for differential privacy
        noised_effects = self._apply_dp_noise(
            user_effects, epsilon=differential_privacy_epsilon
        )

        # Weighted update to population model
        self._update_population_effects(noised_effects)

    def get_population_priors(self) -> Dict[str, EffectSize]:
        """
        Get population-level effect estimates for new users.

        Returns effect sizes with confidence intervals.
        """
        return {
            'alcohol_per_drink': EffectSize(
                mean=-0.05, std=0.02, n_observations=self.n_users
            ),
            'late_eating': EffectSize(
                mean=-0.12, std=0.05, n_observations=self.n_users
            ),
            'high_gi_meal': EffectSize(
                mean=-0.15, std=0.06, n_observations=self.n_users
            ),
            # ... etc
        }

    def predict_for_new_user(
        self,
        baseline_rmssd: float,
        nutrition_events: List[NutritionEvent],
        user_profile: Optional[UserProfile] = None,
    ) -> HRVPrediction:
        """
        Predict HRV for new user using population priors.

        Adjusts based on user profile (age, fitness) if available.
        """
        priors = self.get_population_priors()

        # Adjust priors based on user profile
        if user_profile:
            adjusted_priors = self._adjust_for_profile(priors, user_profile)
        else:
            adjusted_priors = priors

        # Calculate expected effect
        total_effect = 0.0
        for event in nutrition_events:
            effect_size = adjusted_priors.get(event.type)
            if effect_size:
                total_effect += effect_size.mean

        predicted_rmssd = baseline_rmssd * (1 + total_effect)

        # Wider confidence intervals for population model
        uncertainty = sum(
            adjusted_priors[e.type].std
            for e in nutrition_events
            if e.type in adjusted_priors
        )

        return HRVPrediction(
            expected_rmssd=predicted_rmssd,
            confidence_low=predicted_rmssd * (1 - uncertainty),
            confidence_high=predicted_rmssd * (1 + uncertainty),
            model_tier='population',
        )
```

---

## Part 3: Data Collection Requirements

### 3.1 Required Data Points

For accurate acute nutrition-HRV modeling, collect:

#### Nutrition Data (per meal/snack)
| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `consumed_at` | datetime | Yes | Exact time critical |
| `calories` | int | Yes | Total kcal |
| `carbohydrates` | float | Yes | Grams |
| `glycemic_index` | int | Ideal | Estimate if not exact |
| `alcohol_drinks` | float | Yes | Standard drinks |
| `caffeine_mg` | int | Ideal | From coffee/tea/energy drinks |
| `meal_size` | enum | Yes | small/medium/large |
| `water_ml` | int | Ideal | Hydration tracking |

#### HRV Data (continuous preferred)
| Field | Type | Frequency | Source |
|-------|------|-----------|--------|
| `rmssd` | float | Every 5 min (ideal) | Wearable |
| `sdnn` | float | Hourly+ | Wearable |
| `lf_hf_ratio` | float | Hourly+ | Wearable |
| `rhr` | int | Continuous | Wearable |
| `measurement_time` | datetime | With each reading | Wearable |

#### Context Data
| Field | Type | When | Notes |
|-------|------|------|-------|
| `sleep_time` | datetime | Daily | For late eating calc |
| `wake_time` | datetime | Daily | Recovery window |
| `activity_intensity` | enum | Per session | Exercise affects HRV |
| `perceived_stress` | 1-10 | Optional daily | Validation signal |

### 3.2 Data Quality Requirements

**For Tier 2 Model Training**:
- Minimum 30 days of data
- HRV readings at least 4x daily (morning, pre-meal, post-meal, before sleep)
- At least 80% meal logging compliance
- At least 5 "events" of each type (alcohol occasions, late meals, etc.)

**Ideal Data Collection**:
- Continuous HRV from wearable (Apple Watch, Oura, Whoop, Garmin)
- Photo-based meal logging with AI calorie estimation
- Automatic caffeine tracking from beverage logging
- Sleep detection from wearable

---

## Part 4: Beta Testing Protocol

### 4.1 Phase 1: Rule-Based Validation (Weeks 1-4)

**Objective**: Validate rule-based predictions against actual HRV

**Protocol**:
1. Deploy Tier 1 rule-based model
2. Users log meals and receive predictions
3. Compare predictions to actual HRV readings
4. Calculate MAPE, directional accuracy

**Success Criteria**:
- MAPE < 25%
- Directional Accuracy > 65%
- User engagement > 70%

### 4.2 Phase 2: Personalized Model Training (Weeks 5-8)

**Objective**: Train and validate individual models

**Protocol**:
1. For users with 30+ days data, train Tier 2 model
2. A/B test: Rule-based vs Personalized predictions
3. Measure improvement in accuracy

**Success Criteria**:
- Personalized model reduces MAPE by 20%+ vs rule-based
- Feature importance aligns with research
- Model explains at least 30% of HRV variance (R² > 0.3)

### 4.3 Phase 3: Population Learning (Weeks 9-12)

**Objective**: Build and validate population model

**Protocol**:
1. Aggregate anonymized effect sizes across users
2. Validate population priors on held-out users
3. Test cold-start prediction quality

**Success Criteria**:
- Population model improves new user predictions
- Effect size estimates align with literature
- Privacy guarantees maintained (DP verification)

---

## Part 5: Feature Engineering Recommendations

### 5.1 Time-Based Features

```python
def engineer_temporal_features(nutrition_log, target_time):
    """Critical temporal features for acute effects."""

    features = {
        # Hours since last meal (late eating detection)
        'hours_since_last_meal': (
            target_time - nutrition_log['consumed_at'].max()
        ).total_seconds() / 3600,

        # Cumulative effects windows
        'alcohol_last_4h': nutrition_log[
            nutrition_log['consumed_at'] > target_time - timedelta(hours=4)
        ]['alcohol_drinks'].sum(),

        'alcohol_last_24h': nutrition_log[
            nutrition_log['consumed_at'] > target_time - timedelta(hours=24)
        ]['alcohol_drinks'].sum(),

        'alcohol_last_5d': nutrition_log[
            nutrition_log['consumed_at'] > target_time - timedelta(days=5)
        ]['alcohol_drinks'].sum(),

        # Glycemic load in effect window
        'gl_last_3h': nutrition_log[
            nutrition_log['consumed_at'] > target_time - timedelta(hours=3)
        ].apply(
            lambda x: x['carbohydrates'] * x['glycemic_index'] / 100, axis=1
        ).sum(),

        # Circadian phase
        'circadian_phase': _get_circadian_phase(target_time),

        # Days since last significant alcohol
        'days_since_alcohol_3plus': _days_since_heavy_drinking(
            nutrition_log, target_time, threshold=3
        ),
    }

    return features
```

### 5.2 Interaction Features

```python
def engineer_interaction_features(nutrition_log, context, target_time):
    """
    Interaction effects that amplify/attenuate base effects.
    """

    base = engineer_temporal_features(nutrition_log, target_time)

    interactions = {
        # Alcohol + sleep debt = worse
        'alcohol_x_sleep_debt': (
            base['alcohol_last_24h'] * context.get('sleep_debt_hours', 0)
        ),

        # Late eating + large meal = worse
        'late_large_meal': int(
            base['hours_since_last_meal'] < 3 and
            nutrition_log.iloc[-1]['calories'] > 800
        ),

        # Caffeine + afternoon = worse (circadian interaction)
        'afternoon_caffeine': int(
            target_time.hour > 14 and
            base.get('caffeine_last_4h', 0) > 200
        ),

        # High GI + sedentary = worse (no glucose disposal)
        'high_gi_sedentary': (
            base['gl_last_3h'] * (1 - context.get('activity_today', 0))
        ),

        # Alcohol recovery status (recent heavy drinking)
        'alcohol_recovery_phase': int(
            base['days_since_alcohol_3plus'] < 3
        ),
    }

    return {**base, **interactions}
```

### 5.3 Rolling Statistics

```python
def engineer_rolling_features(hrv_data, nutrition_log, window_days=7):
    """
    Rolling statistics for personalization and trend detection.
    """

    return {
        # Personal baselines
        'rmssd_baseline_7d': hrv_data['rmssd'].rolling(window_days).mean(),
        'rmssd_baseline_30d': hrv_data['rmssd'].rolling(30).mean(),

        # Variability (indicates reactivity)
        'rmssd_cv_7d': (
            hrv_data['rmssd'].rolling(window_days).std() /
            hrv_data['rmssd'].rolling(window_days).mean()
        ),

        # Trends
        'rmssd_trend_7d': _calculate_trend(hrv_data['rmssd'].tail(7)),

        # Recovery speed (how fast HRV bounces back)
        'recovery_rate': _calculate_recovery_rate(hrv_data),

        # Personal alcohol sensitivity
        'personal_alcohol_sensitivity': _estimate_alcohol_sensitivity(
            hrv_data, nutrition_log
        ),
    }
```

---

## Part 6: Model Evaluation Metrics

### 6.1 Primary Metrics

| Metric | Formula | Target | Notes |
|--------|---------|--------|-------|
| MAE | mean(\|y - ŷ\|) | < 5 ms | Absolute error in RMSSD |
| MAPE | mean(\|y - ŷ\| / y) | < 15% | Percentage error |
| R² | 1 - SS_res/SS_tot | > 0.3 | Variance explained |
| Directional Accuracy | correct_direction / total | > 75% | Up/down correct |

### 6.2 Secondary Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| Calibration Error | CI reliability | < 5% |
| Feature Stability | Consistent importance | Top 5 stable |
| Prediction Latency | Real-time capability | < 100ms |
| Cold Start MAPE | New user performance | < 25% |

### 6.3 Per-Factor Validation

Validate each nutritional factor separately:

```python
def validate_factor_effect(
    model,
    test_data,
    factor: str,
    expected_direction: str,
) -> FactorValidation:
    """
    Validate that model correctly captures factor effect.
    """

    # Get predictions with and without factor
    with_factor = test_data[test_data[f'{factor}_present'] == True]
    without_factor = test_data[test_data[f'{factor}_present'] == False]

    # Check directional accuracy
    if expected_direction == 'negative':
        # Factor should predict lower HRV
        correct = (
            model.predict(with_factor) < model.predict(without_factor)
        ).mean()
    else:
        correct = (
            model.predict(with_factor) > model.predict(without_factor)
        ).mean()

    # Effect size alignment
    predicted_effect = (
        model.predict(with_factor).mean() -
        model.predict(without_factor).mean()
    ) / model.predict(without_factor).mean()

    return FactorValidation(
        factor=factor,
        directional_accuracy=correct,
        predicted_effect_size=predicted_effect,
        expected_effect_size=RESEARCH_EFFECTS[factor],
        aligned_with_literature=abs(
            predicted_effect - RESEARCH_EFFECTS[factor]
        ) < 0.10,
    )
```

---

## Part 7: Implementation Roadmap

### 7.1 Pre-Beta Development

1. **Implement Data Collection Pipeline**
   - Nutrition logging with GI estimation
   - Wearable HRV integration (Apple HealthKit, Google Fit, Oura API)
   - Real-time data synchronization

2. **Build Tier 1 Rule Engine**
   - Implement all high-confidence rules
   - Add explanation generation
   - Create prediction visualization

3. **Prepare ML Infrastructure**
   - Feature engineering pipeline
   - Model training automation
   - A/B testing framework

### 7.2 Beta Launch (Week 1-4)

4. **Deploy Rule-Based Predictions**
   - Show predictions before/after meals
   - Collect user feedback on accuracy
   - Log all predictions for validation

5. **Data Collection Optimization**
   - Identify logging friction points
   - Improve GI estimation accuracy
   - Add contextual prompts for key factors

### 7.3 Model Development (Week 5-12)

6. **Train Personalized Models**
   - Automated training pipeline
   - User-facing accuracy metrics
   - Feature importance dashboard

7. **Build Population Model**
   - Federated learning infrastructure
   - Privacy compliance verification
   - Cross-validation on held-out users

### 7.4 Post-Beta Refinement

8. **Model Iteration**
   - Incorporate beta feedback
   - Add new factors as data allows
   - Improve cold-start performance

---

## Appendix A: Research References

### Key Studies on Acute Nutrition-HRV Effects

1. **Alcohol**: Irwin et al. (2006) - HRV suppression following alcohol
2. **Glycemic Index**: Zaretsky et al. (2021) - Postprandial HRV response
3. **Caffeine**: Sondermeijer et al. (2002) - Caffeine and autonomic function
4. **Meal Timing**: Wehrens et al. (2017) - Circadian effects of eating timing
5. **Hydration**: Carter et al. (2005) - Dehydration and cardiovascular autonomic control

### Commercial System Methodologies

1. **Whoop Recovery**: Multi-factor HRV-based recovery scoring
2. **Oura Readiness**: Sleep and HRV integration
3. **Garmin Stress Score**: Real-time stress estimation from HRV

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| RMSSD | Root Mean Square of Successive Differences (primary HRV metric) |
| SDNN | Standard Deviation of NN intervals |
| LF/HF Ratio | Low Frequency / High Frequency power ratio |
| HRV | Heart Rate Variability |
| GI | Glycemic Index (0-100 scale) |
| GL | Glycemic Load (GI × carbs / 100) |
| Vagal Tone | Parasympathetic nervous system activity |

---

*Document Version: 1.0*
*Created: December 2025*
*For: Nutri ML Service Beta Testing*
