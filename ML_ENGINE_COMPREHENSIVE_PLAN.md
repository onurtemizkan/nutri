# Nutri ML Engine: Comprehensive Plan
## Predictive Health Analytics Through Nutrition & Activity Correlation

**Version**: 1.0
**Date**: 2025-01-17
**Status**: Planning Phase

---

## 🚨 IMPORTANT - What We're Building

**This is a MOBILE APPLICATION (React Native) with IN-HOUSE ML capabilities**

- ❌ **NOT** a chatbot or conversational AI interface
- ❌ **NOT** using external ML services (OpenAI, Claude, AWS ML, Google AI, Azure ML, etc.)
- ✅ **IS** a React Native mobile app with structured UI (charts, forms, lists)
- ✅ **IS** building ML models from scratch using PyTorch, scikit-learn, XGBoost
- ✅ **IS** training all models in-house on our own infrastructure

**ML Approach**: All machine learning models (LSTM, XGBoost, correlation engines, anomaly detection) are developed, trained, and deployed by us using open-source ML libraries. We build the models with **PyTorch** (for deep learning), we train them, we own them.

**UI Approach**: Mobile app screens with standard UI components (nutrition dashboards, health charts, recommendation cards), NOT chat/conversation interfaces.

---

## 📋 Executive Summary

This document outlines a comprehensive Machine Learning engine for Nutri that will:

1. **Integrate health metrics** from wearables (Apple Health, Fitbit, Garmin, Oura, Whoop)
2. **Analyze patterns** in nutrition, activity, and physiological responses
3. **Predict outcomes** such as how today's nutrition will affect tomorrow's RHR, HRV, and recovery
4. **Provide actionable insights** to optimize health and performance

**Core Capabilities**:
- Real-time correlation analysis between nutrition choices and health metrics
- Predictive modeling for RHR, HRV, and recovery scores
- Personalized recommendations based on individual physiological responses
- Anomaly detection for early warning of health issues

**Expected Impact**:
- Users understand their unique nutritional needs
- Data-driven optimization of diet for health goals
- Early detection of health pattern changes
- Personalized nutrition science at scale

---

## 🎯 Problem Definition

### The Challenge

**Traditional nutrition tracking** is backwards-looking and generic:
- "You ate 2,000 calories yesterday"
- "You hit your protein goal 5 days this week"
- One-size-fits-all macro recommendations

**What's missing**: Understanding the **cause-and-effect relationship** between:
- **Inputs**: Nutrition (macros, timing, quality) + Activity (type, intensity, duration)
- **Outputs**: Health metrics (RHR, HRV, sleep quality, recovery, inflammation markers)

### The Solution

**A personalized ML engine** that learns each user's unique physiological responses:
- "Your high-carb dinner (>100g) correlates with +5 bpm RHR the next morning"
- "Eating protein within 2 hours post-workout improves your HRV by 8ms"
- "Your recovery score drops 15% when you skip breakfast"
- "Predicted tomorrow's RHR: 58 bpm (±2) based on today's nutrition and activity"

### Key Questions We'll Answer

1. **Correlation Analysis**:
   - Which foods/macros correlate with better/worse RHR, HRV, sleep?
   - How does meal timing affect recovery?
   - What's the optimal protein/carb ratio for YOUR body?

2. **Predictive Modeling**:
   - What will my RHR be tomorrow given today's nutrition?
   - How will this meal affect my sleep quality tonight?
   - Am I on track to hit my weight goal based on current patterns?

3. **Causal Inference** (Advanced):
   - Does high sugar intake CAUSE elevated RHR, or just correlate?
   - What's the lag effect? (nutrition → 6h, 12h, 24h, 48h → metrics)

4. **Personalization**:
   - How do I respond differently to carbs vs. others?
   - What's MY optimal eating window?
   - Which supplements actually move the needle for MY metrics?

---

## 🗄️ Data Architecture

### Current State (Nutri DB)

**Existing Tables**:
```prisma
User (id, email, password, nutritionGoals, physicalProfile)
Meal (id, userId, nutrition[7 fields], mealType, consumedAt)
WaterIntake (id, userId, amount, recordedAt)
WeightRecord (id, userId, weight, recordedAt)
```

**Strengths**:
- ✅ Comprehensive nutrition tracking (7 fields)
- ✅ Precise timestamps for temporal analysis
- ✅ User profiles with goals and activity levels
- ✅ Optimized indexes for time-based queries

**Gaps**:
- ❌ No health metrics (RHR, HRV, sleep, etc.)
- ❌ No activity data (workouts, steps, etc.)
- ❌ No environmental context (stress, illness, menstrual cycle)
- ❌ No ML-specific tables (features, predictions, models)

### New Data Models (Required)

#### 1. HealthMetric (Time Series Health Data)

```prisma
model HealthMetric {
  id              String   @id @default(cuid())
  userId          String
  user            User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  // Timestamp (UTC, precise to second)
  recordedAt      DateTime

  // Metric type (enum for type safety)
  metricType      HealthMetricType

  // Value (flexible for different units)
  value           Float
  unit            String   // "bpm", "ms", "%", "steps", "kcal", etc.

  // Source (for data provenance)
  source          String   // "apple_health", "fitbit", "garmin", "oura", "whoop", "manual"
  sourceId        String?  // Original ID from source system

  // Metadata (JSON for flexibility)
  metadata        Json?    // {quality: "high", confidence: 0.95, device: "Apple Watch Series 9"}

  // Audit
  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt

  @@index([userId, recordedAt])
  @@index([userId, metricType, recordedAt])
  @@unique([userId, metricType, recordedAt, source])
}

enum HealthMetricType {
  // Cardiovascular
  RESTING_HEART_RATE
  HEART_RATE_VARIABILITY_SDNN
  HEART_RATE_VARIABILITY_RMSSD
  BLOOD_PRESSURE_SYSTOLIC
  BLOOD_PRESSURE_DIASTOLIC

  // Respiratory
  RESPIRATORY_RATE
  OXYGEN_SATURATION
  VO2_MAX

  // Sleep
  SLEEP_DURATION
  DEEP_SLEEP_DURATION
  REM_SLEEP_DURATION
  SLEEP_EFFICIENCY
  SLEEP_SCORE

  // Activity
  STEPS
  ACTIVE_CALORIES
  TOTAL_CALORIES
  EXERCISE_MINUTES
  STANDING_HOURS

  // Recovery & Strain
  RECOVERY_SCORE
  STRAIN_SCORE
  READINESS_SCORE

  // Body Composition
  BODY_FAT_PERCENTAGE
  MUSCLE_MASS
  BONE_MASS
  WATER_PERCENTAGE

  // Other
  SKIN_TEMPERATURE
  BLOOD_GLUCOSE
  STRESS_LEVEL
}
```

**Rationale**:
- **Flexible schema**: Single table for all metric types (avoid 20+ tables)
- **Time series optimized**: Composite indexes for fast range queries
- **Source tracking**: Know where data came from (critical for ML)
- **Metadata**: Store confidence scores, device info, etc. without schema changes

#### 2. Activity (Exercise & Movement Data)

```prisma
model Activity {
  id              String       @id @default(cuid())
  userId          String
  user            User         @relation(fields: [userId], references: [id], onDelete: Cascade)

  // Timing
  startedAt       DateTime
  endedAt         DateTime
  duration        Int          // Duration in minutes

  // Type & Intensity
  activityType    ActivityType
  intensity       ActivityIntensity

  // Metrics
  caloriesBurned  Float?
  averageHeartRate Float?
  maxHeartRate    Float?
  distance        Float?       // In meters
  steps           Int?

  // Source
  source          String       // "apple_health", "strava", "garmin", "manual"
  sourceId        String?

  // Notes
  notes           String?

  // Audit
  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt

  @@index([userId, startedAt])
  @@index([userId, activityType, startedAt])
}

enum ActivityType {
  // Cardio
  RUNNING
  CYCLING
  SWIMMING
  WALKING
  HIKING
  ROWING
  ELLIPTICAL

  // Strength
  WEIGHT_TRAINING
  BODYWEIGHT
  CROSSFIT
  POWERLIFTING

  // Sports
  BASKETBALL
  SOCCER
  TENNIS
  GOLF

  // Other
  YOGA
  PILATES
  STRETCHING
  MARTIAL_ARTS
  DANCE
  OTHER
}

enum ActivityIntensity {
  LOW
  MODERATE
  HIGH
  MAXIMUM
}
```

**Rationale**:
- **Activity context**: Understand how exercise affects recovery and nutrition needs
- **Intensity tracking**: High-intensity workouts have different recovery needs
- **Integration ready**: Works with Strava, Apple Health, Garmin Connect

#### 3. MLFeature (Engineered Features for Training)

```prisma
model MLFeature {
  id              String   @id @default(cuid())
  userId          String
  user            User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  // Temporal scope
  date            DateTime // Date this feature represents (e.g., 2025-01-17)

  // Feature category
  category        MLFeatureCategory

  // Features (JSON for flexibility)
  features        Json     // {protein_7d_avg: 150, carbs_ratio: 0.4, meal_regularity: 0.85}

  // Metadata
  version         String   // "v1.2.3" - for feature engineering versioning

  // Audit
  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt

  @@index([userId, date, category])
  @@unique([userId, date, category, version])
}

enum MLFeatureCategory {
  NUTRITION_DAILY
  NUTRITION_WEEKLY
  NUTRITION_TEMPORAL
  ACTIVITY_DAILY
  ACTIVITY_WEEKLY
  HEALTH_DAILY
  HEALTH_WEEKLY
  COMBINED_FEATURES
}
```

**Rationale**:
- **Pre-computed features**: Don't re-calculate rolling averages on every prediction
- **Versioning**: Track feature engineering changes over time
- **Fast predictions**: Features ready for inference without DB joins

#### 4. MLPrediction (Model Outputs & Tracking)

```prisma
model MLPrediction {
  id              String   @id @default(cuid())
  userId          String
  user            User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  // What are we predicting?
  targetMetric    HealthMetricType
  targetDate      DateTime // When is this prediction for?

  // Prediction
  predictedValue  Float
  confidence      Float    // 0.0 to 1.0
  predictionRange Json     // {lower: 55, upper: 61} for confidence intervals

  // Model metadata
  modelId         String   // "rhr_lstm_v2.1.0"
  modelVersion    String   // "2.1.0"

  // Feature importance (what drove this prediction?)
  featureImportance Json   // {protein_intake: 0.35, sleep_quality: 0.28, ...}

  // Actual outcome (for model evaluation)
  actualValue     Float?
  predictionError Float?   // Calculated after actual value is recorded

  // Audit
  createdAt       DateTime @default(now())

  @@index([userId, targetDate, targetMetric])
  @@index([userId, modelId, createdAt])
}
```

**Rationale**:
- **Track predictions**: Monitor model performance over time
- **Explainability**: Show users WHY the prediction is what it is
- **Continuous learning**: Compare predictions vs. actuals to retrain models

#### 5. MLInsight (Actionable Recommendations)

```prisma
model MLInsight {
  id              String       @id @default(cuid())
  userId          String
  user            User         @relation(fields: [userId], references: [id], onDelete: Cascade)

  // Insight type
  insightType     MLInsightType
  priority        InsightPriority

  // Content
  title           String       // "Your carb timing affects sleep quality"
  description     String       // Detailed explanation
  recommendation  String       // "Try eating complex carbs earlier in the day"

  // Supporting data
  correlation     Float?       // Correlation coefficient (if applicable)
  confidence      Float        // How confident is the ML model? (0.0 to 1.0)
  dataPoints      Int          // How many data points support this insight?

  // Metadata
  metadata        Json?        // {chart_data: [...], references: [...]}

  // User interaction
  viewed          Boolean      @default(false)
  viewedAt        DateTime?
  dismissed       Boolean      @default(false)
  dismissedAt     DateTime?
  helpful         Boolean?     // User feedback: was this helpful?

  // Audit
  createdAt       DateTime @default(now())
  expiresAt       DateTime?    // Some insights are time-sensitive

  @@index([userId, createdAt])
  @@index([userId, insightType, priority])
}

enum MLInsightType {
  CORRELATION
  PREDICTION
  ANOMALY
  RECOMMENDATION
  GOAL_PROGRESS
  PATTERN_DETECTED
}

enum InsightPriority {
  LOW
  MEDIUM
  HIGH
  CRITICAL
}
```

**Rationale**:
- **Actionable insights**: ML is useless if users don't understand it
- **User feedback loop**: Learn which insights are actually helpful
- **Prioritization**: Don't overwhelm users with noise

#### 6. UserMLProfile (Per-User Model Configuration)

```prisma
model UserMLProfile {
  id              String   @id @default(cuid())
  userId          String   @unique
  user            User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  // Data quality
  totalDataPoints Int      @default(0)
  dataQualityScore Float   @default(0.0) // 0.0 to 1.0

  // Minimum data requirements met?
  hasMinimumNutritionData   Boolean @default(false)
  hasMinimumHealthData      Boolean @default(false)
  hasMinimumActivityData    Boolean @default(false)

  // Model readiness
  modelsAvailable Json     // {rhr_prediction: true, hrv_prediction: false, ...}
  lastTrainingDate DateTime?

  // User preferences
  enablePredictions Boolean @default(true)
  enableInsights    Boolean @default(true)
  insightFrequency  String  @default("daily") // "daily", "weekly", "realtime"

  // Privacy
  shareDataForResearch Boolean @default(false)
  dataRetentionDays    Int     @default(365)

  // Audit
  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt
}
```

**Rationale**:
- **Per-user models**: Everyone's physiology is different
- **Data quality tracking**: Don't make predictions with insufficient data
- **Privacy controls**: User decides how their data is used

---

## 🧠 ML Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  (React Native App - Predictions, Insights, Recommendations)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓ ↑
                         API Gateway
                              ↓ ↑
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
│  (Node.js + Express + TypeScript)                               │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Auth API    │  │  Data API    │  │   ML API     │         │
│  │  (JWT)       │  │  (CRUD)      │  │  (Predict)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────────┐
│                      ML SERVICE LAYER                           │
│  (Python + FastAPI + TensorFlow/PyTorch)                        │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  Feature Engineering │  │  Model Inference     │            │
│  │  - Time series       │  │  - RHR Prediction    │            │
│  │  - Aggregations      │  │  - HRV Prediction    │            │
│  │  - Lag features      │  │  - Sleep Prediction  │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  Correlation Engine  │  │  Insight Generator   │            │
│  │  - Pearson/Spearman  │  │  - NLP Templates     │            │
│  │  - Granger causality │  │  - Ranking           │            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                 │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  PostgreSQL          │  │  Redis Cache         │            │
│  │  (Transactional)     │  │  (Features, Models)  │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  TimescaleDB         │  │  S3 / Object Storage │            │
│  │  (Time Series)       │  │  (Model Artifacts)   │            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              ↓ ↑
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE (Offline)                   │
│  (Apache Airflow / Prefect / Temporal)                          │
│                                                                 │
│  Daily:                                                         │
│  1. Extract data → 2. Engineer features →                      │
│  3. Train models → 4. Evaluate → 5. Deploy (if better)         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. ML Service Layer (Python + FastAPI)

**Why Python?**
- Rich ML ecosystem (scikit-learn, TensorFlow, PyTorch, Prophet)
- Fast numerical computation (NumPy, Pandas)
- Better for data science than Node.js

**Why FastAPI?**
- Modern async Python framework
- Auto-generated OpenAPI docs
- Type hints (Pydantic) for safety
- Easy integration with Node.js backend

**Service Structure**:
```
ml-service/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Environment variables, DB connections
│   ├── models/                 # Pydantic request/response models
│   ├── services/
│   │   ├── feature_engineering.py
│   │   ├── rhr_predictor.py
│   │   ├── hrv_predictor.py
│   │   ├── correlation_engine.py
│   │   └── insight_generator.py
│   ├── ml_models/              # Trained model artifacts
│   │   ├── rhr_lstm_v1.pkl
│   │   ├── hrv_xgboost_v1.pkl
│   │   └── model_registry.json
│   └── utils/
│       ├── db.py               # Database helpers
│       ├── cache.py            # Redis helpers
│       └── metrics.py          # Monitoring
├── training/                   # Offline training scripts
│   ├── train_rhr_model.py
│   ├── train_hrv_model.py
│   └── evaluate_models.py
├── tests/
├── requirements.txt
└── Dockerfile
```

**API Endpoints**:
```python
# Predictions
POST /ml/predict/rhr
POST /ml/predict/hrv
POST /ml/predict/sleep-quality

# Correlations
GET /ml/correlations?userId=X&metric=RHR&days=30

# Insights
GET /ml/insights?userId=X&limit=10

# Feature Importance
GET /ml/explain?userId=X&predictionId=Y

# Health Check
GET /ml/health
```

#### 2. Feature Engineering Pipeline

**Goal**: Transform raw data into ML-ready features

**Input**:
- Meal data (nutrition, timing)
- Activity data (type, intensity, duration)
- Health metrics (RHR, HRV, sleep)

**Output**:
- Engineered features (rolling averages, lag features, ratios, etc.)

**Example Features**:

```python
# Nutritional Features
- protein_daily        # Today's total protein (g)
- carbs_daily          # Today's total carbs (g)
- fat_daily            # Today's total fat (g)
- calories_daily       # Today's total calories
- protein_7d_avg       # 7-day rolling average protein
- carbs_7d_avg         # 7-day rolling average carbs
- calorie_deficit_7d   # Average deficit vs. TDEE
- meal_regularity      # Coefficient of variation in meal times
- protein_ratio        # Protein as % of total calories
- carb_ratio           # Carbs as % of total calories
- fat_ratio            # Fat as % of total calories
- fiber_daily          # Today's fiber (g)
- sugar_daily          # Today's sugar (g)
- meal_count_daily     # Number of meals today
- first_meal_time      # Time of first meal (decimal hours)
- last_meal_time       # Time of last meal
- eating_window        # Hours between first and last meal
- late_night_carbs     # Carbs consumed after 8pm (g)

# Activity Features
- steps_daily          # Today's steps
- active_minutes_daily # Today's active minutes
- calories_burned      # Today's active calories
- workout_intensity    # Average intensity (1-4 scale)
- recovery_time        # Hours since last high-intensity workout
- steps_7d_avg         # 7-day rolling average steps

# Health Features (Lagged)
- rhr_yesterday        # Yesterday's RHR
- rhr_7d_avg           # 7-day rolling average RHR
- rhr_trend            # Slope of RHR over last 7 days
- hrv_yesterday        # Yesterday's HRV
- hrv_7d_avg           # 7-day rolling average HRV
- sleep_duration_last  # Last night's sleep (hours)
- sleep_quality_last   # Last night's sleep score

# Temporal Features
- day_of_week          # Monday=0, Sunday=6
- is_weekend           # Boolean
- day_of_month         # 1-31
- month                # 1-12
- days_since_start     # Days since user started tracking

# Interaction Features
- protein_per_kg       # Protein / body weight
- carbs_per_activity   # Carbs / active minutes
- calorie_surplus      # Calories - TDEE
```

**Feature Engineering Code** (Pseudocode):
```python
def engineer_features(user_id: str, date: datetime) -> dict:
    """
    Engineer features for a specific user and date.
    """
    features = {}

    # Get raw data
    meals = get_meals(user_id, date, lookback_days=30)
    activities = get_activities(user_id, date, lookback_days=30)
    health_metrics = get_health_metrics(user_id, date, lookback_days=30)

    # Daily nutrition
    today_meals = filter_by_date(meals, date)
    features['protein_daily'] = sum(m.protein for m in today_meals)
    features['carbs_daily'] = sum(m.carbs for m in today_meals)
    features['calories_daily'] = sum(m.calories for m in today_meals)

    # Rolling averages (7 days)
    last_7d_meals = filter_last_n_days(meals, date, 7)
    features['protein_7d_avg'] = avg_daily(last_7d_meals, 'protein')
    features['carbs_7d_avg'] = avg_daily(last_7d_meals, 'carbs')

    # Meal timing
    meal_times = [m.consumedAt.hour + m.consumedAt.minute/60 for m in today_meals]
    if meal_times:
        features['first_meal_time'] = min(meal_times)
        features['last_meal_time'] = max(meal_times)
        features['eating_window'] = max(meal_times) - min(meal_times)

    # Activity
    today_activity = filter_by_date(activities, date)
    features['steps_daily'] = sum(a.steps for a in today_activity)
    features['active_minutes_daily'] = sum(a.duration for a in today_activity)

    # Lagged health metrics (yesterday's values)
    yesterday = date - timedelta(days=1)
    rhr_yesterday = get_metric(health_metrics, 'RHR', yesterday)
    features['rhr_yesterday'] = rhr_yesterday or 0

    # Rolling health trends
    last_7d_rhr = get_metric_range(health_metrics, 'RHR', date, 7)
    features['rhr_7d_avg'] = mean(last_7d_rhr)
    features['rhr_trend'] = calculate_trend(last_7d_rhr)

    # Temporal
    features['day_of_week'] = date.weekday()
    features['is_weekend'] = 1 if date.weekday() >= 5 else 0

    return features
```

#### 3. ML Models

**We'll build multiple models for different predictions:**

##### Model 1: RHR Prediction (Next-Day Resting Heart Rate)

**Problem**: Predict tomorrow's RHR given today's nutrition, activity, and recent health trends.

**Approach**: Time series regression

**Model Options**:
1. **LSTM (Long Short-Term Memory)** - Best for sequence data
2. **XGBoost** - Great for tabular features, fast
3. **Prophet** (Facebook) - Good for seasonal patterns

**Recommended**: **LSTM** for capturing complex temporal dependencies

**Architecture**:
```python
Input: [30 days of features] → Shape: (30, 50)
  ↓
LSTM Layer 1 (128 units, return_sequences=True)
  ↓
Dropout (0.2)
  ↓
LSTM Layer 2 (64 units)
  ↓
Dense Layer (32 units, ReLU)
  ↓
Dense Layer (1 unit, Linear) → Predicted RHR
```

**Training**:
- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Metrics**: MAE (Mean Absolute Error), R² score
- **Validation**: Time series split (train on first 80% of days, validate on last 20%)

##### Model 2: HRV Prediction (Next-Day Heart Rate Variability)

**Problem**: Predict tomorrow's HRV (RMSSD) given today's nutrition, activity, sleep, and stress.

**Approach**: Similar to RHR, but HRV is more sensitive to:
- Sleep quality (strongest predictor)
- Alcohol consumption
- Stress levels
- Overtraining

**Model**: LSTM or Gradient Boosting

**Special considerations**:
- HRV is non-linear (not normally distributed)
- May need log transformation
- Outlier detection (illness, alcohol can cause major drops)

##### Model 3: Correlation Engine

**Problem**: Identify which nutritional/activity factors correlate with health metric changes.

**Approach**: Statistical correlation analysis

**Methods**:
1. **Pearson correlation**: For linear relationships
2. **Spearman correlation**: For monotonic (but non-linear) relationships
3. **Granger causality**: For temporal causation (does X predict Y?)
4. **Cross-correlation**: For lagged effects (how does today's X affect tomorrow's Y?)

**Output**:
- Correlation matrix
- Statistical significance (p-values)
- Effect sizes
- Lag analysis (0h, 6h, 12h, 24h, 48h)

**Example**:
```python
def calculate_correlations(user_id: str, metric: str, days: int = 30):
    """
    Calculate correlations between nutrition/activity and a health metric.
    """
    # Get data
    features = get_features(user_id, days)
    target = get_health_metric(user_id, metric, days)

    correlations = []

    for feature_name in features.columns:
        # Pearson correlation
        r, p_value = pearsonr(features[feature_name], target)

        # Lag analysis
        lag_correlations = []
        for lag in [0, 1, 2, 3, 7]:  # 0, 1, 2, 3, 7 days
            lagged_feature = features[feature_name].shift(lag)
            r_lag, p_lag = pearsonr(lagged_feature, target)
            lag_correlations.append({
                'lag_days': lag,
                'correlation': r_lag,
                'p_value': p_lag
            })

        correlations.append({
            'feature': feature_name,
            'correlation': r,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'lag_analysis': lag_correlations
        })

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return correlations
```

##### Model 4: Anomaly Detection

**Problem**: Detect unusual patterns that might indicate health issues, overtraining, or illness.

**Approach**: Unsupervised learning

**Models**:
1. **Isolation Forest**: Detects outliers in multi-dimensional space
2. **Autoencoder**: Neural network that learns normal patterns, flags deviations
3. **Statistical control charts**: Flag values outside 2-3 standard deviations

**Use cases**:
- RHR spike (possible illness, overtraining, stress)
- HRV drop (poor recovery, overtraining)
- Sudden weight changes
- Sleep pattern disruption

**Alert thresholds**:
- **Yellow**: 2 standard deviations from baseline
- **Red**: 3 standard deviations or multiple metrics anomalous

##### Model 5: Personalized Nutrition Recommendations

**Problem**: Suggest optimal macros for user's goals given their unique responses.

**Approach**: Multi-armed bandit or reinforcement learning

**Simplified approach (Phase 1)**:
- Analyze historical data
- Find macro ratios that correlated with best health metrics
- Recommend similar ratios going forward

**Advanced approach (Phase 2)**:
- Reinforcement learning agent
- Try different macro strategies
- Learn from outcomes
- Optimize over time

---

## 🔧 Feature Engineering Deep Dive

### Rolling Window Features

**Why?** Single-day data is noisy. Rolling averages smooth out noise.

**Examples**:
```python
# 7-day rolling average protein
protein_7d_avg = meals.groupby('userId')['protein'].rolling(7).mean()

# 7-day standard deviation (measure of consistency)
protein_7d_std = meals.groupby('userId')['protein'].rolling(7).std()

# Coefficient of variation (normalized consistency)
protein_cv = protein_7d_std / protein_7d_avg
```

### Lag Features

**Why?** Effects aren't immediate. Today's nutrition affects tomorrow's RHR.

**Examples**:
```python
# Yesterday's values
features['protein_lag1'] = features['protein_daily'].shift(1)
features['carbs_lag1'] = features['carbs_daily'].shift(1)

# 2 days ago (for longer effects)
features['protein_lag2'] = features['protein_daily'].shift(2)

# Multi-day lags (for training recovery)
features['workout_intensity_lag3'] = features['workout_intensity'].shift(3)
```

### Temporal Features

**Why?** Body responds differently on different days/times.

**Examples**:
```python
# Day of week (Monday effect, weekend effect)
features['day_of_week'] = date.weekday()
features['is_monday'] = 1 if date.weekday() == 0 else 0
features['is_weekend'] = 1 if date.weekday() >= 5 else 0

# Time since last rest day
features['days_since_rest'] = calculate_days_since_last_rest(activities)

# Menstrual cycle phase (if tracking)
features['cycle_phase'] = get_cycle_phase(user_id, date)  # 1-4
```

### Interaction Features

**Why?** Combinations of features matter more than individuals.

**Examples**:
```python
# Protein relative to body weight
features['protein_per_kg'] = features['protein_daily'] / user.weight

# Carb timing relative to workout
features['carbs_post_workout'] = calculate_carbs_after_workout(meals, activities)

# Calorie deficit/surplus
features['calorie_balance'] = features['calories_daily'] - user.tdee

# Macro ratio
features['protein_ratio'] = features['protein_daily'] * 4 / features['calories_daily']
```

### Feature Importance Analysis

**After training models, we can see which features matter most:**

```python
# For tree-based models (XGBoost)
importance = model.feature_importances_
feature_names = features.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# For neural networks (LSTM)
# Use SHAP (SHapley Additive exPlanations)
import shap
explainer = shap.DeepExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Example output**:
```
Feature                  Importance
sleep_quality_last       0.35
protein_7d_avg           0.18
workout_intensity_lag1   0.12
carbs_post_workout       0.09
stress_level_yesterday   0.08
...
```

**Use this to generate insights**:
- "Your RHR is most affected by sleep quality (35% importance)"
- "Protein intake has 2nd highest impact on your recovery"

---

## 📊 Implementation Phases

### Phase 0: Foundation (Week 1-2)

**Goal**: Set up infrastructure before ML

**Tasks**:
1. ✅ Database schema updates
   - Add HealthMetric table
   - Add Activity table
   - Add MLFeature, MLPrediction, MLInsight, UserMLProfile tables
   - Run migrations

2. ✅ Health data integration
   - Apple Health integration (iOS)
   - Fitbit API integration
   - Garmin API integration
   - Manual entry UI for unsupported devices

3. ✅ Data collection
   - Start collecting health metrics
   - Start collecting activity data
   - Build data quality monitoring dashboard

4. ✅ Python ML service setup
   - FastAPI app skeleton
   - Docker container
   - PostgreSQL connection
   - Redis cache setup

**Deliverables**:
- Database with health data schema
- Data ingestion pipelines
- ML service running locally
- 30 days of user data collected

**Success criteria**:
- 100+ users with 30+ days of nutrition data
- 50+ users with connected wearables
- ML service responding to health checks

---

### Phase 1: Correlation Analysis (Week 3-4)

**Goal**: Find patterns in existing data (no predictions yet)

**Tasks**:
1. ✅ Feature engineering pipeline
   - Extract nutrition features
   - Extract activity features
   - Extract health metric features
   - Calculate rolling averages, lag features

2. ✅ Correlation engine
   - Pearson/Spearman correlations
   - Lag analysis (0-7 days)
   - Statistical significance testing
   - Visualization (correlation heatmaps)

3. ✅ Insight generation (rule-based)
   - Template-based insights
   - "Your X correlates with Y (r=0.65, p<0.01)"
   - Rank insights by correlation strength + significance

4. ✅ Frontend UI
   - "Insights" tab
   - Correlation visualizations
   - Explain correlations in plain English

**Deliverables**:
- Correlation API endpoint
- Insight generation service
- User-facing insights in app

**Example insights**:
- "High protein intake (>150g) correlates with lower RHR the next day (r=-0.42, p=0.003)"
- "Your HRV is 12ms higher on days after eating >50g carbs at dinner"
- "Sleep quality drops 15% when you eat within 2 hours of bedtime"

**Success criteria**:
- 80%+ users receive at least 3 significant insights
- Insights are statistically significant (p < 0.05)
- User feedback: "Was this insight helpful?" > 70% yes

---

### Phase 2: Prediction Models (Week 5-8)

**Goal**: Predict next-day health metrics

**Tasks**:
1. ✅ Data preparation
   - Train/validation/test split (temporal split)
   - Feature normalization
   - Handle missing data
   - Outlier removal

2. ✅ Model development
   - Train LSTM for RHR prediction
   - Train LSTM for HRV prediction
   - Hyperparameter tuning
   - Cross-validation

3. ✅ Model evaluation
   - MAE, RMSE, R² on test set
   - Compare vs. naive baseline (yesterday's value)
   - Per-user evaluation (some users more predictable)

4. ✅ Model deployment
   - Save trained models
   - Inference API endpoints
   - Cache predictions in Redis
   - Store predictions in DB

5. ✅ Frontend UI
   - "Tomorrow's Forecast" section
   - Show predicted RHR, HRV with confidence intervals
   - Explain what drove the prediction

**Deliverables**:
- Trained LSTM models for RHR and HRV
- Prediction API endpoints
- Frontend prediction UI

**Success criteria**:
- RHR prediction MAE < 3 bpm
- HRV prediction MAE < 10ms
- Predictions better than "yesterday's value" baseline
- User feedback: predictions feel accurate

---

### Phase 3: Recommendations (Week 9-10)

**Goal**: Suggest optimal nutrition based on predictions

**Tasks**:
1. ✅ Recommendation engine
   - Analyze user's historical best days
   - Find common patterns on those days
   - Suggest similar nutrition/activity

2. ✅ Scenario testing
   - "What if I eat X grams of protein today?"
   - "What if I do a high-intensity workout?"
   - Run prediction with hypothetical inputs

3. ✅ Optimization
   - Multi-objective optimization
   - Maximize HRV + minimize RHR + hit calorie goal
   - Suggest macro split that achieves all goals

4. ✅ Frontend UI
   - "Recommendations" tab
   - Daily macro suggestions
   - Pre/post workout nutrition timing

**Deliverables**:
- Recommendation API
- Scenario testing ("what-if" analysis)
- Frontend recommendation UI

**Example recommendations**:
- "For optimal recovery tomorrow, aim for 160g protein, 250g carbs, 60g fat"
- "Your HRV is highest when you eat breakfast by 9am"
- "Post-workout carbs within 1 hour boost your recovery by 18%"

---

### Phase 4: Anomaly Detection (Week 11-12)

**Goal**: Alert users to unusual patterns

**Tasks**:
1. ✅ Baseline calculation
   - Calculate per-user normal ranges
   - RHR baseline ± 2 standard deviations
   - HRV baseline ± 2 standard deviations

2. ✅ Anomaly detection
   - Real-time monitoring of new data
   - Flag values outside normal range
   - Multi-metric anomalies (RHR up + HRV down = red flag)

3. ✅ Alert system
   - In-app notifications
   - Email/SMS for critical anomalies
   - Recommendations: "Consider a rest day"

4. ✅ Illness detection
   - Pattern recognition for illness
   - RHR spike + HRV drop + poor sleep = likely illness
   - Suggest reducing training load

**Deliverables**:
- Anomaly detection service
- Alert notification system
- Frontend anomaly alerts

**Example alerts**:
- "Your RHR is 8 bpm higher than normal. This could indicate overtraining or illness. Consider a rest day."
- "Your HRV has dropped 15ms over the last 3 days. Recovery may be impaired."

---

### Phase 5: Advanced Features (Week 13+)

**Goal**: Cutting-edge personalization

**Tasks**:
1. ✅ Multi-step forecasting
   - Predict RHR for next 7 days
   - Predict weight trajectory
   - Goal achievement probability

2. ✅ Causal inference
   - Granger causality tests
   - Do-calculus for interventions
   - "If you increase protein by 20g, your HRV will likely increase by 5ms"

3. ✅ Reinforcement learning
   - Agent learns optimal nutrition strategy
   - Explores different macro ratios
   - Converges on personalized optimal diet

4. ✅ Group insights
   - Aggregate anonymized data across users
   - "Users similar to you find X effective"
   - Population-level nutrition science

5. ✅ Integration with medical data
   - Blood glucose monitors (CGM)
   - Blood tests (lipid panel, HbA1c)
   - Genetic data (if available)

**Deliverables**:
- Advanced prediction models
- Causal inference engine
- Population-level insights

---

## 🔒 Privacy & Security

### Critical Considerations

**Health data is the most sensitive personal data.** We must:
- ✅ Encrypt at rest (database-level encryption)
- ✅ Encrypt in transit (HTTPS, TLS 1.3)
- ✅ User consent (explicit opt-in for ML features)
- ✅ Data retention policies (delete after N days per user preference)
- ✅ Anonymization (for population-level research)
- ✅ GDPR/HIPAA compliance (if operating in EU/US healthcare)

### Data Flow Security

```
User's Wearable Device
  ↓ (Encrypted API call, OAuth 2.0)
Wearable Provider (Apple, Fitbit, Garmin)
  ↓ (HTTPS, API key)
Nutri Backend (Node.js)
  ↓ (TLS, service-to-service auth)
ML Service (Python)
  ↓ (TLS, encrypted connection)
PostgreSQL (AES-256 encryption at rest)
```

### User Controls

**Users must be able to**:
1. ✅ Opt-in to ML features (default: off)
2. ✅ See all data we have about them
3. ✅ Delete any data point
4. ✅ Export their data (JSON download)
5. ✅ Disconnect wearables anytime
6. ✅ Delete their account (permanent data deletion)

### Anonymization for Research

**If users opt-in to share data for research**:
- Remove all PII (name, email, IP address)
- Replace userId with random UUID
- Aggregate metrics (never individual-level data)
- Publish only statistical summaries

---

## 🛠️ Technical Stack

### Backend (Node.js)

**Existing**:
- Express 4.21.2
- TypeScript
- Prisma ORM 6.2.0
- PostgreSQL
- JWT auth

**New**:
- Redis (feature caching, rate limiting)
- BullMQ (job queue for ML tasks)

### ML Service (Python)

**Framework**: FastAPI

**ML Libraries**:
- **TensorFlow 2.x** or **PyTorch** (LSTM models)
- **scikit-learn** (correlation, preprocessing)
- **XGBoost** (gradient boosting)
- **Prophet** (time series forecasting)
- **SHAP** (model explainability)
- **Pandas** (data manipulation)
- **NumPy** (numerical computation)

**Infrastructure**:
- **Redis** (caching, model storage)
- **SQLAlchemy** (PostgreSQL ORM for Python)
- **Pydantic** (type validation)
- **Uvicorn** (ASGI server)

### Database

**PostgreSQL** (existing)
- ✅ ACID compliance
- ✅ Relational data (users, meals)
- ✅ JSON support (metadata, features)

**TimescaleDB** (extension for PostgreSQL)
- ✅ Optimized for time series data
- ✅ Automatic partitioning by time
- ✅ Faster aggregations (rolling averages)
- ✅ Compression for old data

**Redis**
- ✅ Cache frequently accessed features
- ✅ Store latest predictions (TTL: 24h)
- ✅ Rate limiting for ML API

### Training Pipeline

**Apache Airflow** or **Prefect**
- Schedule daily model retraining
- ETL: Extract → Transform → Load
- Model evaluation → Deploy if better

**Workflow**:
```
Daily at 2am UTC:
1. Extract last 30 days of data
2. Engineer features
3. Train models (RHR, HRV)
4. Evaluate on validation set
5. If MAE improved by >5%: Deploy new model
6. Else: Keep existing model
7. Log metrics to monitoring dashboard
```

### Monitoring

**Prometheus + Grafana**
- Model performance metrics (MAE, R²)
- API latency (p50, p95, p99)
- Error rates
- Feature drift (are features changing over time?)

**Sentry**
- Error tracking
- Performance monitoring

---

## 📈 Success Metrics

### Product Metrics

**Engagement**:
- % users who enable ML features
- Daily active users viewing insights
- Time spent on insights/predictions screens

**Retention**:
- 7-day retention for users with ML features
- 30-day retention
- Churn rate

**User satisfaction**:
- "Was this insight helpful?" feedback (target: >70%)
- Net Promoter Score (NPS)
- App store ratings

### ML Metrics

**Model performance**:
- RHR prediction MAE (target: <3 bpm)
- HRV prediction MAE (target: <10ms)
- R² score (target: >0.6)

**Prediction accuracy over time**:
- Are predictions getting better as more data collected?
- Per-user learning curves

**Insight quality**:
- % insights with p-value < 0.05 (target: >80%)
- % insights marked helpful (target: >70%)

### Business Metrics

**Conversion**:
- Free → Paid conversion (if ML is premium feature)

**Revenue**:
- Revenue per user (ARPU)
- Lifetime value (LTV)

**Costs**:
- ML service compute costs
- API costs (wearable integrations)
- Storage costs (time series data)

---

## 🚀 Go-to-Market Strategy

### Phased Rollout

**Beta (Phase 1-2)**:
- Invite 100 engaged users
- Collect feedback
- Iterate on insights and predictions

**Limited release (Phase 3)**:
- 1,000 users
- A/B test: ML features on vs. off
- Measure retention, engagement, satisfaction

**General availability (Phase 4+)**:
- All users (free tier: basic insights, paid tier: predictions)

### Pricing (If Premium Feature)

**Free Tier**:
- Basic insights (top 3 correlations)
- Weekly summary
- 30-day data retention

**Premium Tier** ($9.99/month):
- All insights
- Daily predictions (RHR, HRV, sleep)
- Recommendations
- Anomaly alerts
- 365-day data retention
- Export data

---

## 🧪 Testing Strategy

### Unit Tests

- Feature engineering functions
- Correlation calculations
- Prediction API endpoints

### Integration Tests

- Node.js ↔ Python ML service
- Database read/write
- Redis caching

### Model Tests

- Prediction accuracy on held-out test set
- Robustness to missing data
- Edge cases (new users with <7 days data)

### User Acceptance Testing (UAT)

- Beta users test insights
- Are predictions understandable?
- Are recommendations actionable?

---

## 📚 Documentation

**For developers**:
- API documentation (OpenAPI/Swagger)
- Model architecture diagrams
- Feature engineering guide
- Deployment runbook

**For users**:
- "How ML works" explainer in app
- FAQ: "Why did my RHR prediction change?"
- Privacy policy updates

---

## 🎯 Next Steps (Immediate Actions)

### Week 1: Database & Data Collection

1. ✅ Design Prisma schema for new tables
2. ✅ Write migration scripts
3. ✅ Implement Apple Health integration
4. ✅ Build manual entry UI for health metrics
5. ✅ Deploy to staging, test data flow

### Week 2: ML Service Foundation

1. ✅ Set up FastAPI project
2. ✅ Implement /health endpoint
3. ✅ Connect to PostgreSQL
4. ✅ Implement feature engineering pipeline (basic)
5. ✅ Write unit tests

### Week 3: First ML Feature (Correlations)

1. ✅ Implement correlation engine
2. ✅ Create /ml/correlations endpoint
3. ✅ Build insight generation (template-based)
4. ✅ Create frontend "Insights" screen
5. ✅ Deploy to beta users

---

## 📖 Appendix A: Example API Contracts

### GET /ml/correlations

**Request**:
```json
GET /ml/correlations?userId=123&metric=RHR&days=30
```

**Response**:
```json
{
  "userId": "123",
  "metric": "RHR",
  "period": {
    "start": "2024-12-18",
    "end": "2025-01-17",
    "days": 30
  },
  "correlations": [
    {
      "feature": "protein_7d_avg",
      "correlation": -0.42,
      "p_value": 0.003,
      "significant": true,
      "interpretation": "Higher protein intake correlates with lower resting heart rate",
      "lag_analysis": [
        {"lag_days": 0, "correlation": -0.28, "p_value": 0.08},
        {"lag_days": 1, "correlation": -0.42, "p_value": 0.003},
        {"lag_days": 2, "correlation": -0.35, "p_value": 0.01}
      ]
    },
    {
      "feature": "sleep_quality_last",
      "correlation": -0.65,
      "p_value": 0.0001,
      "significant": true,
      "interpretation": "Better sleep quality strongly correlates with lower resting heart rate"
    }
  ]
}
```

### POST /ml/predict/rhr

**Request**:
```json
POST /ml/predict/rhr
{
  "userId": "123",
  "targetDate": "2025-01-18"
}
```

**Response**:
```json
{
  "userId": "123",
  "targetMetric": "RESTING_HEART_RATE",
  "targetDate": "2025-01-18",
  "prediction": {
    "value": 58.2,
    "unit": "bpm",
    "confidence": 0.87,
    "confidenceInterval": {
      "lower": 55.8,
      "upper": 60.6
    }
  },
  "featureImportance": [
    {"feature": "sleep_quality_last", "importance": 0.35},
    {"feature": "protein_7d_avg", "importance": 0.18},
    {"feature": "workout_intensity_lag1", "importance": 0.12}
  ],
  "explanation": "Your predicted RHR is 58 bpm, primarily driven by good sleep quality last night (35% importance) and consistent protein intake over the last week (18% importance).",
  "modelId": "rhr_lstm_v2.1.0",
  "createdAt": "2025-01-17T10:30:00Z"
}
```

### GET /ml/insights

**Request**:
```json
GET /ml/insights?userId=123&limit=5&priority=HIGH
```

**Response**:
```json
{
  "userId": "123",
  "insights": [
    {
      "id": "insight_abc123",
      "type": "CORRELATION",
      "priority": "HIGH",
      "title": "High protein intake improves recovery",
      "description": "We found that on days when you consume >150g of protein, your next-day HRV is 12ms higher on average.",
      "recommendation": "Try to consistently hit 150g+ protein daily for optimal recovery.",
      "confidence": 0.92,
      "dataPoints": 28,
      "metadata": {
        "correlation": 0.68,
        "p_value": 0.0002
      },
      "createdAt": "2025-01-17T08:00:00Z"
    },
    {
      "id": "insight_def456",
      "type": "ANOMALY",
      "priority": "HIGH",
      "title": "Your RHR is elevated today",
      "description": "Your resting heart rate today (65 bpm) is 7 bpm higher than your 30-day average (58 bpm). This could indicate insufficient recovery, stress, or early illness.",
      "recommendation": "Consider taking a rest day or reducing training intensity.",
      "confidence": 0.88,
      "createdAt": "2025-01-17T07:30:00Z"
    }
  ]
}
```

---

## 📖 Appendix B: Data Requirements

### Minimum Data for ML

**Per user, we need**:

**Phase 1 (Correlations)**:
- ✅ 30 days of nutrition data (meals tracked)
- ✅ 30 days of health metrics (RHR, HRV, sleep)
- ✅ 30 days of activity data (optional but recommended)

**Phase 2 (Predictions)**:
- ✅ 90 days of nutrition data
- ✅ 90 days of health metrics
- ✅ Consistency: >80% days tracked

**Phase 3 (Recommendations)**:
- ✅ 6 months of data
- ✅ Variety: different macro ratios tried
- ✅ Outcome tracking: weight, performance metrics

### Data Quality Metrics

**Score = (Completeness × Consistency × Accuracy)**

**Completeness**:
- % days with nutrition logged
- % days with health metrics synced

**Consistency**:
- Coefficient of variation in logging times
- Gaps between logged days

**Accuracy**:
- User-reported confidence ("I estimated this meal")
- Manual vs. wearable data agreement

**Thresholds**:
- **High quality**: Score >0.8 → Enable all ML features
- **Medium quality**: Score 0.5-0.8 → Enable insights only
- **Low quality**: Score <0.5 → Encourage more tracking

---

## 🎉 Conclusion

This comprehensive plan outlines a **world-class ML engine** for Nutri that will:

1. ✅ **Integrate health metrics** from major wearables
2. ✅ **Analyze correlations** between nutrition, activity, and health outcomes
3. ✅ **Predict next-day metrics** (RHR, HRV, sleep quality)
4. ✅ **Generate personalized insights** unique to each user's physiology
5. ✅ **Recommend optimal nutrition** based on data, not guesswork
6. ✅ **Detect anomalies** early to prevent overtraining and illness

**Expected timeline**: 3-4 months from start to general availability

**Expected impact**:
- 2x user engagement
- 3x retention (users with ML features)
- New revenue stream (premium tier)
- Industry-leading nutrition personalization

**Next step**: Review this plan, prioritize phases, and kick off Phase 0 (Foundation) this week.

---

**Questions? Feedback? Let's discuss and refine!**
