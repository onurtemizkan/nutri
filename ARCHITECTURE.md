# Nutri Architecture - Mobile App with In-House ML Engine

**IMPORTANT**: This is a **mobile nutrition tracking application** with **in-house ML capabilities**.

- ❌ **NOT** a chatbot or conversational AI
- ❌ **NOT** using external ML services (OpenAI, Google AI, AWS ML, etc.)
- ✅ **IS** a React Native mobile app with custom ML models
- ✅ **IS** building ML models from scratch (PyTorch, scikit-learn)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MOBILE APPLICATION                            │
│                    (React Native + Expo)                         │
│                                                                  │
│  - Nutrition tracking UI (log meals, view macros)               │
│  - Health metrics dashboard (RHR, HRV charts)                   │
│  - Insights screen (ML-generated recommendations)               │
│  - Predictions view (tomorrow's health metrics)                 │
│  - Activity tracking (workouts, steps)                          │
│                                                                  │
│  NO CHAT INTERFACE - This is a structured mobile app UI         │
└─────────────────────────────────────────────────────────────────┘
                            ↓ ↑ HTTP/REST
┌─────────────────────────────────────────────────────────────────┐
│                    NODE.JS BACKEND                               │
│                    (Express + TypeScript + Prisma)              │
│                    Port: 3000                                    │
│                                                                  │
│  APIs:                                                           │
│  - POST /api/auth/register, /login (JWT authentication)        │
│  - POST /api/meals (log nutrition)                             │
│  - GET  /api/meals/summary/daily (view nutrition summary)      │
│  - POST /api/health-metrics (sync from smartwatch)             │
│  - GET  /api/health-metrics/timeseries/:metricType (charts)   │
│  - POST /api/activities (log workouts)                         │
│                                                                  │
│  Role: CRUD operations, authentication, data validation         │
└─────────────────────────────────────────────────────────────────┘
                            ↓ ↑ Internal HTTP
┌─────────────────────────────────────────────────────────────────┐
│                    PYTHON ML SERVICE                             │
│                    (FastAPI + In-House ML Models)               │
│                    Port: 8000                                    │
│                                                                  │
│  APIs (TO BE BUILT):                                            │
│  - POST /api/ml/correlations/{userId}                          │
│       → Analyze nutrition → health metric correlations         │
│  - POST /api/ml/predict/rhr                                    │
│       → Predict tomorrow's RHR using our LSTM model            │
│  - POST /api/ml/predict/hrv                                    │
│       → Predict tomorrow's HRV using our XGBoost model         │
│  - GET  /api/ml/insights/{userId}                              │
│       → Generate personalized nutrition recommendations        │
│                                                                  │
│  IN-HOUSE ML MODELS (We train these ourselves):                │
│  - LSTM neural network (RHR prediction)                        │
│  - XGBoost (HRV prediction)                                    │
│  - Correlation engine (Pearson, Spearman, Granger causality)  │
│  - Anomaly detection (Isolation Forest)                        │
│                                                                  │
│  ML Libraries:                                                   │
│  - TensorFlow/PyTorch (deep learning)                          │
│  - scikit-learn (traditional ML)                               │
│  - statsmodels (statistical analysis)                          │
│  - Prophet (time series forecasting)                           │
│                                                                  │
│  NO EXTERNAL ML APIs - All models trained in-house             │
└─────────────────────────────────────────────────────────────────┘
                    ↓ ↑              ↓ ↑
          ┌──────────────────┐  ┌──────────────┐
          │   PostgreSQL     │  │    Redis     │
          │   (User Data)    │  │   (Cache)    │
          │                  │  │              │
          │  - Users         │  │  - Features  │
          │  - Meals         │  │  - Predictions│
          │  - HealthMetric  │  │  - Models    │
          │  - Activity      │  │              │
          │  - MLPrediction  │  │              │
          │  - MLInsight     │  │              │
          └──────────────────┘  └──────────────┘
```

---

## 🎯 User Experience Flow (Mobile App UI)

### Example: Morning Routine

**1. User Opens App (React Native)**
```
┌─────────────────────────────┐
│  Good Morning, John! 👋     │
├─────────────────────────────┤
│  Today's Predictions:        │
│  🫀 RHR: 57 bpm (Good!)     │
│  💚 HRV: 68ms (Excellent)   │
│  😴 Recovery: 85%           │
│                             │
│  📊 [View Details]          │
└─────────────────────────────┘
```

**2. User Taps "View Details"**
```
┌─────────────────────────────┐
│  Today's Forecast            │
├─────────────────────────────┤
│  🫀 Resting Heart Rate      │
│     Predicted: 57 bpm       │
│     Your avg: 59 bpm        │
│     ✅ 2 bpm better!        │
│                             │
│  Why this prediction?       │
│  • Good sleep (7.5h)       │
│  • High protein yesterday  │
│  • No late-night eating    │
│                             │
│  📈 [7-Day Trend Chart]    │
└─────────────────────────────┘
```

**3. User Logs Breakfast**
```
┌─────────────────────────────┐
│  Log Meal                   │
├─────────────────────────────┤
│  🍳 Breakfast               │
│                             │
│  Calories: 450              │
│  Protein: 35g               │
│  Carbs: 40g                 │
│  Fat: 18g                   │
│                             │
│  Time: 8:30 AM              │
│                             │
│  [Save Meal] [Cancel]       │
└─────────────────────────────┘
```

**4. User Views Insights Tab**
```
┌─────────────────────────────┐
│  Your Insights 💡           │
├─────────────────────────────┤
│  🎯 High Priority           │
│                             │
│  Protein Timing Matters     │
│  Your HRV is 12ms higher   │
│  when you eat 30g+ protein │
│  at breakfast.              │
│                             │
│  [Learn More]               │
│                             │
│  ───────────────────────    │
│                             │
│  🌙 Sleep Optimization      │
│  Eating carbs after 8pm    │
│  correlates with +5 bpm    │
│  RHR next morning.          │
│                             │
│  Recommendation: Limit      │
│  carbs to <30g after 7pm   │
│                             │
│  [Try This Week]            │
└─────────────────────────────┘
```

**NO CHAT INTERFACE** - Everything is shown in structured UI components.

---

## 🧠 In-House ML Pipeline (What We're Building)

### Phase 1: Data Collection (✅ COMPLETE)
```
Smartwatch (Apple Health, Fitbit, Garmin, Oura, Whoop)
    ↓
Node.js API (/api/health-metrics POST)
    ↓
PostgreSQL (HealthMetric table)
    ↓
Available for ML analysis
```

### Phase 2: Feature Engineering (⏳ TO BUILD)
```python
# services/feature_engineering.py (WE BUILD THIS)

async def engineer_nutrition_features(user_id: str, date: date) -> dict:
    """
    Transform raw meal data into ML features.
    No external APIs - pure Python computation.
    """
    meals = await db.get_meals(user_id, date, lookback_days=30)

    features = {
        # Daily features
        'protein_daily': sum(m.protein for m in today_meals),
        'carbs_daily': sum(m.carbs for m in today_meals),
        'eating_window': calculate_eating_window(today_meals),

        # Rolling averages (7 days)
        'protein_7d_avg': calculate_rolling_avg(meals, 'protein', 7),
        'calorie_deficit_7d': calculate_deficit(meals, user.tdee, 7),

        # Meal timing
        'first_meal_time': get_first_meal_time(today_meals),
        'last_meal_time': get_last_meal_time(today_meals),
        'late_night_carbs': sum(m.carbs for m in meals if m.hour > 20),

        # Temporal
        'day_of_week': date.weekday(),
        'is_weekend': 1 if date.weekday() >= 5 else 0,
    }

    return features
```

### Phase 3: Correlation Analysis (⏳ TO BUILD)
```python
# services/correlation_engine.py (WE BUILD THIS)

async def analyze_correlations(user_id: str) -> list:
    """
    Find correlations between nutrition and health metrics.
    Uses scipy, statsmodels - no external ML APIs.
    """
    from scipy.stats import pearsonr, spearmanr
    from statsmodels.tsa.stattools import grangercausalitytests

    # Get data
    features = await get_features(user_id, days=30)
    rhr_data = await get_health_metrics(user_id, 'RESTING_HEART_RATE', 30)
    hrv_data = await get_health_metrics(user_id, 'HEART_RATE_VARIABILITY_RMSSD', 30)

    correlations = []

    for feature_name in features.columns:
        # Pearson correlation (linear relationships)
        r_rhr, p_rhr = pearsonr(features[feature_name], rhr_data)
        r_hrv, p_hrv = pearsonr(features[feature_name], hrv_data)

        # Lag analysis (does today's X affect tomorrow's Y?)
        for lag in [0, 1, 2, 3]:
            lagged_feature = features[feature_name].shift(lag)
            r_lag, p_lag = pearsonr(lagged_feature, rhr_data)

            if p_lag < 0.05:  # Statistically significant
                correlations.append({
                    'feature': feature_name,
                    'target': 'RHR',
                    'correlation': r_lag,
                    'p_value': p_lag,
                    'lag_days': lag,
                    'interpretation': generate_interpretation(feature_name, r_lag, lag)
                })

    return correlations
```

### Phase 4: Prediction Models (⏳ TO BUILD)
```python
# services/rhr_predictor.py (WE BUILD THIS)

import torch
import torch.nn as nn

class RHRLSTMModel(nn.Module):
    """LSTM model for RHR prediction."""
    def __init__(self, input_dim, hidden_dim=128):
        super(RHRLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_dim, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

async def train_rhr_model(user_id: str):
    """
    Train LSTM model to predict RHR.
    Model is trained locally - no external ML services.
    """
    # Get training data (90+ days)
    features = await get_features(user_id, days=90)
    rhr_labels = await get_rhr_labels(user_id, days=90)

    # Create sequences (30 days of features → predict day 31)
    X_train, y_train = create_sequences(features, rhr_labels, sequence_length=30)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)

    # Build LSTM model (WE DESIGN THIS)
    num_features = X_train.shape[2]
    model = RHRLSTMModel(input_dim=num_features, hidden_dim=128)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model (LOCAL TRAINING - no cloud ML)
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Save model locally
    torch.save(model.state_dict(), f'./app/ml_models/rhr_lstm_{user_id}_v1.pt')

    return model

async def predict_rhr(user_id: str, date: date) -> dict:
    """
    Predict tomorrow's RHR using trained LSTM model.
    """
    # Load user's model
    model = RHRLSTMModel(input_dim=num_features)
    model.load_state_dict(torch.load(f'./app/ml_models/rhr_lstm_{user_id}_v1.pt'))
    model.eval()

    # Get last 30 days of features
    features = await get_features(user_id, date, lookback_days=30)

    # Predict
    X = prepare_input(features)
    predicted_rhr = model.predict(X)[0][0]

    # Calculate confidence interval
    confidence = calculate_confidence(model, X)

    return {
        'predicted_value': float(predicted_rhr),
        'confidence': float(confidence),
        'lower_bound': float(predicted_rhr - 2),
        'upper_bound': float(predicted_rhr + 2),
    }
```

---

## 🔧 Technology Stack

### Mobile App (React Native)
- **Framework**: React Native + Expo
- **State**: Redux or Zustand
- **Charts**: react-native-charts or Victory Native
- **UI**: React Native Paper or NativeBase
- **Auth**: AsyncStorage + JWT

### Backend (Node.js)
- **Framework**: Express + TypeScript
- **ORM**: Prisma
- **Database**: PostgreSQL 16
- **Validation**: Zod
- **Auth**: JWT (jsonwebtoken)

### ML Service (Python) - IN-HOUSE ONLY
- **Framework**: FastAPI
- **Database**: SQLAlchemy (async)
- **Cache**: Redis (aioredis)
- **ML Libraries**:
  - `scikit-learn` - Traditional ML (correlation, regression)
  - `PyTorch` - Deep learning (LSTM neural networks)
  - `statsmodels` - Statistical analysis (Granger causality)
  - `scipy` - Scientific computing (Pearson, Spearman)
  - `Prophet` - Time series forecasting (Facebook's library)
  - `numpy`, `pandas` - Data manipulation

### Infrastructure
- **Database**: PostgreSQL 16
- **Cache**: Redis 7
- **Containers**: Docker + docker-compose
- **CI/CD**: (TBD - GitHub Actions?)
- **Hosting**: (TBD - AWS, DigitalOcean, or self-hosted?)

---

## 🚫 What We're NOT Using

- ❌ OpenAI API (GPT-4, ChatGPT)
- ❌ Anthropic API (Claude)
- ❌ Google AI APIs (Gemini, Vertex AI)
- ❌ AWS ML services (SageMaker, Comprehend)
- ❌ Azure ML services
- ❌ Any third-party ML APIs
- ❌ Chat interfaces or conversational AI
- ❌ LLMs for predictions (we use traditional ML + deep learning)

---

## ✅ What We're Building In-House

- ✅ Custom LSTM neural networks (PyTorch)
- ✅ Custom XGBoost models
- ✅ Custom correlation engine (scipy, statsmodels)
- ✅ Custom feature engineering pipeline
- ✅ Custom anomaly detection (Isolation Forest)
- ✅ Custom recommendation engine
- ✅ Mobile app UI (React Native)
- ✅ REST APIs (Node.js + Python)

---

## 📱 Mobile App Screens (UI, Not Chat)

### 1. Dashboard
- Today's nutrition summary (calories, macros)
- Health metrics (RHR, HRV from smartwatch)
- Quick log meal button

### 2. Nutrition Tab
- Log meal form
- Daily/weekly nutrition charts
- Meal history

### 3. Health Tab
- RHR chart (7-day, 30-day, 90-day)
- HRV chart
- Sleep quality chart
- Sync smartwatch button

### 4. Insights Tab (ML-Powered)
- Personalized recommendations
- Correlation insights ("Protein → Better HRV")
- Anomaly alerts ("RHR elevated - consider rest day")

### 5. Predictions Tab (ML-Powered)
- Tomorrow's RHR forecast
- Tomorrow's HRV forecast
- "What-if" scenarios ("What if I eat X?")

### 6. Profile Tab
- User settings
- Goals (target weight, target RHR)
- Smartwatch connections

---

## 🔄 Development Workflow

### Phase 0 (✅ COMPLETE)
- Database schema
- Node.js CRUD APIs
- Python ML service foundation

### Phase 1 (⏳ NEXT)
- Feature engineering service
- Correlation analysis
- Basic insights in mobile app

### Phase 2
- Train LSTM model (RHR prediction)
- Train XGBoost model (HRV prediction)
- Predictions in mobile app

### Phase 3
- Recommendation engine
- "What-if" scenarios
- Goal tracking

### Phase 4
- Anomaly detection
- Population-level insights
- Advanced visualizations

---

## 🎯 Key Points for Future Sessions

**REMEMBER**:
1. This is a **mobile app** (React Native), not a chatbot
2. ML models are **trained in-house** using scikit-learn, TensorFlow, PyTorch
3. **No external ML APIs** - everything is custom-built
4. Health metrics come from **smartwatches** (Apple Health, Fitbit, etc.)
5. ML analyzes: **Nutrition/Activity → Health Metrics** (not the other way around)
6. UI is **structured screens** (charts, forms, lists), not conversational

**Architecture in one sentence**:
> A React Native mobile app that uses in-house ML models (LSTM, XGBoost, correlation analysis) to analyze how nutrition and eating schedules affect health metrics from smartwatches, then provides personalized recommendations through a structured mobile UI.

---

**Last Updated**: 2025-01-17
**Status**: Phase 0 Complete, Phase 1 Ready to Begin
