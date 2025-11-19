# ML Engine Quick Start Guide

**Created**: 2025-01-17
**Main Document**: See `ML_ENGINE_COMPREHENSIVE_PLAN.md` for full details (1,000+ lines)

---

## 🚨 IMPORTANT - Project Type

**Mobile App + In-House ML (NOT Chatbot + NOT External Services)**

- ❌ **NOT** a chatbot or conversational AI
- ❌ **NOT** using OpenAI, Claude, AWS ML, Google AI, Azure ML, or any external ML APIs
- ✅ **IS** a React Native mobile app (Expo)
- ✅ **IS** building ML models ourselves (PyTorch, scikit-learn, XGBoost)
- ✅ **IS** training models in-house on our infrastructure

**Data Flow**:
- **INPUTS** (what we control): Nutrition (meals, timing) + Activity (workouts)
- **OUTPUTS** (what we optimize): Health metrics from smartwatches (RHR, HRV, sleep)
- **GOAL**: Understand how nutrition/eating schedules affect health metrics

---

## 🎯 What We're Building

A **personalized ML engine** (built in-house) that learns how YOUR body responds to nutrition and activity, then:
- Predicts tomorrow's health metrics (RHR, HRV, recovery)
- Finds patterns unique to you (not generic advice)
- Recommends optimal nutrition for YOUR goals
- Alerts you to anomalies (overtraining, illness)

---

## 🚀 Quick Overview

### The Problem We Solve

**Traditional nutrition tracking**: "You ate 2,000 calories" ❌

**Our ML engine**: "Your high-carb dinners (>100g) correlate with +5 bpm RHR the next morning. Try reducing evening carbs for better recovery." ✅

### Core Capabilities

1. **Correlation Analysis**: "When you eat X, Y happens"
2. **Predictions**: "Based on today's nutrition, your RHR tomorrow will be 58±2 bpm"
3. **Recommendations**: "For optimal recovery, aim for 160g protein, 250g carbs"
4. **Anomaly Detection**: "Your RHR is 7 bpm higher than normal - consider a rest day"

---

## 📊 New Data Models (6 Tables)

| Table | Purpose | Key Fields |
|-------|---------|------------|
| **HealthMetric** | RHR, HRV, sleep, steps, etc. | metricType, value, unit, recordedAt, source |
| **Activity** | Workouts, exercise | activityType, intensity, duration, calories |
| **MLFeature** | Pre-computed features for fast predictions | category, features (JSON), date, version |
| **MLPrediction** | Track predictions vs. actuals | targetMetric, predictedValue, actualValue, confidence |
| **MLInsight** | User-facing insights | insightType, title, description, recommendation |
| **UserMLProfile** | Per-user ML configuration | dataQualityScore, modelsAvailable, enablePredictions |

---

## 🧠 ML Models (5 Core Models)

### 1. RHR Prediction (LSTM)
- **Input**: 30 days of nutrition + activity + health data
- **Output**: Tomorrow's resting heart rate
- **Target accuracy**: MAE < 3 bpm

### 2. HRV Prediction (LSTM)
- **Input**: Same as RHR + sleep quality emphasis
- **Output**: Tomorrow's heart rate variability
- **Target accuracy**: MAE < 10ms

### 3. Correlation Engine
- **Methods**: Pearson, Spearman, Granger causality
- **Output**: "Feature X correlates with metric Y (r=0.65, p=0.003)"
- **Lag analysis**: 0h, 6h, 12h, 24h, 48h

### 4. Anomaly Detection (Isolation Forest)
- **Input**: Current metrics vs. user's baseline
- **Output**: Alerts for unusual patterns
- **Use cases**: Illness detection, overtraining alerts

### 5. Recommendation Engine
- **Input**: User's goals + historical best days
- **Output**: Optimal macros for tomorrow
- **Methods**: Multi-objective optimization

---

## 🛠️ Tech Stack

### Backend (Node.js)
- ✅ **Existing**: Express, TypeScript, Prisma, PostgreSQL, JWT
- ➕ **New**: Redis (caching), BullMQ (job queue)

### ML Service (Python) - IN-HOUSE ONLY
- **Framework**: FastAPI
- **ML Libraries** (all in-house, no external APIs):
  - **PyTorch** (deep learning - LSTM neural networks)
  - scikit-learn (traditional ML - correlation, regression)
  - XGBoost (gradient boosting for HRV prediction)
  - Prophet (time series forecasting - Facebook's open-source library)
  - statsmodels (statistical analysis - Granger causality)
  - scipy (scientific computing - Pearson/Spearman correlation)
- **Tools**: Pandas, NumPy, SHAP (explainability)
- **Infra**: Redis cache, SQLAlchemy, Uvicorn
- **Training**: All models trained locally on our infrastructure

### Database
- **PostgreSQL**: Main DB (existing)
- **TimescaleDB**: Time series extension (optimized for health metrics)
- **Redis**: Feature cache, prediction cache

---

## 📈 Implementation Phases (3-4 Months)

### Phase 0: Foundation (Week 1-2)
**Goal**: Infrastructure setup
- ✅ Add 6 new database tables
- ✅ Integrate Apple Health, Fitbit, Garmin APIs
- ✅ Set up Python ML service (FastAPI)
- ✅ Start collecting 30 days of health data

**Deliverable**: 100 users with connected wearables

---

### Phase 1: Correlation Analysis (Week 3-4)
**Goal**: Find patterns (no predictions yet)
- ✅ Feature engineering pipeline
- ✅ Correlation engine (Pearson, Spearman)
- ✅ Insight generation (template-based)
- ✅ Frontend "Insights" tab

**Deliverable**: "Your protein intake correlates with HRV (r=0.68)"

---

### Phase 2: Prediction Models (Week 5-8)
**Goal**: Predict tomorrow's RHR, HRV
- ✅ Train LSTM models
- ✅ Evaluate accuracy (MAE, R²)
- ✅ Deploy prediction API
- ✅ Frontend "Tomorrow's Forecast"

**Deliverable**: "Your RHR tomorrow: 58±2 bpm"

---

### Phase 3: Recommendations (Week 9-10)
**Goal**: Suggest optimal nutrition
- ✅ Recommendation engine
- ✅ "What-if" scenario testing
- ✅ Multi-objective optimization
- ✅ Frontend recommendations UI

**Deliverable**: "For best recovery: 160g protein, 250g carbs, 60g fat"

---

### Phase 4: Anomaly Detection (Week 11-12)
**Goal**: Alert users to unusual patterns
- ✅ Baseline calculation
- ✅ Real-time anomaly detection
- ✅ Multi-metric alerts (RHR up + HRV down)
- ✅ Notification system

**Deliverable**: "Your RHR is 7 bpm above baseline - consider rest"

---

### Phase 5: Advanced Features (Week 13+)
**Goal**: Cutting-edge personalization
- ✅ Multi-step forecasting (7-day predictions)
- ✅ Causal inference (not just correlation)
- ✅ Reinforcement learning (optimal nutrition agent)
- ✅ Population-level insights

**Deliverable**: "Users like you find X effective"

---

## 🎨 Feature Engineering (50+ Features)

### Nutrition Features
```
protein_daily, carbs_daily, calories_daily
protein_7d_avg, carbs_7d_avg
meal_regularity, eating_window
protein_ratio, carb_ratio, fat_ratio
late_night_carbs, first_meal_time
```

### Activity Features
```
steps_daily, active_minutes_daily
workout_intensity, recovery_time
steps_7d_avg, calories_burned
```

### Health Features (Lagged)
```
rhr_yesterday, rhr_7d_avg, rhr_trend
hrv_yesterday, hrv_7d_avg
sleep_duration_last, sleep_quality_last
```

### Temporal Features
```
day_of_week, is_weekend
days_since_rest, cycle_phase
```

### Interaction Features
```
protein_per_kg (protein / body weight)
carbs_per_activity (carbs / active minutes)
calorie_balance (calories - TDEE)
```

---

## 📡 API Endpoints (Examples)

### Correlations
```bash
GET /ml/correlations?userId=123&metric=RHR&days=30
```
**Response**: List of features correlated with RHR

### Predictions
```bash
POST /ml/predict/rhr
{
  "userId": "123",
  "targetDate": "2025-01-18"
}
```
**Response**: Predicted RHR with confidence interval

### Insights
```bash
GET /ml/insights?userId=123&limit=5&priority=HIGH
```
**Response**: Top 5 actionable insights

### Feature Importance
```bash
GET /ml/explain?userId=123&predictionId=abc
```
**Response**: "Sleep quality (35%), protein (18%), workout intensity (12%)"

---

## 🔒 Privacy & Security

**Critical for health data**:
- ✅ Encryption at rest (database-level AES-256)
- ✅ Encryption in transit (TLS 1.3)
- ✅ User consent (explicit opt-in)
- ✅ Data retention policies (user-controlled)
- ✅ Anonymization for research
- ✅ GDPR/HIPAA compliance

**User controls**:
- Opt-in to ML features (default: off)
- View all data
- Delete any data point
- Export data (JSON)
- Disconnect wearables
- Delete account (permanent deletion)

---

## 📊 Success Metrics

### Product Metrics
- **Engagement**: % users enabling ML features (target: >60%)
- **Retention**: 30-day retention with ML (target: 2x baseline)
- **Satisfaction**: "Was this helpful?" (target: >70% yes)

### ML Metrics
- **RHR prediction**: MAE < 3 bpm, R² > 0.6
- **HRV prediction**: MAE < 10ms, R² > 0.6
- **Insight quality**: p-value < 0.05 for >80% of insights

### Business Metrics
- **Conversion**: Free → Paid (if ML is premium)
- **ARPU**: Revenue per user
- **LTV**: Lifetime value

---

## 💰 Pricing Strategy (If Premium)

### Free Tier
- Basic insights (top 3 correlations)
- Weekly summary
- 30-day data retention

### Premium Tier ($9.99/month)
- All insights
- Daily predictions (RHR, HRV, sleep)
- Personalized recommendations
- Anomaly alerts
- 365-day data retention
- Data export

---

## 🚀 Next Steps (Week 1)

### Day 1-2: Database Schema
1. Design Prisma schema for 6 new tables
2. Write migration scripts
3. Test migrations on staging

### Day 3-4: Health Data Integration
1. Apple Health integration (HealthKit)
2. Fitbit API setup
3. Manual entry UI

### Day 5-7: ML Service Setup
1. FastAPI project structure
2. PostgreSQL connection
3. `/health` endpoint
4. Feature engineering skeleton

---

## 📚 Resources

### Documentation
- **Full plan**: `ML_ENGINE_COMPREHENSIVE_PLAN.md` (1,000+ lines)
- **API contracts**: See Appendix A in main plan
- **Data requirements**: See Appendix B in main plan

### Codebase Analysis
- **Architecture**: `CODEBASE_ANALYSIS.md`
- **Quick reference**: `ML_INTEGRATION_QUICK_REFERENCE.md`
- **Exploration**: `EXPLORATION_SUMMARY.txt`

---

## ❓ FAQ

**Q: Do we need a data scientist?**
A: Recommended for Phase 2+. Phase 0-1 can be done by backend engineers with ML knowledge.

**Q: How much data do users need before ML works?**
A: Minimum 30 days for correlations, 90 days for predictions.

**Q: What if users don't have wearables?**
A: Manual entry UI for RHR, HRV, sleep. Lower data quality but still works.

**Q: Can we use existing libraries?**
A: Yes! Prophet (Facebook), SHAP (Microsoft), scikit-learn (open source).

**Q: What about model retraining?**
A: Daily retraining pipeline (Apache Airflow). Models improve as more data collected.

**Q: Privacy concerns?**
A: Explicit opt-in, encryption, GDPR/HIPAA compliance, user data deletion controls.

---

## 🎯 Bottom Line

**This ML engine will transform Nutri from a nutrition tracker into a personalized health coach.**

**Timeline**: 3-4 months
**Complexity**: High (but achievable)
**Impact**: Game-changing (2x engagement, 3x retention)

**Let's build it!** 🚀
