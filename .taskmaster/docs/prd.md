# Nutri - Product Requirements Document

## Executive Summary

**Nutri** is a full-stack nutrition tracking mobile application with an in-house ML engine for analyzing how nutrition affects health metrics. The app helps users track meals, calories, macronutrients, and correlate dietary patterns with health metrics from wearables (RHR, HRV, sleep, recovery).

### Vision

Provide personalized, AI-powered nutrition insights by analyzing how food choices impact health outcomes, enabling users to optimize their diet for better energy, recovery, and overall wellness.

### Current State

- **Mobile App**: Core features complete (auth, meal tracking, camera UI)
- **Backend API**: Production-ready with full CRUD, auth, and security
- **ML Service**: Feature engineering and correlation analysis complete; predictions pending

---

## Technical Architecture

### Stack Overview

| Layer | Technology |
|-------|------------|
| Mobile | React Native 0.76 + Expo 52, TypeScript, Expo Router |
| Backend | Node.js + Express, TypeScript, Prisma, PostgreSQL 16 |
| ML Service | Python 3.9+, FastAPI, PyTorch, scikit-learn, XGBoost |
| Cache | Redis |
| Auth | JWT with bcryptjs |

### Repository Structure

```
nutri/
├── app/                    # Mobile app (Expo Router)
├── lib/                    # Shared mobile libraries
├── server/                 # Backend API (Express + Prisma)
├── ml-service/             # Python ML service (FastAPI)
└── .taskmaster/            # Task management
```

---

## Features

### Phase 1: Core Features (Complete)

#### 1.1 User Authentication
- **Status**: Complete
- Email/password registration and login
- JWT token management with secure storage
- Password reset flow with email tokens
- Profile management (goals, physical stats)

#### 1.2 Meal Tracking
- **Status**: Complete
- Manual meal entry with full macro tracking
- Meal types: breakfast, lunch, dinner, snack
- Nutrition fields: calories, protein, carbs, fat, fiber, sugar
- Daily and weekly summaries
- Photo attachment support

#### 1.3 Backend API
- **Status**: Complete
- RESTful API with Express.js
- Prisma ORM with PostgreSQL
- Zod validation on all endpoints
- Rate limiting and input sanitization
- JWT authentication middleware

### Phase 2: ML Foundation (Partially Complete)

#### 2.1 Feature Engineering
- **Status**: Complete
- 50+ computed ML features across domains
- Nutrition features: rolling averages, macro ratios, consistency
- Activity features: intensity patterns, recovery metrics
- Health features: HRV trends, sleep quality, stress
- Redis caching with 1-hour TTL

#### 2.2 Correlation Analysis
- **Status**: Complete
- Pearson, Spearman, Kendall correlations
- Granger causality for causal relationships
- Lag analysis (0-30 day time delays)
- Statistical significance testing with FDR correction

#### 2.3 Food Scanning
- **Status**: Partially Complete
- Camera UI with photo capture: Complete
- Image upload and processing: Complete
- Food classification: Mock (needs real ML model)
- Nutrition database lookup: Complete (50+ foods)
- AR portion measurement: Not started

### Phase 3: Health Integration (API Ready, UI Pending)

#### 3.1 Health Metrics API
- **Status**: Backend complete, Mobile UI not started
- 30+ metric types supported
- Sources: Apple Health, Fitbit, Garmin, Oura, Whoop, Manual
- Bulk sync for wearable data
- Time series and statistics endpoints

#### 3.2 Activity Tracking API
- **Status**: Backend complete, Mobile UI not started
- 21 activity types with intensity levels
- Duration, calories, heart rate tracking
- Recovery time calculations
- Daily/weekly activity summaries

### Phase 4: Predictions & Insights (Not Started)

#### 4.1 Health Predictions
- **Status**: Architecture ready, not implemented
- LSTM models for RHR and HRV predictions
- Multi-horizon forecasting (1-7 days)
- Confidence intervals and model uncertainty

#### 4.2 ML Insights
- **Status**: Schema ready, logic not implemented
- Pattern detection in nutrition-health relationships
- Personalized recommendations based on correlations
- Anomaly detection for health metrics
- Goal progress tracking

---

## User Stories

### Authentication & Onboarding

```
US-1: As a new user, I want to create an account with email and password
      so that I can start tracking my nutrition.

US-2: As a returning user, I want to log in with my credentials
      so that I can access my historical data.

US-3: As a user who forgot my password, I want to reset it via email
      so that I can regain access to my account.

US-4: As a user, I want to set my daily nutrition goals (calories, macros)
      so that the app can track my progress against targets.

US-5: As a user, I want to update my physical stats (weight, height, activity level)
      so that the app can provide accurate recommendations.
```

### Meal Tracking

```
US-6: As a user, I want to manually log a meal with nutrition info
      so that I can track what I eat throughout the day.

US-7: As a user, I want to see my daily calorie and macro progress
      so that I know how much I can still eat today.

US-8: As a user, I want to view my meals organized by type (breakfast, lunch, etc.)
      so that I can review my eating patterns.

US-9: As a user, I want to see weekly nutrition summaries
      so that I can understand my eating trends over time.

US-10: As a user, I want to attach photos to my meals
       so that I can remember what I ate and potentially use food scanning.
```

### Food Scanning (Phase 2)

```
US-11: As a user, I want to take a photo of my food
       so that the app can identify it and estimate nutrition.

US-12: As a user, I want to see food classification with confidence scores
       so that I can verify the app identified my food correctly.

US-13: As a user, I want to see alternative food suggestions
       so that I can select the correct item if the primary guess is wrong.

US-14: As a user, I want the scanned food nutrition to auto-fill the meal form
       so that I can quickly log meals without manual data entry.

US-15: As a user, I want to use AR to measure portion size
       so that the app can more accurately estimate nutrition.
```

### Health Metrics Integration (Phase 3)

```
US-16: As a user, I want to sync health data from my wearable device
       so that the app has accurate RHR, HRV, and sleep data.

US-17: As a user, I want to view my health metrics over time
       so that I can see trends in my health data.

US-18: As a user, I want to manually log health metrics
       so that I can track data not available from wearables.

US-19: As a user, I want to see time series charts of my health data
       so that I can visualize patterns and changes.
```

### Activity Tracking (Phase 3)

```
US-20: As a user, I want to log my workouts and activities
       so that the app can factor exercise into my nutrition needs.

US-21: As a user, I want to sync activities from fitness apps
       so that my exercise data is automatically imported.

US-22: As a user, I want to see my activity summary by day and week
       so that I understand my exercise patterns.

US-23: As a user, I want to see recovery recommendations
       so that I know when to take rest days.
```

### ML Insights & Predictions (Phase 4)

```
US-24: As a user, I want to see how my nutrition affects my health metrics
       so that I can make informed dietary decisions.

US-25: As a user, I want to receive personalized nutrition recommendations
       so that I can optimize my diet for better health outcomes.

US-26: As a user, I want to see predictions for my future health metrics
       so that I can understand how today's choices affect tomorrow.

US-27: As a user, I want to be alerted to anomalies in my health data
       so that I can take action if something seems off.

US-28: As a user, I want to see which foods have the biggest impact on my RHR
       so that I can adjust my diet for better heart health.

US-29: As a user, I want to see my goal progress with AI-generated insights
       so that I stay motivated and informed.
```

---

## Technical Requirements

### Performance Requirements

| Metric | Requirement |
|--------|-------------|
| API Response Time | < 200ms for standard endpoints |
| ML Feature Computation | < 5s with Redis cache |
| Food Classification | < 3s per image |
| Mobile App Launch | < 2s to interactive |
| Database Queries | Indexed for common patterns |

### Security Requirements

- JWT tokens with 7-day expiry
- Bcryptjs password hashing (10 rounds)
- Input sanitization for XSS prevention
- Rate limiting (100 req/15min API, 5 req/15min auth)
- HTTPS only in production
- No sensitive data in logs

### Data Requirements

- 30+ health metric types supported
- 21 activity types with intensity levels
- 50+ engineered ML features
- Wearable sync from 5+ sources
- 50+ foods in nutrition database (expandable)

### Mobile Requirements

- iOS 14+ and Android 10+
- Offline mode for meal logging (future)
- Background sync for wearables (future)
- Push notifications for insights (future)

---

## Development Roadmap

### Sprint 1: Food Classification ML Model

**Goal**: Replace mock food classifier with real ML model

Tasks:
1. Implement ResNet/EfficientNet food classification model
2. Train on food image dataset (Food-101 or custom)
3. Deploy model to ML service
4. Update food analysis endpoint
5. Add model versioning and A/B testing capability
6. Integration testing with mobile app

### Sprint 2: Health Metrics Mobile UI

**Goal**: Build mobile screens for health data visualization

Tasks:
1. Create health metrics list screen
2. Implement health metric detail view with charts
3. Build manual health metric entry form
4. Add time range filters (day, week, month)
5. Create health dashboard summary widget
6. Add pull-to-refresh and loading states

### Sprint 3: Activity Tracking Mobile UI

**Goal**: Build mobile screens for activity tracking

Tasks:
1. Create activity list screen
2. Implement activity detail view
3. Build manual activity entry form
4. Add activity type filters
5. Create weekly activity summary view
6. Show recovery recommendations

### Sprint 4: Wearable Integration

**Goal**: Enable sync from health platforms

Tasks:
1. Implement Apple HealthKit integration
2. Add background sync capability
3. Create sync status UI
4. Handle data conflicts and deduplication
5. Build sync history/logs view
6. Add sync frequency settings

### Sprint 5: ML Predictions

**Goal**: Deploy LSTM models for health predictions

Tasks:
1. Train LSTM model for RHR prediction
2. Train LSTM model for HRV prediction
3. Implement prediction API endpoints
4. Create prediction visualization UI
5. Add confidence intervals display
6. Build prediction accuracy tracking

### Sprint 6: ML Insights & Recommendations

**Goal**: Generate actionable insights from correlations

Tasks:
1. Implement insight generation logic
2. Create insight card components
3. Build recommendations engine
4. Add anomaly detection alerts
5. Create insights feed/dashboard
6. Implement user feedback collection

### Sprint 7: AR Portion Measurement

**Goal**: Use camera AR to estimate food portions

Tasks:
1. Integrate ARKit/ARCore for dimension measurement
2. Implement portion size estimation algorithm
3. Connect AR data to nutrition calculation
4. Update food scanning flow
5. Add calibration for accuracy
6. User testing and refinement

### Sprint 8: Polish & Optimization

**Goal**: Production readiness and performance

Tasks:
1. Implement offline mode with data sync
2. Add push notifications for insights
3. Performance optimization pass
4. Accessibility audit and fixes
5. Generate API documentation (OpenAPI/Swagger)
6. Security audit and penetration testing

---

## Database Schema Summary

### Core Models

| Model | Purpose | Key Fields |
|-------|---------|------------|
| User | Authentication & profile | email, goals, physical stats |
| Meal | Nutrition tracking | calories, macros, mealType, consumedAt |
| HealthMetric | Wearable data | metricType (30+ types), value, source |
| Activity | Exercise tracking | activityType (21 types), duration, intensity |

### ML Models

| Model | Purpose | Key Fields |
|-------|---------|------------|
| MLFeature | Pre-computed features | category, features JSON, version |
| MLPrediction | Model outputs | targetMetric, predictedValue, confidence |
| MLInsight | Recommendations | insightType, title, recommendation |
| UserMLProfile | Per-user ML config | modelsAvailable, preferences |

---

## Success Metrics

### Engagement Metrics

| Metric | Target |
|--------|--------|
| Daily Active Users | Track growth |
| Meals logged per user per day | 2-3 |
| Health metrics synced per week | 50+ per user |
| Time spent in app per session | > 3 minutes |

### ML Performance Metrics

| Metric | Target |
|--------|--------|
| Food classification accuracy | > 80% top-5 |
| RHR prediction MAE | < 3 bpm |
| HRV prediction MAE | < 10 ms |
| Insight helpfulness rating | > 70% positive |

### Technical Metrics

| Metric | Target |
|--------|--------|
| API uptime | > 99.5% |
| Average API response time | < 200ms |
| Mobile crash rate | < 1% |
| Test coverage | > 80% |

---

## Risk Assessment

### Technical Risks

| Risk | Mitigation |
|------|------------|
| ML model accuracy insufficient | Start with ensemble models, iterate on training data |
| Wearable API changes | Abstract integration layer, monitor deprecations |
| Performance at scale | Redis caching, database indexing, async processing |
| Mobile battery drain | Optimize sync frequency, batch operations |

### Business Risks

| Risk | Mitigation |
|------|------------|
| Low user engagement | Focus on quick wins (food scanning), gamification |
| Data privacy concerns | Transparent policies, data export, deletion options |
| Competition from established apps | Differentiate with ML insights |

---

## Appendix

### Health Metric Types (30+)

- Cardiovascular: RESTING_HEART_RATE, HEART_RATE_VARIABILITY_SDNN, HEART_RATE_VARIABILITY_RMSSD, BLOOD_PRESSURE_SYSTOLIC, BLOOD_PRESSURE_DIASTOLIC
- Sleep: SLEEP_DURATION, SLEEP_DEEP, SLEEP_LIGHT, SLEEP_REM, SLEEP_AWAKE, SLEEP_EFFICIENCY, SLEEP_LATENCY
- Fitness: VO2_MAX, STEPS, ACTIVE_CALORIES, TOTAL_CALORIES, FLOORS_CLIMBED, DISTANCE
- Body: WEIGHT, BODY_FAT_PERCENTAGE, MUSCLE_MASS, BONE_MASS, HYDRATION_LEVEL, BMI
- Recovery: RECOVERY_SCORE, READINESS_SCORE, STRESS_LEVEL, RESPIRATORY_RATE, BLOOD_OXYGEN
- Other: BODY_TEMPERATURE, MENSTRUAL_CYCLE, CAFFEINE_INTAKE, ALCOHOL_INTAKE

### Activity Types (21)

- Cardio: RUNNING, CYCLING, SWIMMING, WALKING, HIKING, ELLIPTICAL, ROWING, STAIR_CLIMBING, JUMP_ROPE, DANCING
- Strength: WEIGHTLIFTING, CROSSFIT, CALISTHENICS
- Flexibility: YOGA, PILATES, STRETCHING
- Sports: TENNIS, BASKETBALL, SOCCER, MARTIAL_ARTS
- Other: OTHER

### Data Sources

- APPLE_HEALTH
- FITBIT
- GARMIN
- OURA
- WHOOP
- STRAVA
- MANUAL

---

*Document Version: 1.0*
*Last Updated: 2025-12-04*
