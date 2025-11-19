# Nutri ML Service 🧠

**In-House Machine Learning API for personalized nutrition insights**

Version: 1.0.0
Framework: FastAPI + SQLAlchemy + Redis
Python: 3.11+

---

## 🚨 IMPORTANT - In-House ML Only

**All ML models are built, trained, and deployed IN-HOUSE**

- ✅ **We build** all ML models using open-source libraries (PyTorch, scikit-learn)
- ✅ **We train** all models on our infrastructure with user data
- ✅ **We own** all models and intellectual property
- ❌ **NOT using** external ML APIs (OpenAI, Claude, AWS ML, Google AI, Azure ML, etc.)
- ❌ **NOT a chatbot** - This service provides structured data/predictions, not conversational AI

**ML Stack (All In-House)**:
- **PyTorch** (LSTM neural networks for time series prediction)
- scikit-learn (correlation analysis, regression, Isolation Forest)
- XGBoost (gradient boosting for HRV prediction)
- statsmodels (Granger causality, statistical tests)
- scipy (Pearson/Spearman correlation)
- Prophet (Facebook's time series library)

---

## 🎯 What This Service Does

The Nutri ML Service analyzes the relationship between:
- **Inputs** (what we control): Nutrition (meals, timing, macros), Activity (workouts, intensity)
- **Outputs** (what we optimize): Health metrics from smartwatches (RHR, HRV, sleep, recovery)

**Goal**: Understand how nutrition and eating schedules affect health metrics to provide personalized recommendations.

**Core Capabilities**:
1. **Feature Engineering**: Transforms raw data into 50+ ML features
2. **Correlation Analysis**: Finds patterns (e.g., "high protein → better HRV")
3. **Predictions**: Forecasts tomorrow's RHR, HRV using our LSTM models
4. **Insights**: Generates actionable recommendations based on user's data

---

## 📁 Project Structure

```
ml-service/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Configuration (env variables)
│   ├── database.py             # Async PostgreSQL connection
│   ├── redis_client.py         # Redis caching layer
│   ├── models/                 # SQLAlchemy models
│   │   ├── user.py
│   │   ├── meal.py
│   │   ├── health_metric.py
│   │   └── activity.py
│   ├── schemas/                # Pydantic schemas (TODO)
│   ├── services/               # Business logic (TODO)
│   │   ├── feature_engineering.py
│   │   ├── correlation_engine.py
│   │   └── prediction_service.py
│   └── api/                    # API routes (TODO)
│       ├── ml.py
│       └── correlations.py
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Local development stack
├── .env.example                # Environment variables template
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 16+ (or use Docker)
- Redis 7+ (or use Docker)

### Option 1: Local Development (Python Virtual Environment)

```bash
# 1. Create virtual environment
cd ml-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment variables
cp .env.example .env
# Edit .env with your database credentials

# 4. Run the service
python -m app.main
# Or with uvicorn:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker (Recommended)

```bash
# Start entire stack (PostgreSQL + Redis + ML Service)
docker-compose up -d

# View logs
docker-compose logs -f ml-service

# Stop stack
docker-compose down
```

---

## 🏥 Health Checks

Once running, verify the service:

```bash
# Root endpoint
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# API documentation (Swagger UI)
open http://localhost:8000/docs
```

**Expected response**:
```json
{
  "status": "healthy",
  "service": "Nutri ML Service",
  "version": "1.0.0",
  "environment": "development",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy"
  }
}
```

---

## 🔧 Configuration

All configuration is done via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `ENVIRONMENT` | `development` | Environment (development/staging/production) |
| `DEBUG` | `true` | Enable debug mode and SQL logging |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `CACHE_TTL_FEATURES` | `3600` | Feature cache TTL (1 hour) |
| `CACHE_TTL_PREDICTIONS` | `86400` | Prediction cache TTL (24 hours) |
| `MIN_DATA_POINTS_FOR_ML` | `30` | Minimum days of data required for ML |

---

## 🧪 API Endpoints (Planned)

### Health & Status
- `GET /` - Service information
- `GET /health` - Health check with dependencies
- `GET /ready` - Readiness check for load balancers

### Features (TODO)
- `POST /api/features/engineer` - Engineer features for user
- `GET /api/features/{userId}/{date}` - Get cached features

### Correlations (TODO)
- `GET /api/correlations/{userId}` - Get correlations for all metrics
- `GET /api/correlations/{userId}/{metricType}` - Correlations for specific metric

### Predictions (TODO)
- `POST /api/predictions/rhr` - Predict tomorrow's RHR
- `POST /api/predictions/hrv` - Predict tomorrow's HRV
- `GET /api/predictions/{userId}/{date}` - Get cached predictions

### Insights (TODO)
- `GET /api/insights/{userId}` - Get personalized insights
- `POST /api/insights/{userId}/generate` - Generate new insights

---

## 🗄️ Database Models

The ML service reads from the same PostgreSQL database as the Node.js backend:

**Tables used**:
- `User` - User profile and goals
- `Meal` - Nutrition tracking
- `HealthMetric` - RHR, HRV, sleep, steps, etc.
- `Activity` - Workouts and exercise

**Tables created** (Phase 1):
- `MLFeature` - Pre-computed features (for fast predictions)
- `MLPrediction` - Model outputs and tracking
- `MLInsight` - User-facing insights

---

## 🧠 ML Pipeline (Overview)

```
┌─────────────┐
│  Raw Data   │  ← Meals, Activities, Health Metrics from DB
└─────────────┘
       ↓
┌─────────────────────────────────────────┐
│  Feature Engineering Service            │
│  - Nutrition features (protein_7d_avg)  │
│  - Activity features (recovery_time)    │
│  - Health features (rhr_trend)          │
│  - Temporal features (day_of_week)      │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│  Redis Cache (1h TTL)                   │
│  Key: features:{userId}:{date}:daily    │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│  ML Models                               │
│  - LSTM for RHR prediction              │
│  - XGBoost for HRV prediction           │
│  - Correlation engine                    │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│  Redis Cache (24h TTL)                  │
│  Key: prediction:{userId}:{metric}:date │
└─────────────────────────────────────────┘
       ↓
┌─────────────┐
│  API Response│  → Returned to client
└─────────────┘
```

---

## 📊 Cache Strategy

**Redis is used for**:
1. **Engineered features** (TTL: 1 hour)
   - Key pattern: `features:{userId}:{date}:{category}`
   - Reduces DB queries and computation time

2. **Predictions** (TTL: 24 hours)
   - Key pattern: `prediction:{userId}:{metricType}:{date}`
   - Predictions are expensive, cache aggressively

3. **Model artifacts** (TTL: 7 days)
   - Key pattern: `model:{modelId}:{version}`
   - Avoid reloading models from disk

**Cache invalidation**:
- When new data is added for a user → invalidate their features
- When model is retrained → invalidate all predictions for that metric

---

## 🔬 Development

### Running tests (TODO)
```bash
pytest tests/ -v
pytest tests/ --cov=app --cov-report=html
```

### Code quality
```bash
# Format code
black app/

# Lint
flake8 app/

# Type checking
mypy app/
```

### Database migrations (TODO - Alembic)
```bash
# Generate migration
alembic revision --autogenerate -m "Add new ML tables"

# Run migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 🚧 TODO (Phase 1 - Feature Engineering)

- [ ] Create `services/feature_engineering.py`
- [ ] Create `services/correlation_engine.py`
- [ ] Create `api/features.py` (API routes)
- [ ] Create `api/correlations.py` (API routes)
- [ ] Write unit tests for feature engineering
- [ ] Add Prometheus metrics for monitoring

---

## 🚧 TODO (Phase 2 - Predictions)

- [ ] Train LSTM model for RHR prediction
- [ ] Train XGBoost model for HRV prediction
- [ ] Create `services/prediction_service.py`
- [ ] Create `api/predictions.py` (API routes)
- [ ] Model evaluation pipeline
- [ ] A/B testing framework

---

## 🚧 TODO (Phase 3 - Insights)

- [ ] Create `services/insight_generator.py`
- [ ] Natural language templates for insights
- [ ] Create `api/insights.py` (API routes)
- [ ] User feedback loop (was this insight helpful?)

---

## 📝 Notes

- **Async all the way**: All database and Redis operations are async for better performance
- **Type-safe**: Uses Pydantic for request/response validation
- **Cached aggressively**: Features and predictions are expensive, cache them
- **Stateless**: ML service is stateless and can scale horizontally
- **Separation of concerns**: Node.js handles CRUD, Python handles ML

---

## 🤝 Integration with Node.js Backend

**Node.js backend** (`http://localhost:3000`):
- Handles user authentication (JWT)
- CRUD operations for meals, activities, health metrics
- Serves React Native app

**Python ML service** (`http://localhost:8000`):
- Reads data from same PostgreSQL database
- Performs ML computations
- Returns predictions and insights
- Node.js backend calls ML service when needed

**Communication flow**:
```
React Native App
       ↓ (JWT auth)
Node.js Backend (port 3000)
       ↓ (internal HTTP call)
Python ML Service (port 8000)
       ↓ (reads)
PostgreSQL Database
```

---

## 📄 License

Part of the Nutri nutrition tracking application.

---

**Status**: Phase 0 Complete ✅
**Next**: Phase 1 (Feature Engineering & Correlation Analysis)
