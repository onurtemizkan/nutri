# Nutri - Nutrition Tracking App with ML Engine

A full-stack nutrition tracking mobile application built with React Native (Expo), Node.js, and an in-house Python ML engine. Track your daily meals, calories, and macronutrients while our ML engine analyzes how your nutrition affects your health metrics (RHR, HRV, recovery).

## 🚨 Project Architecture

**Mobile App + In-House ML (NOT Chatbot + NOT External Services)**

- ✅ **React Native mobile app** (Expo) with structured UI
- ✅ **In-house ML models** (PyTorch, scikit-learn, XGBoost)
- ✅ **Node.js backend** (Express + TypeScript + Prisma + PostgreSQL)
- ✅ **Python ML service** (FastAPI + async SQLAlchemy + Redis)
- ❌ **NOT a chatbot** or conversational interface
- ❌ **NOT using external ML APIs** (OpenAI, AWS ML, Google AI, etc.)

### Architecture Highlights

**Type-Safe Backend**: The Node.js backend has been thoroughly refactored with TypeScript strict mode:

- Zero `any` types - all code is fully typed
- Type-safe enum validation for Prisma enums (HealthMetricType, ActivityType, etc.)
- Centralized Zod validation schemas
- Input sanitization middleware for XSS prevention
- Rate limiting middleware with configurable windows

**Security First**: Production-ready security features:

- JWT authentication with secure token handling
- Input sanitization preventing XSS attacks
- Rate limiting (API: 100/15min, Auth: 5/15min, Password Reset: 3/hour)
- Parameter pollution prevention
- Content-Type validation

**Performance Optimized**: Database and API optimizations:

- Composite indexes for common query patterns
- Pagination utilities with configurable limits
- Efficient query filtering by user, date, and type

---

## Features

### Mobile App (React Native)

- **Authentication**: Secure sign up/sign in with JWT tokens, Apple Sign-In, password reset
- **Daily Dashboard**: View calorie and macronutrient progress at a glance
- **Meal Tracking**: Log breakfast, lunch, dinner, and snacks
- **Food Scanning**: Camera-based food recognition with ML analysis
- **Barcode Scanning**: Look up packaged foods via OpenFoodFacts
- **AR Portion Measurement**: Use LiDAR for accurate portion sizing
- **Nutrition Breakdown**: Track calories, protein, carbs, fat, fiber, and micronutrients
- **Supplement Tracking**: Log vitamins and supplements with micronutrient details
- **Health Metrics Dashboard**: View 30+ metric types synced from HealthKit
- **Activity Tracking**: Log 17+ activity types with intensity levels
- **User Goals**: Set and manage your daily nutrition goals
- **Profile Management**: Update your goals and account settings
- **HealthKit Integration**: Sync data from Apple Health (heart rate, HRV, sleep, etc.)
- **ML Insights** (Coming Soon): View how your nutrition affects your health metrics

### Backend API (Node.js)

- **RESTful API**: Built with Express.js and TypeScript
- **PostgreSQL Database**: Robust data storage with Prisma ORM
- **JWT Authentication**: Secure token-based authentication with Apple Sign-In support
- **Meal Management**: Full CRUD operations for meals with food analysis integration
- **Health Metrics API**: Track RHR, HRV, sleep, recovery, steps (30+ metric types)
- **Activity Tracking**: Log workouts and exercise (17+ activity types)
- **Supplement Tracking**: Log vitamins and supplements
- **Food Analysis Proxy**: Routes image analysis to ML service
- **Daily/Weekly Summaries**: Get nutrition insights over time

### ML Service (Python) - FULLY OPERATIONAL

- **Food Image Analysis**: CLIP + Food-101 ensemble classifier for food recognition
- **Multi-Food Detection**: OWL-ViT for detecting multiple foods in one image
- **Barcode Integration**: OpenFoodFacts lookup for packaged foods
- **Inference Queue**: Request queuing with circuit breaker pattern
- **Prometheus Metrics**: Real-time monitoring of ML inference
- **Feature Engineering**: Transforms raw data into 50+ ML features
- **Correlation Analysis**: Finds patterns (e.g., "high protein → better HRV")
- **All models run in-house** using PyTorch, scikit-learn, Hugging Face Transformers

## Tech Stack

### Mobile App (React Native)

- React Native + Expo
- TypeScript
- Expo Router (file-based routing)
- Axios (API calls)
- Expo Secure Store (token storage)
- Victory Native or React Native Charts (data visualization)

### Backend (Node.js)

- Node.js + TypeScript
- Express.js
- Prisma ORM
- PostgreSQL 16
- JWT for authentication
- bcryptjs for password hashing
- Zod for validation

### ML Service (Python) - IN-HOUSE ONLY

- **Framework**: FastAPI + Uvicorn
- **Database**: SQLAlchemy (async) + asyncpg
- **Cache**: Redis (aioredis)
- **ML Libraries** (no external ML APIs):
  - **PyTorch** (LSTM neural networks for deep learning)
  - scikit-learn (correlation, regression, Isolation Forest)
  - XGBoost (gradient boosting)
  - statsmodels (Granger causality, statistical tests)
  - scipy (Pearson/Spearman correlation)
  - Prophet (Facebook's time series library)
  - pandas, numpy (data manipulation)

### Infrastructure

- PostgreSQL 16 (shared database)
- Redis 7 (ML caching layer)
- Docker + docker-compose (local development)

## Code Quality & Refactoring

The codebase has undergone comprehensive refactoring with focus on type safety, security, and maintainability:

### Phase 1: Type Safety Enhancement

- **Eliminated all `any` types**: Removed 20+ `as any` assertions and 15+ `: any` annotations
- **Type-safe enum validation**: Created utilities for parsing Prisma enums (HealthMetricType, ActivityType, ActivityIntensity)
- **Proper error handling**: Type-safe error handling in React Native with `isAxiosError` type guards
- **Strict TypeScript**: All controllers, services, and tests use proper Prisma types

### Phase 2: Code Organization & Consistency

- **Centralized validation**: Consolidated 9 duplicate Zod schemas into `validation/schemas.ts`
- **Constants extraction**: Removed magic numbers/strings from 20+ locations into `config/constants.ts`
- **Reduced codebase**: Net reduction of ~13,600 lines through deduplication and cleanup

### Phase 3: Testing Infrastructure

- **Comprehensive test coverage**: Added tests for health metrics and activities
- **Test utilities**: Created type-safe test fixtures and assertion helpers
- **Jest configuration**: Proper TypeScript integration with ts-jest

### Phase 4: API Enhancement

- **Pagination utilities**: Reusable pagination types and helper functions
- **Consistent response format**: Standardized API responses with metadata

### Phase 5: Performance Optimization

- **Database indexes**: Documented composite indexes for common query patterns
- **Optimized queries**: Efficient filtering by user, date, and type

### Phase 6: Security Hardening

- **Rate limiting**: Configurable rate limiters (API: 100/15min, Auth: 5/15min)
- **Input sanitization**: XSS prevention middleware
- **Parameter pollution prevention**: Array size limits
- **Content-Type validation**: Enforced for POST/PUT/PATCH requests

## Project Structure

```
nutri/
├── app/                       # Mobile app screens (Expo Router)
│   ├── (tabs)/               # Tab navigation
│   │   ├── index.tsx         # Dashboard/Home
│   │   ├── health.tsx        # Health metrics tab
│   │   └── profile.tsx       # Profile tab
│   ├── auth/                 # Auth screens
│   │   ├── welcome.tsx, signin.tsx, signup.tsx
│   │   ├── forgot-password.tsx, reset-password.tsx
│   ├── activity/             # Activity tracking
│   │   ├── index.tsx, [id].tsx, add.tsx
│   ├── health/               # Health metric details
│   │   ├── [metricType].tsx, add.tsx
│   ├── edit-meal/[id].tsx    # Edit meal
│   ├── add-meal.tsx          # Add meal modal
│   ├── scan-food.tsx         # Camera food scanning
│   ├── scan-barcode.tsx      # Barcode scanner
│   ├── ar-measure.tsx        # AR portion measurement
│   ├── supplements.tsx       # Supplement tracking
│   ├── health-settings.tsx   # HealthKit settings
│   └── _layout.tsx           # Root layout
│
├── lib/                       # Shared mobile libraries
│   ├── api/                  # API clients
│   │   ├── client.ts         # Axios with JWT interceptors
│   │   ├── auth.ts, meals.ts, activities.ts
│   │   ├── health-metrics.ts, supplements.ts
│   │   ├── food-analysis.ts, food-feedback.ts
│   │   └── openfoodfacts.ts  # Barcode lookup
│   ├── components/           # Reusable components
│   │   ├── SwipeableMealCard.tsx, SwipeableHealthMetricCard.tsx
│   │   ├── ARMeasurementOverlay.tsx, MicronutrientDisplay.tsx
│   │   └── responsive/       # Responsive design components
│   ├── context/              # React contexts
│   │   └── AuthContext.tsx
│   ├── services/
│   │   └── healthkit/        # HealthKit integration
│   ├── types/                # TypeScript interfaces
│   └── utils/                # Utility functions
│
├── server/                    # Node.js Backend API
│   ├── src/
│   │   ├── controllers/      # auth, meal, healthMetric, activity, supplement
│   │   ├── services/         # Business logic
│   │   ├── routes/           # API routes
│   │   ├── middleware/       # auth, errorHandler, rateLimiter, sanitize, requestLogger
│   │   ├── validation/       # Centralized Zod schemas
│   │   ├── config/           # database, constants, env, logger
│   │   ├── utils/            # enumValidation, helpers
│   │   ├── types/            # TypeScript types
│   │   └── __tests__/        # Test files
│   └── prisma/
│       └── schema.prisma     # Database schema
│
├── ml-service/                # Python ML Service
│   ├── app/
│   │   ├── main.py           # FastAPI entry point
│   │   ├── config.py, database.py, redis_client.py
│   │   ├── api/              # API routes (food_analysis)
│   │   ├── core/             # Core utilities
│   │   │   ├── logging.py, device.py
│   │   │   └── queue/        # Inference queue with circuit breaker
│   │   ├── middleware/       # Request logging
│   │   ├── ml_models/        # CLIP, Food-101, OWL-ViT classifiers
│   │   ├── models/           # SQLAlchemy models
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # ML business logic
│   │   └── data/             # Food database
│   ├── tests/                # Test files
│   ├── Makefile              # Development commands
│   └── requirements.txt
│
├── scripts/                   # Development scripts
│   ├── start-all.sh          # Start everything
│   ├── start-dev.sh          # Start Docker + ML
│   ├── docker-dev.sh         # Docker helper
│   └── deploy-device.sh      # Device deployment
│
├── e2e/                       # E2E tests (Maestro)
│   ├── tests/                # Test flows
│   └── scripts/              # Test runners
│
└── .claude/
    └── settings.local.json   # Claude Code settings
```

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- PostgreSQL (v12 or higher)
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (for Mac) or Android Emulator

### Backend Setup

1. **Navigate to the server directory**:

   ```bash
   cd server
   ```

2. **Install dependencies**:

   ```bash
   npm install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the `server` directory:

   ```env
   DATABASE_URL="postgresql://postgres:password@localhost:5432/nutri_db"
   PORT=3000
   NODE_ENV=development
   JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
   JWT_EXPIRES_IN=7d
   ```

4. **Create PostgreSQL database**:

   ```bash
   # Using psql
   createdb nutri_db

   # Or using PostgreSQL CLI
   psql -U postgres
   CREATE DATABASE nutri_db;
   \q
   ```

5. **Generate Prisma client and push schema**:

   ```bash
   npm run db:generate
   npm run db:push
   ```

6. **Start the development server**:

   ```bash
   npm run dev
   ```

   The API will be available at `http://localhost:3000`

### Mobile App Setup

1. **Navigate to the project root**:

   ```bash
   cd ..
   ```

2. **Install dependencies**:

   ```bash
   npm install
   ```

3. **Update API URL** (if needed):

   For physical devices, update the API URL in `lib/api/client.ts`:

   ```typescript
   // Change localhost to your computer's IP address
   const API_BASE_URL = __DEV__
     ? 'http://192.168.1.XXX:3000/api' // Replace with your IP
     : 'https://your-production-api.com/api';
   ```

4. **Start the Expo development server**:

   ```bash
   npm start
   ```

5. **Run on iOS/Android**:

   ```bash
   # iOS Simulator (Mac only)
   npm run ios

   # Android Emulator
   npm run android

   # Web
   npm run web
   ```

## Docker Development Environment

### Quick Start - Full Docker Setup (Recommended)

Start all services in Docker containers:

```bash
docker compose -f docker-compose.dev.yml up -d
```

This starts:
| Service | Port | URL |
|---------|------|-----|
| PostgreSQL | 5432 | `postgresql://postgres:postgres@localhost:5432/nutri_db` |
| Redis | 6379 | `redis://localhost:6379` |
| Backend API | 3000 | http://localhost:3000 |
| ML Service | 8000 | http://localhost:8000 |
| Adminer (DB UI) | 8080 | http://localhost:8080 |
| Redis Commander | 8081 | http://localhost:8081 |
| Prisma Studio | 5555 | http://localhost:5555 |

Stop all services:

```bash
docker compose -f docker-compose.dev.yml down
```

View logs:

```bash
# All services
docker compose -f docker-compose.dev.yml logs -f

# Specific service
docker compose -f docker-compose.dev.yml logs -f backend
docker compose -f docker-compose.dev.yml logs -f ml-service
```

### Alternative - Hybrid Setup (Docker + Local)

Start infrastructure in Docker, run services locally:

```bash
# Start PostgreSQL, Redis, and ML Service
./scripts/start-dev.sh

# Start Backend API (in a separate terminal)
cd server && npm run dev
```

Or use the all-in-one script:

```bash
./scripts/start-all.sh
```

Stop all services:

```bash
./scripts/stop-all.sh
```

### Shell Scripts Reference

| Script                       | Description                              |
| ---------------------------- | ---------------------------------------- |
| `./scripts/start-dev.sh`     | Start Docker services + ML Service       |
| `./scripts/start-all.sh`     | Start everything (Docker + Backend + ML) |
| `./scripts/start-backend.sh` | Start Backend + ML (when Docker running) |
| `./scripts/stop-dev.sh`      | Stop Docker and local services           |
| `./scripts/stop-all.sh`      | Stop all services                        |
| `./scripts/docker-dev.sh`    | Docker helper (start/stop/logs/build)    |

### Health Checks

```bash
# Backend API
curl http://localhost:3000/health

# ML Service
curl http://localhost:8000/health
```

### Debug Ports

- **Backend (Node.js):** 9229 (attach VS Code debugger)
- **ML Service (Python):** 5678 (debugpy)

### Docker Volumes

- `nutri_dev_postgres_data` - PostgreSQL data
- `nutri_dev_redis_data` - Redis data
- `nutri_dev_backend_node_modules` - Backend dependencies
- `nutri_dev_backend_prisma_client` - Prisma client

### Common Docker Issues

**Port Already in Use:**

```bash
# Find process using port
lsof -ti:3000

# Kill process
kill -9 $(lsof -ti:3000)
```

**ML Service Database Error:**
If you see "psycopg2 is not async", ensure DATABASE_URL uses:

```
postgresql+asyncpg://...
```

NOT:

```
postgresql://...
```

**Network Issues (containers can't communicate):**

```bash
# Clean up and restart
docker compose -f docker-compose.dev.yml down
docker network prune -f
docker compose -f docker-compose.dev.yml up -d
```

---

## API Documentation

### Base URL

```
http://localhost:3000/api
```

### Authentication Endpoints

#### Register

```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "name": "John Doe"
}
```

#### Login

```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

#### Apple Sign-In

```http
POST /api/auth/apple-signin
Content-Type: application/json

{
  "identityToken": "eyJ...",
  "user": "apple-user-id",
  "email": "user@privaterelay.appleid.com",
  "fullName": { "givenName": "John", "familyName": "Doe" }
}
```

#### Forgot Password

```http
POST /api/auth/forgot-password
Content-Type: application/json

{
  "email": "user@example.com"
}
```

#### Verify Reset Token

```http
POST /api/auth/verify-reset-token
Content-Type: application/json

{
  "token": "reset-token-from-email"
}
```

#### Reset Password

```http
POST /api/auth/reset-password
Content-Type: application/json

{
  "token": "reset-token-from-email",
  "password": "newPassword123"
}
```

#### Get Profile (Protected)

```http
GET /api/auth/profile
Authorization: Bearer {token}
```

#### Update Profile (Protected)

```http
PUT /api/auth/profile
Authorization: Bearer {token}
Content-Type: application/json

{
  "goalCalories": 2200,
  "goalProtein": 160,
  "goalCarbs": 220,
  "goalFat": 70
}
```

#### Delete Account (Protected)

```http
DELETE /api/auth/account
Authorization: Bearer {token}
```

### Meal Endpoints (All Protected)

#### Create Meal

```http
POST /api/meals
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "Grilled Chicken Salad",
  "mealType": "lunch",
  "calories": 450,
  "protein": 35,
  "carbs": 30,
  "fat": 18,
  "fiber": 8,
  "servingSize": "1 bowl"
}
```

#### Get Today's Meals

```http
GET /api/meals
Authorization: Bearer {token}
```

#### Get Meals by Date

```http
GET /api/meals?date=2025-01-25T00:00:00.000Z
Authorization: Bearer {token}
```

#### Get Daily Summary

```http
GET /api/meals/summary/daily
Authorization: Bearer {token}
```

#### Get Weekly Summary

```http
GET /api/meals/summary/weekly
Authorization: Bearer {token}
```

#### Update Meal

```http
PUT /api/meals/{mealId}
Authorization: Bearer {token}
Content-Type: application/json

{
  "calories": 500,
  "protein": 40
}
```

#### Delete Meal

```http
DELETE /api/meals/{mealId}
Authorization: Bearer {token}
```

### Health Metrics Endpoints (All Protected)

#### Create Health Metric

```http
POST /api/health-metrics
Authorization: Bearer {token}
Content-Type: application/json

{
  "metricType": "RESTING_HEART_RATE",
  "value": 65,
  "unit": "bpm",
  "recordedAt": "2025-01-25T08:00:00.000Z",
  "source": "APPLE_HEALTH",
  "sourceId": "optional-device-id",
  "metadata": {}
}
```

#### Get Health Metrics

```http
GET /api/health-metrics
Authorization: Bearer {token}

# Query parameters:
# - metricType: Filter by specific type (e.g., RESTING_HEART_RATE)
# - startDate: ISO date string
# - endDate: ISO date string
# - source: Filter by source (APPLE_HEALTH, FITBIT, etc.)
# - page: Page number (default: 1)
# - limit: Results per page (default: 100, max: 1000)
```

#### Get Health Metric by ID

```http
GET /api/health-metrics/{metricId}
Authorization: Bearer {token}
```

#### Update Health Metric

```http
PUT /api/health-metrics/{metricId}
Authorization: Bearer {token}
Content-Type: application/json

{
  "value": 67,
  "unit": "bpm"
}
```

#### Delete Health Metric

```http
DELETE /api/health-metrics/{metricId}
Authorization: Bearer {token}
```

### Activity Endpoints (All Protected)

#### Create Activity

```http
POST /api/activities
Authorization: Bearer {token}
Content-Type: application/json

{
  "activityType": "RUNNING",
  "intensity": "MODERATE",
  "startedAt": "2025-01-25T06:00:00.000Z",
  "endedAt": "2025-01-25T06:30:00.000Z",
  "duration": 30,
  "caloriesBurned": 250,
  "source": "manual"
}
```

#### Get Activities

```http
GET /api/activities
Authorization: Bearer {token}

# Query parameters:
# - activityType: Filter by type (RUNNING, CYCLING, etc.)
# - intensity: Filter by intensity (LOW, MODERATE, HIGH, VERY_HIGH)
# - startDate: ISO date string
# - endDate: ISO date string
# - page: Page number (default: 1)
# - limit: Results per page (default: 100, max: 1000)
```

#### Get Activity by ID

```http
GET /api/activities/{activityId}
Authorization: Bearer {token}
```

#### Update Activity

```http
PUT /api/activities/{activityId}
Authorization: Bearer {token}
Content-Type: application/json

{
  "duration": 35,
  "caloriesBurned": 275
}
```

#### Delete Activity

```http
DELETE /api/activities/{activityId}
Authorization: Bearer {token}
```

## Database Schema

### User

- id (String, Primary Key)
- email (String, Unique)
- password (String, Hashed)
- name (String)
- goalCalories (Int, Default: 2000)
- goalProtein (Float, Default: 150)
- goalCarbs (Float, Default: 200)
- goalFat (Float, Default: 65)
- currentWeight (Float, Optional)
- goalWeight (Float, Optional)
- height (Float, Optional)
- activityLevel (String, Default: "moderate")

### Meal

- id (String, Primary Key)
- userId (String, Foreign Key → User)
- name (String)
- mealType (String: breakfast | lunch | dinner | snack)
- calories (Float)
- protein (Float)
- carbs (Float)
- fat (Float)
- fiber (Float, Optional)
- sugar (Float, Optional)
- servingSize (String, Optional)
- notes (String, Optional)
- consumedAt (DateTime)

### WaterIntake

- id (String, Primary Key)
- userId (String, Foreign Key → User)
- amount (Float, in ml)
- recordedAt (DateTime)

### WeightRecord

- id (String, Primary Key)
- userId (String, Foreign Key → User)
- weight (Float, in kg)
- recordedAt (DateTime)

### HealthMetric

- id (String, Primary Key)
- userId (String, Foreign Key → User)
- metricType (Enum: 30+ types including RESTING_HEART_RATE, HEART_RATE_VARIABILITY_SDNN, SLEEP_DURATION, etc.)
- value (Float)
- unit (String: bpm, ms, hours, etc.)
- recordedAt (DateTime)
- source (Enum: APPLE_HEALTH, FITBIT, GARMIN, OURA, WHOOP, MANUAL, etc.)
- sourceId (String, Optional)
- metadata (JSON, Optional)

### Activity

- id (String, Primary Key)
- userId (String, Foreign Key → User)
- activityType (Enum: 17+ types including RUNNING, CYCLING, SWIMMING, WEIGHTLIFTING, etc.)
- intensity (Enum: LOW, MODERATE, HIGH, VERY_HIGH)
- startedAt (DateTime)
- endedAt (DateTime, Optional)
- duration (Int, in minutes)
- caloriesBurned (Float, Optional)
- source (String)

## Development Scripts

### Mobile App

```bash
npm start          # Start Expo development server
npm run ios        # Run on iOS simulator
npm run android    # Run on Android emulator
npm run web        # Run in web browser
npm run lint       # Run ESLint
npm test          # Run tests
```

### Backend

#### Development & Build

```bash
npm run dev        # Start development server with hot reload (ts-node-dev)
npm run build      # Build TypeScript to JavaScript
npm start          # Run production server (requires npm run build first)
```

#### Database Management

```bash
npm run db:generate    # Generate Prisma client from schema
npm run db:push        # Push schema changes to database (dev only)
npm run db:migrate     # Create and run migrations (production)
npm run db:studio      # Open Prisma Studio GUI (http://localhost:5555)
```

#### Testing

```bash
npm test              # Run all tests once
npm run test:watch    # Run tests in watch mode (auto-rerun on changes)
npm run test:coverage # Run tests with coverage report
npm run test:verbose  # Run tests with verbose output
```

#### Code Quality

```bash
npm run lint       # Run ESLint for code quality checks
```

## Environment Variables

### Backend (.env)

```env
DATABASE_URL="postgresql://user:password@localhost:5432/nutri_db"
PORT=3000
NODE_ENV=development
JWT_SECRET=your-secret-key
JWT_EXPIRES_IN=7d
```

## Design Philosophy

The app follows modern iOS/Android design patterns with:

- **Minimal UI**: Clean, uncluttered interface
- **Visual Progress**: Circular gauges and progress bars
- **Card-based Layout**: Information grouped in digestible cards
- **Color-coded Data**: Macronutrients use distinct colors
- **Pull-to-Refresh**: Quick data updates
- **Modal Navigation**: Add meal screen as modal overlay

## Feature Status

### Implemented

- [x] Barcode scanning (OpenFoodFacts integration)
- [x] Food photo analysis (ML-powered with CLIP + Food-101)
- [x] AR portion measurement with LiDAR
- [x] HealthKit integration (Apple Health sync)
- [x] Supplement tracking with micronutrients
- [x] Health metrics dashboard (30+ metric types)
- [x] Activity tracking (17+ activity types)
- [x] Daily/weekly nutrition summaries

### In Progress

- [ ] ML insights on nutrition-health correlations
- [ ] Predictive analytics for health metrics

### Planned

- [ ] Recipe creation and sharing
- [ ] Meal plans and suggestions
- [ ] Dark mode support
- [ ] Export data (CSV, PDF)
- [ ] Android Wear / Apple Watch companion app

## License

MIT

## Author

Built with ❤️ using modern web and mobile technologies
