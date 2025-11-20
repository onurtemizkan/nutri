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
- **Authentication**: Secure sign up/sign in with JWT tokens
- **Daily Dashboard**: View calorie and macronutrient progress at a glance
- **Meal Tracking**: Log breakfast, lunch, dinner, and snacks
- **Nutrition Breakdown**: Track calories, protein, carbs, fat, and fiber
- **User Goals**: Set and manage your daily nutrition goals
- **Profile Management**: Update your goals and account settings
- **Health Metrics**: Sync data from smartwatches (Apple Health, Fitbit, Garmin, Oura, Whoop)
- **ML Insights** (Coming Soon): View how your nutrition affects your health metrics
- **Predictions** (Coming Soon): See forecasts for tomorrow's RHR, HRV based on today's nutrition

### Backend API (Node.js)
- **RESTful API**: Built with Express.js and TypeScript
- **PostgreSQL Database**: Robust data storage with Prisma ORM
- **JWT Authentication**: Secure token-based authentication
- **Meal Management**: Full CRUD operations for meals
- **Health Metrics API**: Track RHR, HRV, sleep, recovery, steps (30+ metric types)
- **Activity Tracking**: Log workouts and exercise (17+ activity types)
- **Daily/Weekly Summaries**: Get nutrition insights over time

### ML Service (Python) - IN-HOUSE MODELS
- **Feature Engineering**: Transforms raw data into 50+ ML features
- **Correlation Analysis**: Finds patterns (e.g., "high protein → better HRV")
- **Predictions**: LSTM neural networks (PyTorch) for RHR/HRV forecasting
- **Recommendations**: Personalized nutrition plans based on your data
- **Anomaly Detection**: Alerts for unusual health patterns
- **All models trained in-house** using PyTorch, scikit-learn, XGBoost

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
├── app/                    # Mobile app screens (Expo Router)
│   ├── (tabs)/            # Tab navigation screens
│   │   ├── index.tsx      # Dashboard/Home screen
│   │   └── profile.tsx    # Profile screen
│   ├── auth/              # Authentication screens
│   │   ├── welcome.tsx    # Welcome screen
│   │   ├── signin.tsx     # Sign in screen
│   │   └── signup.tsx     # Sign up screen
│   ├── add-meal.tsx       # Add meal modal
│   └── _layout.tsx        # Root layout with auth routing
├── lib/                   # Shared libraries
│   ├── api/               # API client
│   ├── context/           # React contexts (Auth)
│   ├── types/             # TypeScript types
│   └── utils/             # Utility functions
│       └── errorHandling.ts # Type-safe error handling (isAxiosError, getErrorMessage)
├── server/                # Node.js Backend API
│   ├── src/
│   │   ├── controllers/   # Request handlers (auth, meals, health-metrics, activities)
│   │   ├── services/      # Business logic (healthMetricService, activityService)
│   │   ├── routes/        # API routes
│   │   ├── middleware/    # Middleware functions
│   │   │   ├── auth.ts           # JWT authentication
│   │   │   ├── errorHandler.ts   # Error handling middleware
│   │   │   ├── rateLimiter.ts    # Rate limiting (in-memory store)
│   │   │   └── sanitize.ts       # Input sanitization (XSS prevention)
│   │   ├── validation/    # Validation schemas
│   │   │   └── schemas.ts        # Centralized Zod schemas (meals, health metrics, activities)
│   │   ├── config/        # Configuration files
│   │   │   └── constants.ts      # Constants (limits, HTTP status codes, error messages)
│   │   ├── utils/         # Utility functions
│   │   │   └── enumValidation.ts # Type-safe enum parsing (HealthMetricType, ActivityType, etc.)
│   │   ├── types/         # TypeScript types
│   │   │   └── pagination.ts     # Pagination utilities (PaginatedResponse, createPaginatedResponse)
│   │   └── __tests__/     # Test files
│   │       ├── setup.ts          # Test utilities and fixtures
│   │       ├── auth.test.ts      # Authentication tests
│   │       ├── meal.test.ts      # Meal CRUD tests
│   │       ├── healthMetric.test.ts # Health metric tests
│   │       └── activity.test.ts  # Activity tests
│   └── prisma/            # Database schema
│       ├── schema.prisma         # Prisma schema (User, Meal, HealthMetric, Activity, MLFeature, etc.)
│       └── migrations/           # Database migrations and indexes
├── ml-service/            # Python ML Service (IN-HOUSE MODELS)
│   ├── app/
│   │   ├── main.py        # FastAPI app entry point
│   │   ├── config.py      # Environment configuration
│   │   ├── database.py    # Async SQLAlchemy connection
│   │   ├── redis_client.py # ML-specific caching
│   │   ├── models/        # SQLAlchemy models (matching Prisma)
│   │   ├── schemas/       # Pydantic schemas (TODO - Phase 1)
│   │   ├── services/      # ML business logic (TODO - Phase 1)
│   │   │   ├── feature_engineering.py (50+ features)
│   │   │   ├── correlation_engine.py (Pearson, Spearman, Granger)
│   │   │   └── prediction_service.py (LSTM, XGBoost)
│   │   └── api/           # ML API routes (TODO - Phase 1)
│   ├── requirements.txt   # Python dependencies (PyTorch, scikit-learn, XGBoost, etc.)
│   ├── Dockerfile         # Multi-stage Docker build
│   ├── docker-compose.yml # Full stack (PostgreSQL + Redis + ML service)
│   └── README.md          # ML service documentation
└── components/            # Reusable UI components
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
     ? 'http://192.168.1.XXX:3000/api'  // Replace with your IP
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

## Future Enhancements

- [ ] Barcode scanning for packaged foods
- [ ] Food database integration (USDA, etc.)
- [ ] Meal history and search
- [ ] Weekly/monthly charts and analytics
- [ ] Water intake tracking
- [ ] Weight tracking with progress charts
- [ ] Photo uploads for meals
- [ ] Recipe creation and sharing
- [ ] Meal plans and suggestions
- [ ] Dark mode support
- [ ] Export data (CSV, PDF)
- [ ] Integration with fitness trackers

## License

MIT

## Author

Built with ❤️ using modern web and mobile technologies
