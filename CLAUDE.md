# Project Context for Claude Code

**This file provides project-specific context for Claude Code and its agents.**

---

## Agent System

**Scope:** Project-scoped agents (in `~/.claude/agents/`)

**Available Agents:** 7 production-ready agents for Expo/React Native

- **Grand Architect** - Meta-orchestrator for complex features
- **Design Token Guardian** - Enforces design system consistency
- **A11y Enforcer** - WCAG 2.2 compliance validation
- **Test Generator** - Auto-generates tests with ROI prioritization
- **Performance Enforcer** - Tracks performance budgets
- **Performance Prophet** - Predictive performance analysis
- **Security Specialist** - Security audits & penetration testing

**Updating Agents:**

```bash
# From this project root
"../claude-code-reactnative-expo-agent-system/scripts/install-agents.sh --scope project"
# Windows alternative: scripts/install-agents.ps1 -Scope project
```

**Team Sync:** Agents are version controlled in `.claude/` - team members get them automatically via git.

---

## Project Overview

**Project Name:** Nutri
**Description:** Full-stack nutrition tracking mobile application with an in-house ML engine for analyzing how nutrition affects health metrics.
**Target Platforms:** iOS, Android (React Native/Expo)

**Goals:**

- Track meals, calories, macronutrients
- Correlate nutrition with health metrics (RHR, HRV, recovery)
- Provide ML-powered insights on nutrition-health relationships

---

## Tech Stack

### Mobile App (Core)

- **Framework:** React Native + Expo
- **Language:** TypeScript (strict mode)
- **Routing:** Expo Router (file-based)
- **HTTP Client:** Axios
- **Storage:** Expo Secure Store (JWT tokens)
- **State Management:** React Context (AuthContext)

### Backend API

- **Runtime:** Node.js (v16+)
- **Framework:** Express.js
- **Language:** TypeScript (strict mode, zero `any` types)
- **ORM:** Prisma
- **Database:** PostgreSQL 16
- **Authentication:** JWT (bcryptjs for password hashing)
- **Validation:** Zod schemas (centralized in `validation/schemas.ts`)

### ML Service

- **Framework:** FastAPI + Uvicorn
- **Language:** Python 3.9+
- **Database:** SQLAlchemy (async) + asyncpg
- **Cache:** Redis (aioredis)
- **ML Libraries:** PyTorch, scikit-learn, XGBoost, statsmodels, Prophet

---

## Architecture

### Folder Structure

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
│   ├── edit-health-metric/[id].tsx
│   ├── add-meal.tsx          # Add meal modal
│   ├── scan-food.tsx         # Camera food scanning
│   ├── scan-barcode.tsx      # Barcode scanner
│   ├── scan-supplement-barcode.tsx
│   ├── ar-measure.tsx        # AR measurement
│   ├── ar-scan-food.tsx      # AR food scanning
│   ├── supplements.tsx       # Supplement tracking
│   ├── health-settings.tsx   # HealthKit settings
│   ├── privacy.tsx, terms.tsx
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
│   │   ├── ARMeasurementOverlay.tsx, ManualSizePicker.tsx
│   │   ├── MicronutrientDisplay.tsx, SupplementTracker.tsx
│   │   └── responsive/       # Responsive design components
│   ├── context/
│   │   └── AuthContext.tsx
│   ├── modules/
│   │   └── LiDARModule.ts    # LiDAR depth sensing
│   ├── services/
│   │   └── healthkit/        # HealthKit integration
│   │       ├── index.ts, permissions.ts, sync.ts
│   │       ├── cardiovascular.ts, respiratory.ts
│   │       ├── sleep.ts, activity.ts
│   ├── theme/
│   │   └── colors.ts         # Design tokens
│   ├── responsive/           # Responsive utilities
│   ├── types/                # TypeScript interfaces
│   └── utils/
│       ├── errorHandling.ts, formatters.ts
│       ├── depth-projection.ts, portion-estimation.ts
│       └── nutritionSanitizer.ts
│
├── server/                    # Backend API
│   ├── src/
│   │   ├── controllers/      # auth, meal, healthMetric, activity, supplement
│   │   ├── services/         # Business logic (auth, meal, healthMetric, activity, supplement)
│   │   ├── routes/           # auth, meal, healthMetric, activity, supplement, foodAnalysis
│   │   ├── middleware/       # auth, errorHandler, rateLimiter, sanitize, requestLogger
│   │   ├── validation/       # Zod schemas
│   │   ├── config/           # database, constants, env, logger
│   │   ├── utils/            # enumValidation, authHelpers, controllerHelpers, dateHelpers
│   │   ├── types/            # TypeScript types (index, pagination)
│   │   ├── __tests__/        # Test files
│   │   └── index.ts          # Express app entry point
│   ├── prisma/
│   │   └── schema.prisma     # Database schema
│   └── package.json
│
├── ml-service/                # Python ML service
│   ├── app/
│   │   ├── main.py           # FastAPI entry point
│   │   ├── config.py, database.py, redis_client.py
│   │   ├── api/              # API routes (food_analysis)
│   │   ├── core/             # Core utilities
│   │   │   ├── logging.py, device.py
│   │   │   └── queue/        # Inference queue system
│   │   ├── middleware/       # Request logging
│   │   ├── ml_models/        # CLIP, Food-101, ensemble classifiers
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
│   ├── start-backend.sh      # Start Backend + ML
│   ├── stop-all.sh, stop-dev.sh
│   ├── docker-dev.sh         # Docker helper
│   └── deploy-device.sh      # Device deployment
│
├── e2e/                       # E2E tests (Maestro)
│   ├── tests/                # Test flows
│   └── scripts/              # Test runners
│
└── .claude/
    └── settings.local.json   # Claude Code local settings
```

### Design Patterns

- **Type Safety:** Zero `any` types, strict TypeScript, type-safe enum parsing
- **Validation:** Centralized Zod schemas for all API inputs
- **Error Handling:** Type guards (`isAxiosError`), middleware-based error handling
- **Security:** JWT auth, input sanitization, rate limiting, parameter pollution prevention

---

## Coding Conventions

### TypeScript

```typescript
// ✅ Strict mode - no implicit any
// ✅ Zero `any` types - use proper typing or `unknown` with type guards
// ✅ Use Prisma types (Prisma.HealthMetricWhereInput, etc.)
// ✅ Type-safe enum parsing via utils/enumValidation.ts

// ❌ Wrong - unsafe
const metricType = req.params.metricType as any;

// ✅ Correct - type-safe
import { parseHealthMetricType } from '../utils/enumValidation';
const metricType = parseHealthMetricType(req.params.metricType);
```

### Validation

```typescript
// ✅ All API inputs validated with Zod schemas
import { createMealSchema } from '../validation/schemas';

const validatedData = createMealSchema.parse(req.body);
// validatedData is fully typed and validated
```

### Error Handling (Mobile)

```typescript
// ❌ Wrong
catch (error: any) {
  Alert.alert('Error', error.response?.data?.error || 'Failed');
}

// ✅ Correct
import { getErrorMessage } from '@/lib/utils/errorHandling';
catch (error) {
  Alert.alert('Error', getErrorMessage(error, 'Failed to load data'));
}
```

### Constants

```typescript
// ✅ No magic numbers or strings
// ✅ Use constants from config/constants.ts:
//    - DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
//    - HTTP_STATUS (200, 201, 400, 401, 404, 500)
//    - ERROR_MESSAGES
```

### Naming Conventions

- **Components:** PascalCase (`UserProfile.tsx`)
- **Hooks:** camelCase with 'use' prefix (`useAuth.ts`)
- **Utilities:** camelCase (`formatDate.ts`)
- **Constants:** UPPER_SNAKE_CASE (`API_BASE_URL`)

---

## Database Schema

### Key Models

**User**

- Authentication (email, password)
- Nutrition goals (goalCalories, goalProtein, goalCarbs, goalFat)
- Physical info (currentWeight, goalWeight, height, activityLevel)

**Meal**

- Nutrition tracking (calories, protein, carbs, fat, fiber, sugar)
- Meal type: breakfast, lunch, dinner, snack
- Timestamps: consumedAt

**HealthMetric**

- 30+ metric types (RESTING_HEART_RATE, HEART_RATE_VARIABILITY_SDNN, SLEEP_DURATION, etc.)
- Source integrations (APPLE_HEALTH, FITBIT, GARMIN, OURA, WHOOP, MANUAL)
- Metadata support (JSON field)

**Activity**

- 17+ activity types (RUNNING, CYCLING, SWIMMING, WEIGHTLIFTING, etc.)
- Intensity levels (LOW, MODERATE, HIGH, VERY_HIGH)
- Duration and calorie tracking

### Database Indexes

- Composite indexes on (userId, date) for efficient queries
- Defined in `prisma/schema.prisma` using `@@index` directives

---

## Development Commands

### Mobile App (from project root)

```bash
# Development
npm start              # Start Expo development server
npm run ios            # iOS simulator (Mac only)
npm run android        # Android emulator
npm run web            # Web browser

# Testing
npm test               # Run Jest tests
npm run lint           # ESLint

# E2E Testing (Maestro)
npm run test:e2e       # Run all E2E tests
npm run test:e2e:auth  # Auth flows only
npm run test:e2e:meals # Meal flows only

# Docker Development
npm run docker:dev:start   # Start Docker dev environment
npm run docker:dev:stop    # Stop Docker dev environment
npm run docker:dev:logs    # View Docker logs

# Device Deployment
npm run deploy:device      # Deploy to connected device
npm run build:ios:local    # Local iOS build

# Service Management
npm run dev:full           # Start all services
npm run dev:stop           # Stop all services
```

### Backend (from /server directory)

```bash
# Development & Build
npm run dev            # Start dev server with hot reload (tsx watch)
npm run dev:debug      # Start with Node.js debugger (port 9229)
npm run dev:with-ml    # Start with ML service
npm run build          # Compile TypeScript
npm start              # Run production server

# Database Management
npm run db:generate    # Generate Prisma client from schema
npm run db:push        # Push schema to database (dev only)
npm run db:migrate     # Create and run migrations (dev)
npm run db:migrate:deploy # Run migrations (production)
npm run db:studio      # Open Prisma Studio GUI
npm run db:seed        # Seed database with test data
npm run db:reset       # Reset database (destructive!)

# Testing
npm test               # Run all tests once
npm run test:watch     # Watch mode
npm run test:coverage  # Coverage report
npm run test:verbose   # Verbose output

# Code Quality
npm run lint           # ESLint
npm run lint:fix       # ESLint with auto-fix
```

**Important:** Always run `db:generate` after modifying `prisma/schema.prisma`.

### ML Service (from /ml-service directory)

```bash
# Setup
make setup             # Complete setup (env + venv + deps)
make setup-redis       # Install and configure Redis

# Development
make dev               # Start development server
make dev-with-redis    # Start dev server with Redis

# Testing
make test              # Run all tests
make test-watch        # Watch mode
make test-coverage     # Coverage report
make test-fast         # Stop on first failure

# Code Quality
make lint              # Run flake8
make format            # Format with black
make typecheck         # Run mypy type checker

# Redis
make redis-start       # Start Redis
make redis-stop        # Stop Redis
make redis-status      # Check Redis status

# Docker
make docker-up         # Start Docker services
make docker-down       # Stop Docker services

# Utilities
make clean             # Clean cache files
make health            # Check service health
make help              # Show all commands
```

---

## Testing Requirements

### Coverage Targets

- **Critical paths** (auth, payments): 90%+
- **Core features:** 80%+
- **UI components:** 60%+

### Test Structure

- Framework: Jest (ts-jest)
- Pattern: Arrange, Act, Assert
- Test files: `src/__tests__/*.test.ts`
- Setup: `src/__tests__/setup.ts`

### Test Utilities

```typescript
// Available from __tests__/setup.ts:
createTestUser(), createTestToken()
createTestMeal(), createTestActivity(), createTestHealthMetric()
assertUserStructure(), assertMealStructure(), etc.
```

---

## Security Guidelines

### Rate Limiting

- API endpoints: 100 requests / 15 minutes
- Auth endpoints: 5 requests / 15 minutes
- Password reset: 3 requests / hour
- Configured in `middleware/rateLimiter.ts`

### Input Sanitization

- All user input sanitized via `middleware/sanitize.ts`
- Prevents XSS attacks (script tags, event handlers, javascript: URIs)
- Applied to req.body, req.query, req.params

### Authentication

- JWT tokens stored in Expo Secure Store (mobile)
- Tokens expire after 7 days (configurable via JWT_EXPIRES_IN)
- Password hashing with bcryptjs (10 rounds)

---

## API Integration

### Base URL

- **Development:** `http://localhost:3000/api`
- **Production:** Configure in `lib/api/client.ts`

### Authentication

- **Method:** JWT
- **Storage:** Expo SecureStore
- **Token refresh:** 7-day expiration

---

## Environment Variables

### Backend (server/.env)

```env
DATABASE_URL="postgresql://postgres:password@localhost:5432/nutri"
PORT=3000
NODE_ENV=development
JWT_SECRET=your-secret-key-change-in-production
JWT_EXPIRES_IN=7d

# CORS Configuration (comma-separated origins for production)
# CORS_ORIGIN=https://your-app.com,https://admin.your-app.com

# Apple Sign In (required for production Apple token verification)
# APPLE_APP_ID=com.your-company.your-app

# Redis Configuration (optional, falls back to in-memory rate limiting)
# REDIS_URL=redis://localhost:6379
```

**Production Requirements:**

- `JWT_SECRET`: Must be a secure, unique secret (not default)
- `CORS_ORIGIN`: Required in production (comma-separated list of allowed origins)
- `APPLE_APP_ID`: Required if using Apple Sign In (your app's bundle identifier)

### ML Service (ml-service/.env)

```env
# Application
APP_NAME=Nutri ML Service
ENVIRONMENT=development
DEBUG=true

# Security (required for production - min 32 chars)
# SECRET_KEY=your-secure-secret-key-min-32-characters

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Database (PostgreSQL with async driver)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/nutri_db
DATABASE_POOL_SIZE=10

# Redis
REDIS_URL=redis://localhost:6379/0

# Cache TTL (seconds)
CACHE_TTL_FEATURES=3600        # 1 hour
CACHE_TTL_PREDICTIONS=86400    # 24 hours

# ML Configuration
MODEL_STORAGE_PATH=./app/ml_models
SKIP_MODEL_WARMUP=false        # Set true for faster dev startup
FAST_MODE=false                # Skip OWL-ViT for faster inference

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

**Production Requirements:**

- `SECRET_KEY`: Must be set in production (minimum 32 characters)
- `ENVIRONMENT`: Must be set to "production"

### Mobile (lib/api/client.ts)

For physical devices, use your computer's IP address instead of localhost.

---

## Common Tasks

### Adding a New API Endpoint

1. Define Zod schema in `validation/schemas.ts`
2. Create/update controller in `controllers/`
3. Create/update service (if business logic needed) in `services/`
4. Add route in `routes/`
5. Write tests in `__tests__/`
6. Update types if needed in `types/`

### Adding a New Prisma Model

1. Update schema in `prisma/schema.prisma`
2. Generate client: `npm run db:generate`
3. Create migration: `npm run db:migrate` (or `db:push` for dev)
4. Update TypeScript types in `types/index.ts`
5. Create validation schemas in `validation/schemas.ts`
6. Add test fixtures in `__tests__/setup.ts`

---

## Anti-Patterns to Avoid

- **No `any` types** - use proper types or `unknown` with type guards
- **No magic numbers/strings** - use constants from `config/constants.ts`
- **No inline styles** - use StyleSheet or design tokens
- **No direct AsyncStorage for sensitive data** - use SecureStore
- **No console.log in production** - use proper logging
- **No hardcoded strings** - use i18n
- **No missing accessibility props** - always add accessibilityLabel
- **No prop drilling >3 levels** - use Context or state management
- **No native navigation headers** - all screens use custom headers; register new screens in `app/_layout.tsx` with `headerShown: false`

---

## Troubleshooting

### TypeScript Errors

```bash
npm run build              # Check for errors
npm run db:generate        # Regenerate Prisma client
```

### Test Database Issues

- Tests use SQLite by default (see `__tests__/setup.ts`)
- May have issues with PostgreSQL-specific features

### Prisma Client Not Found

```bash
npm run db:generate        # Regenerate Prisma client
```

### Port Already in Use

```bash
lsof -ti:3000              # Find process using port 3000
kill -9 $(lsof -ti:3000)   # Kill process
```

---

## Code Review Checklist

Before committing code, verify:

- [ ] TypeScript compilation clean (`npm run build`)
- [ ] No `any` types (use proper types or `unknown` with type guards)
- [ ] All API inputs validated with Zod schemas
- [ ] Error handling uses type guards (isAxiosError, etc.)
- [ ] Constants used instead of magic numbers/strings
- [ ] Tests written for new functionality
- [ ] Tests pass (`npm test`)
- [ ] Prisma client regenerated after schema changes (`npm run db:generate`)
- [ ] No sensitive data in code (use environment variables)

---

## Resources

- **Expo Docs:** https://docs.expo.dev/
- **Prisma Docs:** https://www.prisma.io/docs
- **Express Docs:** https://expressjs.com/
- **Zod Docs:** https://zod.dev/
- **TypeScript Docs:** https://www.typescriptlang.org/docs/

## Task Master AI Instructions

**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md

---

## Claude Code Workflow Configuration

### Service-Aware Development

| Path Pattern                  | Service     | Test Command                 | Check Command                     |
| ----------------------------- | ----------- | ---------------------------- | --------------------------------- |
| `server/**/*.ts`              | Backend API | `cd server && npm test`      | `cd server && npm run build`      |
| `app/**/*.tsx`, `lib/**/*.ts` | Mobile      | `npm test`                   | `npx tsc --noEmit`                |
| `ml-service/**/*.py`          | ML Service  | `cd ml-service && make test` | `cd ml-service && make typecheck` |
| `prisma/schema.prisma`        | Database    | -                            | `npm run db:generate`             |

### Development Workflow

#### Starting Development

```bash
./scripts/start-all.sh     # Start everything (PostgreSQL, Redis, Backend, ML)
./scripts/start-dev.sh     # Docker + ML Service only
./scripts/start-backend.sh # Backend + ML (when Docker already running)
npm run dev:full           # Same as start-all.sh
npm run dev:backend        # Same as start-backend.sh
```

**Services Started:**

- PostgreSQL: 5432
- Redis: 6379
- Backend API: 3000
- ML Service: 8000

**Note:** ML Service (port 8000) always runs alongside the Backend API.

#### Testing

```bash
# Backend
cd server && npm test           # Run all tests
cd server && npm run test:watch # Watch mode
cd server && npm run test:coverage

# ML Service
cd ml-service && make test      # Run all tests
cd ml-service && make test-watch

# Mobile
npm test                        # Run Jest tests
```

#### Code Quality

```bash
# Backend
cd server && npm run lint       # ESLint
cd server && npm run build      # TypeScript check

# ML Service
cd ml-service && make lint      # Flake8
cd ml-service && make typecheck # Mypy
cd ml-service && make format    # Black formatter
```

### Implementation Workflow

1. **Follow the stack order:**
   - Database schema (Prisma) → `db:generate` → `db:push`
   - Backend validation (Zod schemas)
   - Backend service/controller
   - Backend routes + tests
   - Mobile API client updates
   - Mobile UI components + tests
2. **Verify with tests and type checks**

### Configuration Files

```
.claude/
└── settings.local.json   # Claude Code local settings (git-ignored)
```

### MCP Configuration

**iOS Simulator MCP:** This project uses `ios-simulator-eyes-mcp` from a local clone.

- **DO NOT** suggest or switch to `ios-simulator-mcp` (npm public package) or any other iOS simulator MCP
- Location: `~/Projects/ios-simulator-eyes-mcp`
- If the MCP fails or needs updating:
  ```bash
  cd ~/Projects/ios-simulator-eyes-mcp && git pull && npm install && npm run build
  ```

### Best Practices

1. **Start sessions by checking service health:**
   ```bash
   curl http://localhost:3000/health  # Backend
   curl http://localhost:8000/health  # ML Service
   ```
2. **Run tests and type checks before committing**
3. **Follow the implementation workflow** (database → backend → mobile)
4. **Always regenerate Prisma client** after schema changes: `npm run db:generate`
5. **Use Docker for consistent development** environment

---

## Notes

- Codebase thoroughly refactored for type safety and code quality
- All controllers, services, and tests use strict TypeScript (zero `any` types)
- Security middleware is production-ready (rate limiting, sanitization)
- Test infrastructure is comprehensive with fixtures and assertion helpers
- Database schema supports 30+ health metric types and 17+ activity types
- ML service is fully operational with:
  - Food image analysis (CLIP + Food-101 ensemble)
  - Barcode scanning (OpenFoodFacts integration)
  - Inference queue with circuit breaker pattern
  - Prometheus metrics for monitoring

---

**Last Updated:** December 2025
**Maintained By:** Nutri Development Team

---

_© 2025 Nutri | Claude Code Project Context_
