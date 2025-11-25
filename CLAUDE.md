# Nutri - AI Assistant Development Guide

This document provides comprehensive information for AI assistants working on the Nutri codebase.

## Project Overview

**Nutri** is a full-stack nutrition tracking mobile application with an in-house ML engine for analyzing how nutrition affects health metrics.

- **Mobile App**: React Native (Expo) with TypeScript
- **Backend API**: Node.js + Express + TypeScript + Prisma + PostgreSQL
- **ML Service**: Python (FastAPI) with PyTorch, scikit-learn, XGBoost
- **Goal**: Track meals, calories, macronutrients, and correlate with health metrics (RHR, HRV, recovery)

## Tech Stack

### Mobile App
- **Framework**: React Native + Expo
- **Language**: TypeScript (strict mode)
- **Routing**: Expo Router (file-based)
- **HTTP Client**: Axios
- **Storage**: Expo Secure Store (JWT tokens)
- **State Management**: React Context (AuthContext)

### Backend API
- **Runtime**: Node.js (v16+)
- **Framework**: Express.js
- **Language**: TypeScript (strict mode, zero `any` types)
- **ORM**: Prisma
- **Database**: PostgreSQL 16
- **Authentication**: JWT (bcryptjs for password hashing)
- **Validation**: Zod schemas (centralized in `validation/schemas.ts`)

### ML Service
- **Framework**: FastAPI + Uvicorn
- **Language**: Python 3.9+
- **Database**: SQLAlchemy (async) + asyncpg
- **Cache**: Redis (aioredis)
- **ML Libraries**: PyTorch, scikit-learn, XGBoost, statsmodels, Prophet

## Architecture

### Type Safety (Strict TypeScript)
- **Zero `any` types** - all code is fully typed
- Type-safe enum parsing for Prisma enums (HealthMetricType, ActivityType, ActivityIntensity)
- Centralized Zod validation schemas
- Type guards for error handling (isAxiosError)

### Security
- JWT authentication with secure token handling
- Input sanitization middleware (XSS prevention)
- Rate limiting (configurable windows)
- Parameter pollution prevention
- Content-Type validation for POST/PUT/PATCH

### Performance
- Composite database indexes for common query patterns
- Pagination utilities with configurable limits (default: 100, max: 1000)
- Efficient query filtering by user, date, and type

## Directory Structure

```
nutri/
├── app/                    # Mobile app screens (Expo Router)
│   ├── (tabs)/            # Tab navigation (index.tsx, profile.tsx)
│   ├── auth/              # Auth screens (welcome, signin, signup, forgot-password, reset-password)
│   ├── add-meal.tsx       # Add meal modal
│   └── _layout.tsx        # Root layout with auth routing
│
├── lib/                   # Shared mobile libraries
│   ├── api/
│   │   └── client.ts      # Axios client with JWT interceptors
│   ├── context/
│   │   └── AuthContext.tsx # Authentication state management
│   ├── types/             # TypeScript interfaces
│   └── utils/
│       └── errorHandling.ts # Type-safe error handling (isAxiosError, getErrorMessage)
│
├── server/                # Backend API
│   ├── src/
│   │   ├── controllers/   # Request handlers
│   │   │   ├── authController.ts
│   │   │   ├── mealController.ts
│   │   │   ├── healthMetricController.ts
│   │   │   └── activityController.ts
│   │   │
│   │   ├── services/      # Business logic
│   │   │   ├── healthMetricService.ts
│   │   │   └── activityService.ts
│   │   │
│   │   ├── routes/        # API routes
│   │   │   ├── authRoutes.ts
│   │   │   ├── mealRoutes.ts
│   │   │   ├── healthMetricRoutes.ts
│   │   │   └── activityRoutes.ts
│   │   │
│   │   ├── middleware/    # Middleware functions
│   │   │   ├── auth.ts           # JWT authentication
│   │   │   ├── errorHandler.ts   # Error handling
│   │   │   ├── rateLimiter.ts    # Rate limiting (in-memory)
│   │   │   └── sanitize.ts       # Input sanitization (XSS prevention)
│   │   │
│   │   ├── validation/    # Validation schemas
│   │   │   └── schemas.ts        # Centralized Zod schemas
│   │   │
│   │   ├── config/        # Configuration
│   │   │   ├── database.ts       # Prisma client
│   │   │   └── constants.ts      # Constants (limits, HTTP codes, messages)
│   │   │
│   │   ├── utils/         # Utilities
│   │   │   └── enumValidation.ts # Type-safe enum parsing
│   │   │
│   │   ├── types/         # TypeScript types
│   │   │   ├── index.ts          # Main type exports
│   │   │   └── pagination.ts     # Pagination utilities
│   │   │
│   │   ├── __tests__/     # Test files
│   │   │   ├── setup.ts          # Test utilities and fixtures
│   │   │   ├── auth.test.ts
│   │   │   ├── meal.test.ts
│   │   │   ├── healthMetric.test.ts
│   │   │   └── activity.test.ts
│   │   │
│   │   └── index.ts       # Express app entry point
│   │
│   ├── prisma/
│   │   ├── schema.prisma         # Database schema
│   │   └── migrations/           # Migrations
│   │
│   ├── package.json
│   ├── tsconfig.json
│   └── jest.config.js
│
└── ml-service/            # Python ML service (TODO - implementation in progress)
    └── ...
```

## Development Commands

### Mobile App (from project root)

```bash
# Start development server
npm start

# Run on specific platforms
npm run ios        # iOS simulator (Mac only)
npm run android    # Android emulator
npm run web        # Web browser

# Code quality
npm run lint       # ESLint
npm test          # Run tests
```

### Backend (from /server directory)

#### Development & Build
```bash
npm run dev        # Start dev server with hot reload (ts-node-dev)
npm run build      # Compile TypeScript to JavaScript (outputs to dist/)
npm start          # Run production server (requires build first)
```

#### Database Management
```bash
npm run db:generate    # Generate Prisma client from schema
npm run db:push        # Push schema to database (dev only - no migrations)
npm run db:migrate     # Create and run migrations (production)
npm run db:studio      # Open Prisma Studio GUI (http://localhost:5555)
```

**Important**: Always run `db:generate` after modifying `prisma/schema.prisma`.

#### Testing
```bash
npm test              # Run all tests once
npm run test:watch    # Watch mode (auto-rerun on changes)
npm run test:coverage # Coverage report
npm run test:verbose  # Verbose output
```

**Test Configuration**:
- Framework: Jest
- TypeScript: ts-jest
- Test files: `src/__tests__/*.test.ts`
- Setup: `src/__tests__/setup.ts` (test utilities, fixtures, assertions)

#### Code Quality
```bash
npm run lint       # ESLint (currently requires config migration to v9)
```

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
- See `prisma/migrations/add_performance_indexes.sql` for documentation

## Code Quality Standards

### TypeScript
- **Strict mode enabled** - no implicit any
- **Zero `any` types** - use proper typing or `unknown` with type guards
- Use Prisma types (`Prisma.HealthMetricWhereInput`, etc.)
- Type-safe enum parsing via `utils/enumValidation.ts`

### Validation
- All API inputs validated with Zod schemas
- Schemas centralized in `validation/schemas.ts`
- Example: `createMealSchema`, `updateMealSchema`, `createHealthMetricSchema`

### Error Handling
- Backend: Use `errorHandler` middleware
- Mobile: Use `getErrorMessage()` from `lib/utils/errorHandling.ts`
- Type guards: `isAxiosError()` for API errors

### Constants
- No magic numbers or strings
- Use constants from `config/constants.ts`:
  - `DEFAULT_PAGE_LIMIT`, `MAX_PAGE_LIMIT`
  - `HTTP_STATUS` (200, 201, 400, 401, 404, 500)
  - `ERROR_MESSAGES`

### Testing
- Test structure: Arrange, Act, Assert
- Use test utilities from `__tests__/setup.ts`:
  - `createTestUser()`, `createTestToken()`
  - `createTestMeal()`, `createTestActivity()`, `createTestHealthMetric()`
  - Assertion helpers: `assertUserStructure()`, `assertMealStructure()`, etc.

## Important Patterns

### Enum Validation
```typescript
// ❌ Wrong - unsafe
const metricType = req.params.metricType as any;

// ✅ Correct - type-safe
import { parseHealthMetricType } from '../utils/enumValidation';
const metricType = parseHealthMetricType(req.params.metricType);
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

### Pagination
```typescript
import { getPaginationParams, createPaginatedResponse } from '../types/pagination';

// In controller
const { skip, take } = getPaginationParams(req.query);
const data = await prisma.meal.findMany({ skip, take });
const total = await prisma.meal.count({ where });
return createPaginatedResponse(data, total, page, limit);
```

### Validation
```typescript
import { createMealSchema } from '../validation/schemas';

// In controller
const validatedData = createMealSchema.parse(req.body);
// Now use validatedData (fully typed and validated)
```

## Security Considerations

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

## Common Tasks

### Adding a New API Endpoint

1. **Define Zod schema** in `validation/schemas.ts`
2. **Create/update controller** in `controllers/`
3. **Create/update service** (if business logic needed) in `services/`
4. **Add route** in `routes/`
5. **Write tests** in `__tests__/`
6. **Update types** if needed in `types/`

### Adding a New Prisma Model

1. **Update schema** in `prisma/schema.prisma`
2. **Generate client**: `npm run db:generate`
3. **Create migration**: `npm run db:migrate` (or `db:push` for dev)
4. **Update TypeScript types** in `types/index.ts`
5. **Create validation schemas** in `validation/schemas.ts`
6. **Add test fixtures** in `__tests__/setup.ts`

### Running Tests

```bash
# Run all tests
npm test

# Run specific test file
npm test -- auth.test.ts

# Run tests in watch mode
npm run test:watch

# Get coverage report
npm run test:coverage
```

### Database Operations

```bash
# View database in GUI
npm run db:studio

# Reset database (dev only)
npm run db:push -- --force-reset

# Create a migration
npm run db:migrate -- --name add_new_field

# Apply migrations (production)
npm run db:migrate
```

## Environment Variables

### Backend (.env)
```env
DATABASE_URL="postgresql://user:password@localhost:5432/nutri_db"
PORT=3000
NODE_ENV=development
JWT_SECRET=your-secret-key-change-in-production
JWT_EXPIRES_IN=7d
```

### Mobile (lib/api/client.ts)
```typescript
const API_BASE_URL = __DEV__
  ? 'http://localhost:3000/api'  // Development
  : 'https://your-api.com/api';   // Production
```

For physical devices, use your computer's IP address instead of localhost.

## Troubleshooting

### TypeScript Errors
```bash
# Check for errors
npm run build

# Common fix: regenerate Prisma client
npm run db:generate
```

### Test Database Issues
- Tests use SQLite by default (see `__tests__/setup.ts`)
- May have issues with PostgreSQL-specific features
- Pre-existing issue, not a blocker for type safety improvements

### Prisma Client Not Found
```bash
# Solution: regenerate Prisma client
npm run db:generate
```

### Port Already in Use
```bash
# Find process using port 3000
lsof -ti:3000

# Kill process
kill -9 $(lsof -ti:3000)
```

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

## Git Workflow

```bash
# Current branch
git status

# Create feature branch
git checkout -b feature/your-feature

# Commit changes
git add .
git commit -m "feat: description"

# Push to remote
git push origin feature/your-feature
```

## Resources

- **Expo Docs**: https://docs.expo.dev/
- **Prisma Docs**: https://www.prisma.io/docs
- **Express Docs**: https://expressjs.com/
- **Zod Docs**: https://zod.dev/
- **TypeScript Docs**: https://www.typescriptlang.org/docs/

## Notes

- This codebase has been thoroughly refactored for type safety and code quality
- All controllers, services, and tests use strict TypeScript (zero `any` types)
- Security middleware is production-ready (rate limiting, sanitization)
- Test infrastructure is comprehensive with fixtures and assertion helpers
- Database schema supports 30+ health metric types and 17+ activity types
- ML service implementation is in progress (Python FastAPI)

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md

---

## Claude Code Workflow Configuration

This project is configured with an optimized Claude Code workflow for efficient development.

### Quick Reference: Slash Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/dev` | Start all development services | `/dev`, `/dev stop`, `/dev status` |
| `/test` | Smart test runner (auto-detects) | `/test`, `/test server`, `/test:watch` |
| `/check` | Code quality checks | `/check`, `/check server`, `/check fix` |
| `/db` | Database operations | `/db status`, `/db studio`, `/db generate` |
| `/build` | Build services | `/build`, `/build server` |
| `/status` | Check all service health | `/status` |
| `/fix` | Auto-fix common issues | `/fix`, `/fix lint`, `/fix deps` |
| `/impl` | Implementation helper | `/impl add user profile endpoint` |
| `/log` | View service logs | `/log`, `/log server -f` |
| `/api` | Test API endpoints | `/api health`, `/api GET /api/meals` |

### Service-Aware Routing

Claude Code automatically routes tasks based on file locations:

| Path Pattern | Service | Test Command | Check Command |
|--------------|---------|--------------|---------------|
| `server/**/*.ts` | Backend API | `cd server && npm test` | `cd server && npm run build` |
| `app/**/*.tsx`, `lib/**/*.ts` | Mobile | `npm test` | `npx tsc --noEmit` |
| `ml-service/**/*.py` | ML Service | `cd ml-service && make test` | `cd ml-service && make typecheck` |
| `prisma/schema.prisma` | Database | - | `npm run db:generate` |

### Development Workflow

#### Starting Development

```bash
# Option 1: Use slash command
/dev

# Option 2: Manual
./scripts/start-all.sh
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- Backend API (port 3000)

#### Testing Flow

```bash
# Auto-detect what to test based on changes
/test

# Test specific service
/test server    # Backend
/test mobile    # Mobile app
/test ml        # ML service

# Watch mode
/test server:watch

# Coverage
/test server:coverage
```

#### Code Quality

```bash
# Check everything
/check

# Auto-fix issues
/check fix

# Service-specific
/check server
/check mobile
/check ml
```

### Implementation Workflow

When implementing new features:

1. **Use `/impl` command** to get guided implementation:
   ```
   /impl add meal history pagination
   ```

2. **Follow the stack order**:
   - Database schema (Prisma) → `db:generate` → `db:push`
   - Backend validation (Zod schemas)
   - Backend service/controller
   - Backend routes + tests
   - Mobile API client updates
   - Mobile UI components + tests

3. **Verify with**:
   ```
   /check
   /test
   ```

### Hooks

The project has automated hooks:

#### Post-Edit Hook
After editing files, Claude provides hints:
- `*.ts` in server → "Run `/check server`"
- `schema.prisma` → "Run `/db generate`"
- Test files → "Run `/test`"

#### Pre-Bash Hook
Validates commands before execution:
- Blocks dangerous patterns (`rm -rf /`, etc.)
- Warns about production database access
- Warns about force flags

### Intelligent Context Awareness

Claude Code understands this project's patterns:

1. **TypeScript Strict Mode**: Zero `any` types enforced
2. **Validation**: All inputs use Zod schemas
3. **Error Handling**: Use type guards (`isAxiosError`)
4. **Constants**: No magic numbers (use `config/constants.ts`)
5. **Testing**: Arrange-Act-Assert pattern with fixtures

### Common Development Scenarios

#### Adding a New API Endpoint

```
User: Add an endpoint to get meal statistics
Claude: Uses /impl pattern:
  1. Creates Zod schema in validation/schemas.ts
  2. Creates controller method
  3. Adds route
  4. Writes tests
  5. Runs /check and /test
```

#### Fixing a Bug

```
User: The meal calories aren't being saved correctly
Claude:
  1. Reads relevant files (controller, service)
  2. Identifies issue
  3. Fixes code
  4. Runs /test server to verify
  5. Provides summary
```

#### Database Schema Change

```
User: Add a 'notes' field to meals
Claude:
  1. Updates prisma/schema.prisma
  2. Runs cd server && npm run db:generate
  3. Runs cd server && npm run db:push
  4. Updates Zod schema
  5. Updates controller if needed
  6. Runs /test server
```

### Performance Tips

1. **Parallel Operations**: When possible, Claude runs independent tasks in parallel
2. **Smart File Reading**: Uses Glob/Grep before reading full files
3. **Incremental Testing**: Only runs tests for changed areas
4. **Cached Context**: Remembers project structure within session

### Troubleshooting

#### Services Won't Start
```
/fix deps      # Reinstall dependencies
/fix clean     # Clean build artifacts
/dev           # Try starting again
```

#### TypeScript Errors
```
/db generate   # Regenerate Prisma client
/check server  # Find remaining issues
```

#### Tests Failing
```
/status        # Check if services are running
/test:verbose  # Get detailed error output
```

### Configuration Files

```
.claude/
├── commands/          # Slash commands
│   ├── dev.md        # Start development
│   ├── test.md       # Smart testing
│   ├── check.md      # Code quality
│   ├── db.md         # Database ops
│   ├── build.md      # Build services
│   ├── status.md     # Service health
│   ├── fix.md        # Auto-fix issues
│   ├── impl.md       # Implementation helper
│   ├── log.md        # View logs
│   └── api.md        # API testing
├── hooks/            # Automation hooks
│   ├── nutri-post-edit.sh
│   └── nutri-pre-bash.sh
└── settings.json     # Claude Code settings
```

### Best Practices with Claude Code

1. **Start sessions with `/status`** to verify services are running
2. **Use `/impl` for new features** to get guided implementation
3. **Run `/check` before committing** to catch issues early
4. **Use `/test` liberally** - it's smart about what to test
5. **Trust the hooks** - they provide helpful reminders
6. **Ask for `/db` help** when working with schema changes

