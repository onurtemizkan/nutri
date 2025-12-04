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
│   │   ├── services/      # Business logic
│   │   ├── routes/        # API routes
│   │   ├── middleware/    # Middleware (auth, errorHandler, rateLimiter, sanitize)
│   │   ├── validation/    # Zod schemas
│   │   ├── config/        # Configuration (database, constants)
│   │   ├── utils/         # Utilities (enumValidation)
│   │   ├── types/         # TypeScript types
│   │   ├── __tests__/     # Test files
│   │   └── index.ts       # Express app entry point
│   ├── prisma/
│   │   ├── schema.prisma  # Database schema
│   │   └── migrations/    # Migrations
│   └── package.json
│
└── ml-service/            # Python ML service (in progress)
    └── ...
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
- See `prisma/migrations/add_performance_indexes.sql` for documentation

---

## Development Commands

### Mobile App (from project root)
```bash
npm start              # Start development server
npm run ios            # iOS simulator (Mac only)
npm run android        # Android emulator
npm run web            # Web browser
npm run lint           # ESLint
npm test               # Run tests
```

### Backend (from /server directory)
```bash
# Development & Build
npm run dev            # Start dev server with hot reload
npm run build          # Compile TypeScript
npm start              # Run production server

# Database Management
npm run db:generate    # Generate Prisma client from schema
npm run db:push        # Push schema to database (dev only)
npm run db:migrate     # Create and run migrations (production)
npm run db:studio      # Open Prisma Studio GUI

# Testing
npm test               # Run all tests once
npm run test:watch     # Watch mode
npm run test:coverage  # Coverage report

# Code Quality
npm run lint           # ESLint
```

**Important:** Always run `db:generate` after modifying `prisma/schema.prisma`.

### ML Service (from /ml-service directory)
```bash
make test              # Run tests
make typecheck         # Type checking
make dev               # Start development server
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

### Backend (.env)
```env
DATABASE_URL="postgresql://user:password@localhost:5432/nutri_db"
PORT=3000
NODE_ENV=development
JWT_SECRET=your-secret-key-change-in-production
JWT_EXPIRES_IN=7d
```

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

| Path Pattern | Service | Test Command | Check Command |
|--------------|---------|--------------|---------------|
| `server/**/*.ts` | Backend API | `cd server && npm test` | `cd server && npm run build` |
| `app/**/*.tsx`, `lib/**/*.ts` | Mobile | `npm test` | `npx tsc --noEmit` |
| `ml-service/**/*.py` | ML Service | `cd ml-service && make test` | `cd ml-service && make typecheck` |
| `prisma/schema.prisma` | Database | - | `npm run db:generate` |

### Development Workflow

#### Starting Development
```bash
/dev                       # Use slash command
./scripts/start-all.sh     # Manual option
```

This starts: PostgreSQL (5432), Redis (6379), Backend API (3000)

#### Testing Flow
```bash
/test                      # Auto-detect what to test
/test server               # Test specific service
/test server:watch         # Watch mode
/test server:coverage      # Coverage
```

#### Code Quality
```bash
/check                     # Check everything
/check fix                 # Auto-fix issues
/check server              # Service-specific
```

### Implementation Workflow

1. **Use `/impl` command** for guided implementation
2. **Follow the stack order:**
   - Database schema (Prisma) → `db:generate` → `db:push`
   - Backend validation (Zod schemas)
   - Backend service/controller
   - Backend routes + tests
   - Mobile API client updates
   - Mobile UI components + tests
3. **Verify with:** `/check` and `/test`

### Hooks

**Post-Edit Hook:** Provides hints after editing files
- `*.ts` in server → "Run `/check server`"
- `schema.prisma` → "Run `/db generate`"
- Test files → "Run `/test`"

**Pre-Bash Hook:** Validates commands before execution
- Blocks dangerous patterns
- Warns about production database access

### Configuration Files

```
.claude/
├── commands/          # Slash commands (dev, test, check, db, build, status, fix, impl, log, api)
├── hooks/             # Automation hooks (nutri-post-edit.sh, nutri-pre-bash.sh)
└── settings.json      # Claude Code settings
```

### Best Practices

1. **Start sessions with `/status`** to verify services are running
2. **Use `/impl` for new features** for guided implementation
3. **Run `/check` before committing** to catch issues early
4. **Use `/test` liberally** - it's smart about what to test
5. **Trust the hooks** - they provide helpful reminders
6. **Ask for `/db` help** when working with schema changes

---

## Notes

- Codebase thoroughly refactored for type safety and code quality
- All controllers, services, and tests use strict TypeScript (zero `any` types)
- Security middleware is production-ready (rate limiting, sanitization)
- Test infrastructure is comprehensive with fixtures and assertion helpers
- Database schema supports 30+ health metric types and 17+ activity types
- ML service implementation is in progress (Python FastAPI)

---

**Last Updated:** December 2025
**Maintained By:** Nutri Development Team

---

*© 2025 Nutri | Claude Code Project Context*
