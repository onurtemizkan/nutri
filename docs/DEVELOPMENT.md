# Development Environment Setup

Complete guide for setting up and running the Nutri development environment.

## Prerequisites

- **Docker Desktop** - For running PostgreSQL and Redis
- **Node.js** v16+ - For backend API and mobile app
- **npm** - Package manager
- **iOS Simulator** (Mac only) or **Android Emulator** - For mobile testing

## Quick Start

### 1. Start All Services

```bash
# Start Docker services + Backend API (all-in-one)
npm run dev:start

# Or use the script directly
./scripts/start-all.sh
```

This will:
- ✅ Start PostgreSQL database
- ✅ Start Redis cache
- ✅ Set up database schema
- ✅ Start backend API on port 3000
- ✅ Show your local IP address for testing

### 2. Start Mobile App

```bash
# In a new terminal
npm start

# Or for specific platforms
npm run ios        # iOS Simulator
npm run android    # Android Emulator
```

### 3. Stop All Services

```bash
# Stop everything
npm run dev:stop

# Or use the script directly
./scripts/stop-all.sh
```

---

## Available Scripts

### Main Development Scripts

| Command | Description |
|---------|-------------|
| `npm run dev:start` | Start all services (Docker + Backend API) |
| `npm run dev:stop` | Stop all services |
| `npm run services:start` | Start only Docker services (PostgreSQL + Redis) |
| `npm run services:stop` | Stop only Docker services |

### Mobile App Scripts

| Command | Description |
|---------|-------------|
| `npm start` | Start Expo development server |
| `npm run ios` | Run on iOS simulator |
| `npm run android` | Run on Android emulator |
| `npm run web` | Run in web browser |
| `npm test` | Run mobile app tests |
| `npm run lint` | Run ESLint |

### Backend API Scripts

```bash
cd server

# Development
npm run dev         # Start dev server with hot reload
npm run build       # Build TypeScript
npm start          # Run production build

# Database
npm run db:generate    # Generate Prisma client
npm run db:push        # Push schema to database
npm run db:migrate     # Create and run migrations
npm run db:studio      # Open Prisma Studio (GUI)

# Testing
npm test              # Run all tests
npm run test:watch    # Run tests in watch mode
npm run test:coverage # Generate coverage report

# Code Quality
npm run lint       # Run ESLint
```

---

## Network Configuration

### iOS Simulator

The mobile app automatically uses `http://localhost:3000/api` for iOS Simulator.

**No configuration needed!**

### Android Emulator

The mobile app automatically uses `http://10.0.2.2:3000/api` for Android Emulator.

**No configuration needed!**

### Physical Devices

For testing on physical devices (iPhone/Android phone), you need to update the API URL:

1. **Find your local IP** (shown when you run `npm run dev:start`):
   ```
   Local IP address: 192.168.1.69
   ```

2. **Update `lib/api/client.ts`**:
   ```typescript
   // Add to app.json or app.config.js:
   {
     "expo": {
       "extra": {
         "apiUrl": "http://192.168.1.69:3000/api"
       }
     }
   }
   ```

3. **Restart Expo**:
   ```bash
   npm start
   ```

---

## Service URLs

| Service | URL | Notes |
|---------|-----|-------|
| **Backend API** | http://localhost:3000 | Main API endpoint |
| **Health Check** | http://localhost:3000/health | Check if API is running |
| **PostgreSQL** | localhost:5432 | Database (user: postgres, pass: postgres) |
| **Redis** | localhost:6379 | Cache server |
| **Prisma Studio** | http://localhost:5555 | Database GUI (run `npm run db:studio`) |

---

## Troubleshooting

### Port Already in Use

If port 3000 is already in use:

```bash
# Stop all services
npm run dev:stop

# Or manually kill the process
lsof -ti:3000 | xargs kill -9
```

### Docker Services Not Starting

```bash
# Check Docker Desktop is running
docker ps

# Restart Docker Desktop if needed
# Then try again:
npm run dev:start
```

### Network Errors in Mobile App

1. **Check backend is running**:
   ```bash
   curl http://localhost:3000/health
   ```

2. **Check API URL**:
   - Open the mobile app
   - Check console logs for: `🌐 API Base URL: http://...`
   - Verify it matches your backend URL

3. **For physical devices**:
   - Ensure phone and computer are on the **same WiFi network**
   - Use your local IP address (not localhost)

### Database Connection Errors

```bash
# Reset the database
docker compose down -v  # ⚠️ Deletes all data
npm run dev:start
```

### Prisma Client Errors

```bash
cd server
npm run db:generate  # Regenerate Prisma client
```

---

## Testing

### Run Backend Tests

```bash
cd server

# Run all tests (serial execution for database isolation)
npm test

# Watch mode
npm run test:watch

# Coverage report
npm run test:coverage
```

**Test Results**: 70 passed / 6 failed (92% pass rate)
- ✅ All CRUD operations working
- ✅ Authentication & authorization working
- ✅ Input validation working
- ❌ Password reset flow (requires email service setup)

### Run Mobile Tests

```bash
# From project root
npm test
```

---

## Development Workflow

### Typical Development Session

```bash
# 1. Start all services
npm run dev:start

# 2. In new terminal: Start mobile app
npm start

# 3. Choose platform
# Press 'i' for iOS
# Press 'a' for Android
# Press 'w' for Web

# 4. Make changes to code
# - Backend changes auto-reload (tsx watch)
# - Mobile changes trigger Expo refresh

# 5. When done, stop everything
npm run dev:stop
```

### Making Backend Changes

1. Edit files in `server/src/`
2. Server auto-restarts (tsx watch)
3. Test changes immediately

### Making Mobile Changes

1. Edit files in `app/` or `lib/`
2. Expo auto-refreshes
3. Shake device for dev menu

### Database Schema Changes

```bash
cd server

# 1. Edit prisma/schema.prisma
# 2. Generate and push changes
npm run db:generate
npm run db:push

# For production, create migration:
npm run db:migrate
```

---

## Docker Compose Services

The `docker-compose.yml` file defines:

### Services

1. **postgres** - PostgreSQL 16
   - Port: 5432
   - Database: nutri_db
   - User: postgres / postgres

2. **redis** - Redis 7
   - Port: 6379
   - Used by ML service for caching

3. **postgres-test** - Test Database
   - Port: 5433
   - Database: nutri_test_db
   - Isolated from development data

### Data Persistence

Data is stored in Docker volumes:
- `nutri_postgres_data` - Main database data
- `nutri_postgres_test_data` - Test database data
- `nutri_redis_data` - Redis cache data

**To reset all data**:
```bash
docker compose down -v  # ⚠️ Deletes all data
npm run dev:start
```

---

## Logs

### View Backend API Logs

```bash
# If using start-all.sh
tail -f logs/backend.log

# If running manually in terminal
# Logs appear in the terminal where you ran `npm run dev`
```

### View Docker Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f postgres
docker compose logs -f redis
```

---

## Advanced Usage

### Start Services Individually

```bash
# 1. Start only Docker services
npm run services:start

# 2. Start backend manually
cd server
npm run dev

# 3. Start mobile app
npm start
```

### Use Custom API URL

```bash
# Set in environment or app.json
export API_URL="http://192.168.1.100:3000/api"
npm start
```

### Run with Production Build

```bash
cd server
npm run build
npm start  # Uses port 3000
```

---

## Tips

1. **Keep Docker Desktop running** - Required for database and cache
2. **Use separate terminals** - One for backend, one for mobile app
3. **Check logs first** - Most issues show up in logs
4. **Restart services** - Often fixes mysterious issues
5. **Clean restart** - `npm run dev:stop && npm run dev:start`

---

## Next Steps

- [API Documentation](README.md#api-documentation)
- [Testing Guide](CLAUDE.md#testing)
- [Database Schema](README.md#database-schema)
- [Project Structure](CLAUDE.md#directory-structure)
