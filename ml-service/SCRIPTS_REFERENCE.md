# Scripts Reference - Quick Command Guide

Quick reference for all npm scripts and Makefile commands.

## 📦 Setup Commands

| npm | make | Description |
|-----|------|-------------|
| `npm run setup` | `make setup` | Complete setup (env + venv + deps) |
| `npm run setup:env` | `make setup-env` | Create .env files |
| `npm run setup:venv` | `make setup-venv` | Create virtual environment |
| `npm run setup:deps` | `make setup-deps` | Install Python dependencies |
| `npm run setup:redis` | `make setup-redis` | Install Redis |

## 🧪 Testing Commands

| npm | make | Description |
|-----|------|-------------|
| `npm test` | `make test` | Run all tests |
| `npm run test:coverage` | `make test-coverage` | Run with coverage report |
| `npm run test:phase1` | `make test-phase1` | Run Phase 1 (Feature Engineering) |
| `npm run test:phase2` | `make test-phase2` | Run Phase 2 (LSTM Training) |
| `npm run test:phase3` | `make test-phase3` | Run Phase 3 (Interpretability) |
| `npm run test:pipeline` | `make test-pipeline` | Run Full Pipeline tests |
| `npm run test:fast` | `make test-fast` | Fast mode (stop on first fail) |
| `npm run test:debug` | `make test-debug` | Debug mode (verbose) |
| `npm run test:with-redis` | `make test-with-redis` | Run with Redis |

## 📡 Redis Commands

| npm | make | Description |
|-----|------|-------------|
| `npm run redis:start` | `make redis-start` | Start Redis server |
| `npm run redis:stop` | `make redis-stop` | Stop Redis server |
| `npm run redis:status` | `make redis-status` | Check Redis status |
| `npm run redis:flush` | `make redis-flush` | Flush all Redis data |
| `npm run redis:cli` | `make redis-cli` | Open Redis CLI |

## 🛠️ Development Commands

| npm | make | Description |
|-----|------|-------------|
| `npm run dev` | `make dev` | Start dev server |
| `npm run dev:with-redis` | `make dev-with-redis` | Start dev + Redis |
| `npm run lint` | `make lint` | Run linter |
| `npm run format` | `make format` | Format code |
| `npm run format:check` | `make format-check` | Check formatting |
| `npm run typecheck` | `make typecheck` | Run type checker |

## 🐳 Docker Commands

| npm | make | Description |
|-----|------|-------------|
| `npm run docker:up` | `make docker-up` | Start Docker services |
| `npm run docker:down` | `make docker-down` | Stop Docker services |
| `npm run docker:logs` | `make docker-logs` | Show Docker logs |
| `npm run docker:test` | `make docker-test` | Run tests in Docker |

## 🧹 Cleanup Commands

| npm | make | Description |
|-----|------|-------------|
| `npm run clean` | `make clean` | Clean cache files |
| `npm run clean:models` | `make clean-models` | Clean model files |
| `npm run clean:all` | `make clean-all` | Clean everything |

## ℹ️ Utility Commands

| npm | make | Description |
|-----|------|-------------|
| `npm run info` | `make info` | Show project info |
| `npm run health` | `make health` | Check service health |
| - | `make help` | Show Makefile help |

## 🚀 Common Workflows

### First Time Setup
```bash
npm run setup              # Complete setup
npm run redis:start        # Start Redis
npm test                   # Run all tests
```

### Daily Development
```bash
npm run dev                # Start dev server
npm run test:fast          # Quick test check
npm run test:phase2        # Test specific phase
```

### Before Commit
```bash
npm run format             # Format code
npm run lint               # Check linting
npm test                   # Run all tests
```

### Full CI/CD Check
```bash
npm run clean              # Clean cache
npm run setup:deps         # Ensure deps updated
npm run test:coverage      # Tests with coverage
npm run lint               # Code quality
```

## 📁 Script Files Location

All helper scripts are in the `scripts/` directory:

```
scripts/
├── setup-env.js           # Node.js env setup
├── setup-redis.sh         # Redis installation
├── redis-start.sh         # Start Redis
├── redis-stop.sh          # Stop Redis
├── redis-status.sh        # Redis status
├── redis-flush.sh         # Flush Redis
└── info.js                # Project info
```

## 🔧 Environment Files

- `.env` - Development configuration (auto-generated)
- `.env.test` - Test configuration (auto-generated)
- `.env.example` - Template file (committed to git)

## 📊 Test Suite Summary

| Phase | Tests | Duration | Status |
|-------|-------|----------|--------|
| Phase 1 | 8 tests | ~8s | 7/8 (87.5%) |
| Phase 2 | 10 tests | ~17s | 10/10 (100%) |
| Phase 3 | 7 tests | ~21s | 7/7 (100%) |
| Pipeline | 2 tests | ~8s | 2/2 (100%) |
| **Total** | **27 tests** | **~53s** | **26/27 (96.3%)** |

## 💡 Tips

1. **Use npm or make** - Both work identically, use whichever you prefer
2. **Redis is optional** - Only 1 test requires it, all others work without
3. **Fast feedback** - Use `test:fast` to stop on first failure
4. **Coverage reports** - Generated in `htmlcov/index.html`
5. **Clean between runs** - Use `clean` if tests behave unexpectedly

## 🆘 Need Help?

```bash
make help                  # Show all Makefile commands
npm run info               # Show project status
npm run redis:status       # Check Redis
```

---

**Quick Start**: `npm run setup && npm test` 🚀
