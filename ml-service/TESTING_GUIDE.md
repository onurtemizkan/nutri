# Testing Guide - Nutri ML Service

Complete guide for setting up the test environment and running tests.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running Tests](#running-tests)
- [Redis Support](#redis-support)
- [Development Workflow](#development-workflow)
- [Test Phases](#test-phases)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

```bash
# Complete setup (first time only)
npm run setup              # or: make setup

# Run all tests
npm test                   # or: make test

# Run specific phase
npm run test:phase2        # or: make test-phase2

# Start dev server
npm run dev                # or: make dev
```

---

## ✅ Prerequisites

### Required:
- **Python 3.11+** - ML service backend
- **Node.js 16+** - For npm script automation
- **pip** - Python package manager

### Optional (but recommended):
- **Redis** - For caching tests (1 test requires Redis)
- **Make** - For Makefile commands (alternative to npm)
- **Docker** - For containerized deployment

### Check your setup:
```bash
npm run info               # or: make info
```

---

## 🔧 Setup

### 1. Complete Setup (First Time)

```bash
npm run setup
```

This will:
1. ✅ Create `.env` and `.env.test` files
2. ✅ Create Python virtual environment (`venv/`)
3. ✅ Install all dependencies from `requirements.txt`

### 2. Manual Setup (Step by Step)

```bash
# Create environment files
npm run setup:env          # or: make setup-env

# Create virtual environment
npm run setup:venv         # or: make setup-venv

# Install dependencies
npm run setup:deps         # or: make setup-deps
```

### 3. Redis Setup (Optional)

```bash
# Install Redis (macOS/Linux)
npm run setup:redis        # or: make setup-redis

# Start Redis
npm run redis:start        # or: make redis-start

# Check Redis status
npm run redis:status       # or: make redis-status
```

---

## 🧪 Running Tests

### All Tests

```bash
npm test                   # Run all 27 tests
npm run test:coverage      # With coverage report
npm run test:fast          # Stop on first failure
npm run test:debug         # Verbose debug output
```

### By Phase

```bash
npm run test:phase1        # Feature Engineering (8 tests)
npm run test:phase2        # LSTM Training (10 tests)
npm run test:phase3        # Interpretability (7 tests)
npm run test:pipeline      # Full Pipeline (2 tests)
```

### With Redis

```bash
npm run test:with-redis    # Start Redis → Run tests → Stop Redis
```

### Makefile Alternatives

```bash
make test                  # All tests
make test-coverage         # With coverage
make test-phase2           # Phase 2 only
make test-with-redis       # With Redis
```

---

## 📊 Test Results (Current)

| Phase | Tests | Status | Time |
|-------|-------|--------|------|
| **Phase 1** | 7/8 (87.5%) | ✅ Passing | ~8s |
| **Phase 2** | 10/10 (100%) | ✅ Complete | ~17s |
| **Phase 3** | 7/7 (100%) | ✅ Complete | ~21s |
| **Full Pipeline** | 2/2 (100%) | ✅ Complete | ~8s |
| **TOTAL** | **26/27 (96.3%)** | 🎉 **Success** | ~53s |

**Note**: 1 test (`test_feature_engineering_caching`) requires Redis server running.

---

## 🔴 Redis Support

Redis is **optional** but enables caching tests.

### Install Redis

**macOS (Homebrew)**:
```bash
npm run setup:redis
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update && sudo apt-get install redis-server
```

**Linux (CentOS/RHEL)**:
```bash
sudo yum install redis
```

### Redis Commands

```bash
# Start Redis server
npm run redis:start        # or: make redis-start

# Check status
npm run redis:status       # or: make redis-status

# Stop Redis
npm run redis:stop         # or: make redis-stop

# Flush all data
npm run redis:flush        # or: make redis-flush

# Open Redis CLI
npm run redis:cli          # or: make redis-cli
```

### Redis Status Output Example

```
📊 Redis Status Check
====================

✅ Redis is RUNNING
   PID: 12345
   Port: 6379

📡 Connection Status
   Status: PONG
   Port: 6379

📈 Server Information
   redis_version:7.2.0
   os:Darwin 24.6.0
   tcp_port:6379
   uptime_in_seconds:120

💾 Memory Usage
   used_memory_human:1.23M

🔑 Keyspace
   Total keys: 0
```

---

## 🛠️ Development Workflow

### Start Development Server

```bash
npm run dev                # or: make dev
npm run dev:with-redis     # or: make dev-with-redis
```

Server runs on: `http://localhost:8000`

API docs: `http://localhost:8000/docs`

### Code Quality

```bash
# Format code
npm run format             # or: make format

# Check formatting
npm run format:check       # or: make format-check

# Run linter
npm run lint               # or: make lint

# Type checking
npm run typecheck          # or: make typecheck
```

### Clean Up

```bash
# Clean cache files
npm run clean              # or: make clean

# Clean model files
npm run clean:models       # or: make clean-models

# Clean everything (including venv)
npm run clean:all          # or: make clean-all
```

---

## 📦 Test Phases Explained

### Phase 1: Feature Engineering & Correlation (8 tests)

Tests feature extraction, correlation analysis, and lag analysis.

**Tests**:
- ✅ Basic feature generation
- ✅ Feature validation
- ⚠️ Feature caching (requires Redis)
- ✅ Correlation analysis (RHR)
- ✅ Correlation analysis (HRV)
- ✅ Correlation summary endpoint
- ✅ Lag analysis (delayed effects)
- ✅ Lag analysis endpoint

**Run**: `npm run test:phase1`

### Phase 2: LSTM Training & Predictions (10 tests)

Tests LSTM model training, predictions, and model management.

**Tests**:
- ✅ LSTM training (RHR metric)
- ✅ LSTM training (HRV metric)
- ✅ Early stopping mechanism
- ✅ Single predictions
- ✅ Batch predictions
- ✅ Prediction caching
- ✅ Model listing
- ✅ Model deletion
- ✅ Error handling (missing models)
- ✅ Error handling (insufficient data)

**Run**: `npm run test:phase2`

### Phase 3: Model Interpretability (7 tests)

Tests SHAP explainability, what-if analysis, and counterfactual scenarios.

**Tests**:
- ✅ SHAP global importance
- ✅ SHAP local explanations
- ✅ What-if single scenario
- ✅ What-if multiple scenarios
- ✅ Counterfactual target value
- ✅ Counterfactual improvements
- ✅ Complete interpretability workflow

**Run**: `npm run test:phase3`

### Phase 4: Full Pipeline (2 tests)

Integration tests covering the complete ML pipeline.

**Tests**:
- ✅ Complete ML pipeline (feature → train → predict → interpret)
- ✅ Multi-metric pipeline

**Run**: `npm run test:pipeline`

---

## 🐳 Docker Testing

### Run Tests in Docker

```bash
# Start all services
npm run docker:up          # or: make docker-up

# Run tests
npm run docker:test        # or: make docker-test

# View logs
npm run docker:logs        # or: make docker-logs

# Stop services
npm run docker:down        # or: make docker-down
```

---

## 🔍 Troubleshooting

### Virtual Environment Not Activated

**Error**: `command not found: pytest`

**Solution**:
```bash
source venv/bin/activate   # Activate manually
# or
npm run setup:venv         # Recreate venv
```

### Redis Connection Failed

**Error**: `Redis not connected, skipping cache`

**Solution**:
```bash
npm run redis:start        # Start Redis
npm run redis:status       # Verify running
```

### Redis Port Already in Use

**Error**: `Port 6379 is already in use`

**Solution**:
```bash
npm run redis:stop         # Stop existing Redis
# or
lsof -i :6379              # Find process using port
kill -9 <PID>              # Kill process
```

### Tests Failing After Code Changes

**Solution**:
```bash
npm run clean              # Clean cache
npm run test:fast          # Find first failure
npm run test:debug         # Verbose output
```

### PyTorch/CUDA Issues

**Error**: `RuntimeError: CUDA out of memory` (unlikely on CPU)

**Solution**:
```python
# Tests use CPU by default (no GPU required)
# Check app/ml_models/lstm.py device selection
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'app'`

**Solution**:
```bash
# Ensure you're in project root
pwd  # Should be: .../ml-service

# Reinstall dependencies
npm run setup:deps
```

### Model Training Too Slow

Tests should complete in ~53 seconds total. If much slower:

```bash
# Check system resources
top

# Reduce test epochs (already minimal)
# Check app/ml_models/lstm.py training config
```

---

## 📚 Additional Resources

### Configuration Files

- `.env` - Development configuration
- `.env.test` - Test configuration
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker services

### Documentation

- `README.md` - Project overview
- `TESTING_GUIDE.md` - This file

### Key Directories

```
ml-service/
├── app/                   # Application code
│   ├── ml_models/        # LSTM models
│   ├── services/         # Business logic
│   └── api/              # API endpoints
├── tests/                # Test suite
│   ├── fixtures.py       # Test fixtures
│   ├── conftest.py       # Pytest configuration
│   └── test_*.py         # Test files
├── scripts/              # Automation scripts
│   ├── setup-env.js      # Environment setup
│   ├── setup-redis.sh    # Redis installation
│   ├── redis-start.sh    # Start Redis
│   ├── redis-stop.sh     # Stop Redis
│   ├── redis-status.sh   # Check Redis
│   └── info.js           # Project info
├── models/               # Trained models (*.pth)
└── venv/                 # Virtual environment
```

---

## 🎯 Best Practices

### Before Committing

```bash
make format                # Format code
make lint                  # Check linting
make test                  # Run all tests
make clean                 # Clean cache
```

### Daily Workflow

```bash
make test-fast             # Quick check
# ... make changes ...
make test-phase2           # Test affected phase
make test                  # Full test suite
```

### CI/CD Pipeline

```bash
make setup                 # Setup environment
make test-coverage         # Tests with coverage
make lint                  # Code quality checks
```

---

## 📞 Support

**Issues**: https://github.com/yourusername/nutri/issues

**Documentation**: See `README.md` for project overview

**Test Status**: See `CURRENT_SESSION_PROGRESS.md` for detailed results

---

**Last Updated**: 2025-11-17 | **Test Success Rate**: 96.3% (26/27 passing) ✅
