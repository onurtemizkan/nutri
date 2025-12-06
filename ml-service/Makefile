.PHONY: help setup setup-env setup-venv setup-deps setup-redis
.PHONY: test test-watch test-coverage test-phase1 test-phase2 test-phase3 test-pipeline test-fast test-debug test-with-redis
.PHONY: redis-start redis-stop redis-status redis-flush redis-cli
.PHONY: dev dev-with-redis
.PHONY: lint format format-check typecheck
.PHONY: clean clean-models clean-all
.PHONY: docker-up docker-down docker-logs docker-test
.PHONY: info health

# Colors for output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
BLUE   := \033[0;34m
NC     := \033[0m # No Color

# Python & Virtual Environment
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PYTHON_VENV := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

# Redis
REDIS_PORT := 6379

help: ## Show this help message
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║         Nutri ML Service - Available Commands           ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@grep -E '^setup[a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing Commands:$(NC)"
	@grep -E '^test[a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Redis Commands:$(NC)"
	@grep -E '^redis[a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@grep -E '^(dev|lint|format|typecheck)[a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@grep -E '^docker[a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Utility Commands:$(NC)"
	@grep -E '^(clean|info|health)[a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# Setup Commands
# ============================================================================

setup: setup-env setup-venv setup-deps ## Complete setup (env + venv + dependencies)
	@echo "$(GREEN)✅ Setup complete!$(NC)"

setup-env: ## Create .env files from examples
	@echo "$(BLUE)🔧 Setting up environment files...$(NC)"
	@node scripts/setup-env.js

setup-venv: ## Create Python virtual environment
	@echo "$(BLUE)🐍 Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✅ Virtual environment created$(NC)"

setup-deps: $(VENV_BIN)/activate ## Install Python dependencies
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

setup-redis: ## Install and configure Redis
	@bash scripts/setup-redis.sh

# ============================================================================
# Testing Commands
# ============================================================================

test: $(VENV_BIN)/activate ## Run all tests
	@$(PYTHON_VENV) -m pytest tests/ -v --tb=short

test-watch: $(VENV_BIN)/activate ## Run tests in watch mode
	@$(PYTHON_VENV) -m ptw tests/ -- -v --tb=short

test-coverage: $(VENV_BIN)/activate ## Run tests with coverage report
	@$(PYTHON_VENV) -m pytest tests/ --cov=app --cov-report=html --cov-report=term
	@echo ""
	@echo "$(GREEN)📊 Coverage report generated: htmlcov/index.html$(NC)"

test-fast: $(VENV_BIN)/activate ## Run tests (fast mode - stop on first failure)
	@$(PYTHON_VENV) -m pytest tests/ -v --tb=line -x

test-debug: $(VENV_BIN)/activate ## Run tests in debug mode (verbose output)
	@$(PYTHON_VENV) -m pytest tests/ -v --tb=long -s

test-with-redis: redis-start test redis-stop ## Run tests with Redis running

# ============================================================================
# Redis Commands
# ============================================================================

redis-start: ## Start Redis server
	@bash scripts/redis-start.sh

redis-stop: ## Stop Redis server
	@bash scripts/redis-stop.sh

redis-status: ## Check Redis status
	@bash scripts/redis-status.sh

redis-flush: ## Flush all Redis data
	@bash scripts/redis-flush.sh

redis-cli: ## Open Redis CLI
	@redis-cli -p $(REDIS_PORT)

# ============================================================================
# Development Commands
# ============================================================================

dev: $(VENV_BIN)/activate ## Start development server
	@$(PYTHON_VENV) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-with-redis: redis-start dev ## Start development server with Redis

lint: $(VENV_BIN)/activate ## Run linter
	@$(PYTHON_VENV) -m flake8 app/ tests/

format: $(VENV_BIN)/activate ## Format code with black
	@$(PYTHON_VENV) -m black app/ tests/

format-check: $(VENV_BIN)/activate ## Check code formatting
	@$(PYTHON_VENV) -m black --check app/ tests/

typecheck: $(VENV_BIN)/activate ## Run type checker
	@$(PYTHON_VENV) -m mypy app/ --config-file mypy.ini

# ============================================================================
# Docker Commands
# ============================================================================

docker-up: ## Start all Docker services
	@docker-compose up -d

docker-down: ## Stop all Docker services
	@docker-compose down

docker-logs: ## Show Docker logs for ml-service
	@docker-compose logs -f ml-service

docker-test: ## Run tests in Docker
	@docker-compose exec ml-service pytest tests/ -v

# ============================================================================
# Utility Commands
# ============================================================================

clean: ## Clean Python cache files
	@echo "$(BLUE)🧹 Cleaning cache files...$(NC)"
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "$(GREEN)✅ Cache cleaned$(NC)"

clean-models: ## Clean trained model files
	@echo "$(BLUE)🧹 Cleaning model files...$(NC)"
	@rm -rf models/*.pth 2>/dev/null || true
	@echo "$(GREEN)✅ Model files cleaned$(NC)"

clean-all: clean clean-models ## Clean everything including venv
	@echo "$(BLUE)🧹 Cleaning virtual environment...$(NC)"
	@rm -rf $(VENV)/ 2>/dev/null || true
	@echo "$(GREEN)✅ Complete cleanup done$(NC)"

info: ## Show project information
	@node scripts/info.js

health: ## Check service health
	@curl -f http://localhost:8000/health || echo "$(RED)❌ Service not running$(NC)"

# ============================================================================
# Internal Targets
# ============================================================================

$(VENV_BIN)/activate:
	@echo "$(YELLOW)⚠️  Virtual environment not found. Creating...$(NC)"
	@make setup-venv
	@make setup-deps
