#!/bin/bash

# Nutri Development Environment Shutdown Script
# Stops all running services

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect docker compose command
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"  # Default to v2
fi

print_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

print_success() {
    echo -e "${GREEN}✓${NC}  $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_header "🛑 Stopping Nutri Development Environment"

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# Step 1: Stop Node.js Backend (if running)
# ============================================================================
print_header "📦 Step 1: Stopping Backend API"

# Try PID file first (graceful shutdown)
if [ -f "$PROJECT_ROOT/logs/backend.pid" ]; then
    BACKEND_PID=$(cat "$PROJECT_ROOT/logs/backend.pid")
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_info "Stopping backend API (PID: $BACKEND_PID)..."
        kill -15 $BACKEND_PID 2>/dev/null || true
        sleep 1
        # Force kill if still running
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill -9 $BACKEND_PID 2>/dev/null || true
        fi
    fi
    rm -f "$PROJECT_ROOT/logs/backend.pid"
fi

# Also kill by port (in case PID file is missing)
if lsof -ti:3000 >/dev/null 2>&1; then
    print_info "Stopping process on port 3000..."
    kill -9 $(lsof -ti:3000) 2>/dev/null || true
    print_success "Backend server stopped"
else
    print_info "Backend server is not running"
fi

# ============================================================================
# Step 2: Stop ML Service (if running)
# ============================================================================
print_header "🧠 Step 2: Stopping ML Service"

# Try PID file first (graceful shutdown)
if [ -f "$PROJECT_ROOT/logs/ml-service.pid" ]; then
    ML_PID=$(cat "$PROJECT_ROOT/logs/ml-service.pid")
    if kill -0 $ML_PID 2>/dev/null; then
        print_info "Stopping ML Service (PID: $ML_PID)..."
        kill -15 $ML_PID 2>/dev/null || true
        sleep 1
        # Force kill if still running
        if kill -0 $ML_PID 2>/dev/null; then
            kill -9 $ML_PID 2>/dev/null || true
        fi
    fi
    rm -f "$PROJECT_ROOT/logs/ml-service.pid"
fi

# Also kill by port (in case PID file is missing)
if lsof -ti:8000 >/dev/null 2>&1; then
    print_info "Stopping process on port 8000..."
    kill -9 $(lsof -ti:8000) 2>/dev/null || true
    print_success "ML service stopped"
else
    print_info "ML service is not running"
fi

# ============================================================================
# Step 3: Stop Docker Services
# ============================================================================
print_header "🐳 Step 3: Stopping Docker Services"

if $DOCKER_COMPOSE ps | grep -q "Up"; then
    print_info "Stopping Docker services..."
    $DOCKER_COMPOSE down
    print_success "Docker services stopped"
else
    print_info "Docker services are not running"
fi

# ============================================================================
# Step 4: Display Status
# ============================================================================
print_header "📊 Final Status"

echo ""
print_success "✅ All services stopped successfully!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To restart services, run:"
echo "  ${GREEN}./scripts/start-dev.sh${NC}"
echo ""
echo "To remove all Docker volumes (⚠️  deletes data):"
echo "  ${YELLOW}docker-compose down -v${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
