#!/bin/bash

# Nutri Complete Shutdown Script
# Stops ALL services: Backend API + Docker Services

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

print_success() {
    echo -e "${GREEN}✓${NC}  $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_header "🛑 Stopping ALL Nutri Services"

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# Step 1: Stop Backend API
# ============================================================================
print_header "📦 Step 1: Stopping Backend API"

# Check for PID file
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_info "Stopping backend API (PID: $BACKEND_PID)..."
        kill -15 $BACKEND_PID 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill -9 $BACKEND_PID 2>/dev/null || true
        fi
        print_success "Backend API stopped"
    else
        print_info "Backend API is not running"
    fi
    rm -f logs/backend.pid
else
    print_info "No backend PID file found"
fi

# Also try to kill by port (in case PID file is missing)
if lsof -ti:3000 >/dev/null 2>&1; then
    print_info "Stopping process on port 3000..."
    kill -9 $(lsof -ti:3000) 2>/dev/null || true
    print_success "Stopped process on port 3000"
fi

# ============================================================================
# Step 2: Stop Docker Services
# ============================================================================
print_info "Running stop-dev.sh to stop Docker services..."
./scripts/stop-dev.sh

echo ""
print_success "✅ All services stopped!"
echo ""
