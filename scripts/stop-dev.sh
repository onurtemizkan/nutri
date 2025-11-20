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

# ============================================================================
# Step 1: Stop Node.js Backend (if running)
# ============================================================================
print_header "📦 Step 1: Stopping Backend API"

if lsof -ti:3000 >/dev/null 2>&1; then
    print_info "Stopping backend server on port 3000..."
    kill -9 $(lsof -ti:3000) 2>/dev/null || true
    print_success "Backend server stopped"
else
    print_info "Backend server is not running"
fi

# ============================================================================
# Step 2: Stop ML Service (if running)
# ============================================================================
print_header "🤖 Step 2: Stopping ML Service"

if lsof -ti:8000 >/dev/null 2>&1; then
    print_info "Stopping ML service on port 8000..."
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
