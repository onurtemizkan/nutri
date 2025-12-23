#!/bin/bash

# Nutri Backend + ML Service Startup Script
# Starts Backend API and ML Service together (assumes Docker is already running)

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

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_error() {
    echo -e "${RED}✗${NC}  $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to check if a port is in use
port_in_use() {
    lsof -i ":$1" >/dev/null 2>&1
}

print_header "🚀 Starting Backend API + ML Service"

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create logs directory if needed
mkdir -p "$PROJECT_ROOT/logs"

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================
print_header "📋 Step 1: Checking Prerequisites"

# Check if Docker services are running (PostgreSQL)
if ! docker ps | grep -q "nutri-postgres"; then
    print_warning "PostgreSQL is not running!"
    print_info "Run './scripts/start-dev.sh' first to start all services including ML."
    print_info "Or run 'docker compose up -d postgres redis' to start just Docker."
    exit 1
fi
print_success "PostgreSQL is running"

# Check if Redis is running
if ! docker ps | grep -q "nutri-redis"; then
    print_warning "Redis is not running!"
    print_info "Run 'docker compose up -d redis' to start Redis."
    exit 1
fi
print_success "Redis is running"

# ============================================================================
# Step 2: Start ML Service
# ============================================================================
print_header "🧠 Step 2: Starting ML Service"

# Check if already running
if port_in_use 8000; then
    print_warning "ML Service is already running on port 8000"
else
    print_info "Starting ML Service on port 8000..."

    # Check if venv exists
    if [ ! -f "$PROJECT_ROOT/ml-service/venv/bin/uvicorn" ]; then
        print_warning "ML Service venv not set up. Installing dependencies..."
        cd "$PROJECT_ROOT/ml-service"
        python3 -m venv venv 2>/dev/null || true
        ./venv/bin/pip install -r requirements.txt > "$PROJECT_ROOT/logs/ml-service-install.log" 2>&1
        cd "$PROJECT_ROOT"
        print_success "ML Service dependencies installed"
    fi

    # Start ML service in background with hot reload
    cd "$PROJECT_ROOT/ml-service"
    ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > "$PROJECT_ROOT/logs/ml-service.log" 2>&1 &
    ML_PID=$!
    echo $ML_PID > "$PROJECT_ROOT/logs/ml-service.pid"
    cd "$PROJECT_ROOT"

    # Wait for server to start
    sleep 3

    # Check if it's running
    if port_in_use 8000; then
        print_success "ML Service started (PID: $ML_PID)"
    else
        print_warning "ML Service may still be starting. Check logs/ml-service.log"
    fi
fi

# ============================================================================
# Step 3: Start Backend API
# ============================================================================
print_header "📦 Step 3: Starting Backend API"

# Check if already running
if port_in_use 3000; then
    print_warning "Backend API is already running on port 3000"
else
    print_info "Starting backend API on port 3000..."

    # Ensure node_modules exists
    if [ ! -d "$PROJECT_ROOT/server/node_modules" ]; then
        print_info "Installing backend dependencies..."
        cd "$PROJECT_ROOT/server"
        npm install
        cd "$PROJECT_ROOT"
        print_success "Dependencies installed"
    fi

    # Start backend in background
    cd "$PROJECT_ROOT/server"
    npm run dev > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$PROJECT_ROOT/logs/backend.pid"
    cd "$PROJECT_ROOT"

    # Wait for server to start
    sleep 3

    # Check if it's running
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_success "Backend API started (PID: $BACKEND_PID)"
    else
        print_error "Backend API failed to start. Check logs/backend.log"
        exit 1
    fi
fi

# ============================================================================
# Step 4: Test Services
# ============================================================================
print_header "🔍 Step 4: Testing Services"

print_info "Waiting for services to be ready..."
sleep 2

# Test Backend API
if curl -s http://localhost:3000/health >/dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:3000/health)
    print_success "Backend API is healthy!"
    echo "    Response: $HEALTH_RESPONSE"
else
    print_warning "Backend API health check failed. Server may still be starting..."
fi

# Test ML Service
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    ML_HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    print_success "ML Service is healthy!"
    echo "    Response: $ML_HEALTH_RESPONSE"
else
    print_warning "ML Service health check failed. Server may still be starting..."
fi

# ============================================================================
# Final Status
# ============================================================================
print_header "✅ Services Running"

# Get local IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
else
    LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎉 ${GREEN}SUCCESS!${NC} Backend API + ML Service running!"
echo ""
echo "📊 Service URLs:"
echo "  • Backend API:     ${BLUE}http://localhost:3000${NC}"
echo "  • ML Service:      ${BLUE}http://localhost:8000${NC}"
echo "  • Health Check:    ${BLUE}http://localhost:3000/health${NC}"
echo "  • ML Health:       ${BLUE}http://localhost:8000/health${NC}"
echo "  • For Simulator:   ${YELLOW}http://$LOCAL_IP:3000/api${NC}"
echo ""
echo "📝 Logs:"
echo "  • Backend API:     ${BLUE}tail -f logs/backend.log${NC}"
echo "  • ML Service:      ${BLUE}tail -f logs/ml-service.log${NC}"
echo ""
echo "🛑 Stop services:"
echo "  • ${GREEN}./scripts/stop-all.sh${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
