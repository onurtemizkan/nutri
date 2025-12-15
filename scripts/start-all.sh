#!/bin/bash

# Nutri Complete Startup Script
# Starts ALL services: Docker + Backend API + (optionally) ML Service

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

print_header "🚀 Starting ALL Nutri Services"

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# Step 1: Start Docker Services
# ============================================================================
print_info "Running start-dev.sh to start Docker services..."
./scripts/start-dev.sh

# ============================================================================
# Step 2: Start Backend API
# ============================================================================
print_header "🚀 Step 2: Starting Backend API"

cd server

# Check if already running
if lsof -ti:3000 >/dev/null 2>&1; then
    print_warning "Backend API is already running on port 3000"
    print_info "Stop it with: ./scripts/stop-dev.sh"
else
    print_info "Starting backend API on port 3000..."

    # Start in background and save PID
    npm run dev > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../logs/backend.pid

    # Wait a moment for server to start
    sleep 3

    # Check if it's running
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_success "Backend API started (PID: $BACKEND_PID)"
        print_info "Logs: tail -f logs/backend.log"
    else
        print_error "Backend API failed to start. Check logs/backend.log"
        exit 1
    fi
fi

cd ..

# ============================================================================
# Step 3: Start ML Service
# ============================================================================
print_header "🧠 Step 3: Starting ML Service"

# Check if already running
if lsof -ti:8000 >/dev/null 2>&1; then
    print_warning "ML Service is already running on port 8000"
else
    print_info "Starting ML Service on port 8000..."

    # Check if venv exists
    if [ ! -f "ml-service/venv/bin/uvicorn" ]; then
        print_warning "ML Service venv not set up. Installing dependencies..."
        cd ml-service
        python3 -m venv venv 2>/dev/null || true
        ./venv/bin/pip install -r requirements.txt > ../logs/ml-service-install.log 2>&1
        cd ..
        print_success "ML Service dependencies installed"
    fi

    # Start ML service in background
    cd ml-service
    ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../logs/ml-service.log 2>&1 &
    ML_PID=$!
    echo $ML_PID > ../logs/ml-service.pid
    cd ..

    # Wait for server to start
    sleep 5

    # Check if it's running
    if lsof -ti:8000 >/dev/null 2>&1; then
        print_success "ML Service started (PID: $ML_PID)"
        print_info "Logs: tail -f logs/ml-service.log"
    else
        print_warning "ML Service may still be starting. Check logs/ml-service.log"
    fi
fi

# ============================================================================
# Step 4: Get Network Info
# ============================================================================
print_header "📱 Network Configuration"

# Get local IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
else
    LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
fi

print_success "Local IP: $LOCAL_IP"
print_success "Backend API: http://localhost:3000"
print_success "Health Check: http://localhost:3000/health"

# ============================================================================
# Step 5: Test Services
# ============================================================================
print_header "🔍 Testing Services"

print_info "Checking Backend API health endpoint..."
sleep 2  # Give server a moment to fully start

if curl -s http://localhost:3000/health >/dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:3000/health)
    print_success "Backend API is healthy!"
    echo "    Response: $HEALTH_RESPONSE"
else
    print_warning "Backend API health check failed. Server may still be starting..."
fi

print_info "Checking ML Service health endpoint..."
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
print_header "✅ All Services Running"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎉 ${GREEN}SUCCESS!${NC} All services are running!"
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
echo "  • Docker:          ${BLUE}docker-compose logs -f${NC}"
echo ""
echo "🛑 Stop all services:"
echo "  • ${GREEN}./scripts/stop-all.sh${NC}"
echo ""
echo "📱 Next Steps:"
echo "  1. Update ${BLUE}lib/api/client.ts${NC} with:"
echo "     ${YELLOW}http://$LOCAL_IP:3000/api${NC}"
echo ""
echo "  2. Start mobile app:"
echo "     ${GREEN}npm start${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
