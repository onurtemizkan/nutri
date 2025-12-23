#!/bin/bash

# Nutri Development Environment Startup Script
# Starts all required services for local development

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for a service to be healthy
wait_for_service() {
    local service_name=$1
    local max_attempts=30
    local attempt=1

    print_info "Waiting for $service_name to be healthy..."

    while [ $attempt -le $max_attempts ]; do
        if $DOCKER_COMPOSE ps | grep "$service_name" | grep -q "healthy"; then
            print_success "$service_name is healthy!"
            return 0
        fi

        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    print_error "$service_name failed to become healthy within $max_attempts seconds"
    return 1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i ":$1" >/dev/null 2>&1
}

print_header "🚀 Starting Nutri Development Environment"

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================
print_header "📋 Step 1: Checking Prerequisites"

if ! command_exists docker; then
    print_error "Docker is not installed. Please install Docker Desktop."
    exit 1
fi
print_success "Docker found"

# Check for docker compose (v2 plugin) or docker-compose (v1 standalone)
if command_exists "docker compose" || docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
    print_success "Docker Compose v2 found"
elif command_exists docker-compose; then
    DOCKER_COMPOSE="docker-compose"
    print_success "Docker Compose v1 found"
else
    print_error "Docker Compose is not installed."
    exit 1
fi

if ! command_exists node; then
    print_error "Node.js is not installed."
    exit 1
fi
print_success "Node.js found ($(node --version))"

if ! command_exists npm; then
    print_error "npm is not installed."
    exit 1
fi
print_success "npm found ($(npm --version))"

# ============================================================================
# Step 2: Start Docker Services
# ============================================================================
print_header "🐳 Step 2: Starting Docker Services"

print_info "Starting PostgreSQL and Redis..."
$DOCKER_COMPOSE up -d postgres redis

# Wait for services to be healthy
wait_for_service "nutri-postgres" || exit 1
wait_for_service "nutri-redis" || exit 1

print_success "Docker services started successfully!"

# ============================================================================
# Step 3: Setup Database Schema
# ============================================================================
print_header "🗄️  Step 3: Setting Up Database Schema"

cd server

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_info "Installing backend dependencies..."
    npm install
    print_success "Dependencies installed"
fi

# Generate Prisma client
print_info "Generating Prisma client..."
npm run db:generate >/dev/null 2>&1
print_success "Prisma client generated"

# Push schema to database (--accept-data-loss for non-interactive mode)
print_info "Pushing schema to database..."
npx prisma db push --accept-data-loss >/dev/null 2>&1
print_success "Database schema is up to date"

cd ..

# ============================================================================
# Step 4: Check Port Availability
# ============================================================================
print_header "🔍 Step 4: Checking Port Availability"

if port_in_use 3000; then
    print_warning "Port 3000 is already in use. Backend API will fail to start."
    print_info "Run './scripts/stop-dev.sh' to stop existing services."
else
    print_success "Port 3000 is available"
fi

if port_in_use 8000; then
    print_warning "Port 8000 is already in use (ML Service)."
else
    print_success "Port 8000 is available"
fi

# ============================================================================
# Step 5: Start ML Service
# ============================================================================
print_header "🧠 Step 5: Starting ML Service"

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create logs directory if needed
mkdir -p "$PROJECT_ROOT/logs"

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

    # Start ML service in background
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
        print_info "Logs: tail -f logs/ml-service.log"
    else
        print_warning "ML Service may still be starting. Check logs/ml-service.log"
    fi
fi

# ============================================================================
# Step 6: Get Local IP Address
# ============================================================================
print_header "📱 Step 6: Network Configuration"

# Get local IP address for simulator/device access
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
else
    # Linux
    LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
fi

print_success "Local IP address: $LOCAL_IP"
print_info "Update lib/api/client.ts to use: http://$LOCAL_IP:3000/api"

# ============================================================================
# Step 7: Test ML Service Health
# ============================================================================
print_header "🔍 Step 7: Testing ML Service"

print_info "Checking ML Service health endpoint..."
sleep 2  # Give server a moment to fully start

if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    ML_HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    print_success "ML Service is healthy!"
    echo "    Response: $ML_HEALTH_RESPONSE"
else
    print_warning "ML Service health check failed. Server may still be starting..."
    print_info "Check logs: tail -f logs/ml-service.log"
fi

# ============================================================================
# Step 8: Display Service Status
# ============================================================================
print_header "📊 Service Status"

echo ""
echo "Docker Services:"
$DOCKER_COMPOSE ps

echo ""
print_success "✅ All services are ready!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎉 ${GREEN}SUCCESS!${NC} Infrastructure + ML Service running!"
echo ""
echo "📊 Service URLs:"
echo "  • ML Service:      ${BLUE}http://localhost:8000${NC}"
echo "  • ML Health:       ${BLUE}http://localhost:8000/health${NC}"
echo ""
echo "📝 Logs:"
echo "  • ML Service:      ${BLUE}tail -f logs/ml-service.log${NC}"
echo "  • Docker:          ${BLUE}docker-compose logs -f${NC}"
echo ""
echo "📌 Next Steps:"
echo ""
echo "  1. Start Backend API:"
echo "     ${GREEN}cd server && npm run dev${NC}"
echo ""
echo "  2. Start Mobile App:"
echo "     ${GREEN}npm start${NC}"
echo ""
echo "  3. Update API URL in ${BLUE}lib/api/client.ts${NC}:"
echo "     ${YELLOW}http://$LOCAL_IP:3000/api${NC}"
echo ""
echo "  4. Stop all services:"
echo "     ${GREEN}./scripts/stop-dev.sh${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
