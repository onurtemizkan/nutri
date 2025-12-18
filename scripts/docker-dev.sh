#!/bin/bash

# =============================================================================
# Nutri Docker Development Environment Manager
#
# Usage:
#   ./scripts/docker-dev.sh [command]
#
# Commands:
#   start   - Start all development services
#   stop    - Stop all development services
#   restart - Restart all services
#   build   - Rebuild and start services
#   logs    - View logs from all services
#   status  - Show status of all services
#   clean   - Stop and remove all containers and volumes
#   migrate - Run database migrations
#   shell   - Open shell in a service (backend/ml)
#   help    - Show this help message
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Compose file
COMPOSE_FILE="docker-compose.dev.yml"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_info() {
    echo -e "${CYAN}ℹ${NC}  $1"
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

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
}

# Wait for a service to be healthy
wait_for_health() {
    local service=$1
    local max_wait=${2:-60}
    local interval=2
    local elapsed=0

    print_info "Waiting for $service to be healthy..."

    while [ $elapsed -lt $max_wait ]; do
        if docker compose -f "$COMPOSE_FILE" ps "$service" 2>/dev/null | grep -q "healthy"; then
            print_success "$service is healthy!"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        echo -n "."
    done

    echo ""
    print_warning "$service did not become healthy within ${max_wait}s"
    return 1
}

# Start services
cmd_start() {
    print_header "🚀 Starting Nutri Development Environment"

    check_docker

    print_info "Starting Docker services..."
    docker compose -f "$COMPOSE_FILE" up -d

    # Wait for core services
    wait_for_health "postgres" 30
    wait_for_health "redis" 15

    # Run migrations if backend started
    if docker compose -f "$COMPOSE_FILE" ps backend 2>/dev/null | grep -q "Up"; then
        print_info "Running database migrations..."
        sleep 5  # Give backend time to initialize
        docker compose -f "$COMPOSE_FILE" exec -T backend npx prisma migrate deploy 2>/dev/null || \
        docker compose -f "$COMPOSE_FILE" exec -T backend npx prisma db push --accept-data-loss 2>/dev/null || \
        print_warning "Migration skipped - run manually with: npm run docker:dev -- migrate"
    fi

    echo ""
    print_success "Development environment is ready!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📌 Available Services:"
    echo ""
    echo "  ${GREEN}Backend API${NC}:        http://localhost:3000"
    echo "  ${GREEN}ML Service${NC}:         http://localhost:8000"
    echo "  ${GREEN}Adminer (DB UI)${NC}:    http://localhost:8080"
    echo "  ${GREEN}Redis Commander${NC}:    http://localhost:8081"
    echo "  ${GREEN}Prisma Studio${NC}:      http://localhost:5555"
    echo ""
    echo "📌 Useful Commands:"
    echo ""
    echo "  View logs:          ${CYAN}npm run docker:dev:logs${NC}"
    echo "  Stop services:      ${CYAN}npm run docker:dev:stop${NC}"
    echo "  Backend shell:      ${CYAN}npm run docker:dev:shell:backend${NC}"
    echo "  ML Service shell:   ${CYAN}npm run docker:dev:shell:ml${NC}"
    echo ""
    echo "📌 Database Connection:"
    echo ""
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  User: postgres"
    echo "  Password: postgres"
    echo "  Database: nutri_db"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Stop services
cmd_stop() {
    print_header "🛑 Stopping Nutri Development Environment"

    check_docker

    print_info "Stopping Docker services..."
    docker compose -f "$COMPOSE_FILE" down

    print_success "All services stopped"
}

# Restart services
cmd_restart() {
    print_header "🔄 Restarting Nutri Development Environment"

    cmd_stop
    cmd_start
}

# Build and start services
cmd_build() {
    print_header "🔨 Building and Starting Services"

    check_docker

    print_info "Building Docker images..."
    docker compose -f "$COMPOSE_FILE" build

    print_info "Starting services..."
    docker compose -f "$COMPOSE_FILE" up -d

    cmd_start
}

# View logs
cmd_logs() {
    local service=${1:-""}

    check_docker

    if [ -n "$service" ]; then
        print_info "Viewing logs for $service..."
        docker compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        print_info "Viewing logs for all services (Ctrl+C to exit)..."
        docker compose -f "$COMPOSE_FILE" logs -f
    fi
}

# Show status
cmd_status() {
    print_header "📊 Nutri Development Environment Status"

    check_docker

    echo "Container Status:"
    echo ""
    docker compose -f "$COMPOSE_FILE" ps

    echo ""
    echo "Resource Usage:"
    echo ""
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
        $(docker compose -f "$COMPOSE_FILE" ps -q 2>/dev/null) 2>/dev/null || \
        print_warning "No running containers"
}

# Clean up everything
cmd_clean() {
    print_header "🧹 Cleaning Up Development Environment"

    check_docker

    print_warning "This will stop and remove all containers, networks, and volumes!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Stopping services..."
        docker compose -f "$COMPOSE_FILE" down -v --remove-orphans

        print_info "Removing volumes..."
        docker volume rm nutri_dev_postgres_data nutri_dev_redis_data \
            nutri_dev_backend_node_modules nutri_dev_backend_prisma_client 2>/dev/null || true

        print_success "Cleanup complete!"
    else
        print_info "Cleanup cancelled"
    fi
}

# Run migrations
cmd_migrate() {
    print_header "📦 Running Database Migrations"

    check_docker

    print_info "Running Prisma migrations..."
    docker compose -f "$COMPOSE_FILE" exec backend npx prisma migrate deploy || \
    docker compose -f "$COMPOSE_FILE" exec backend npx prisma db push --accept-data-loss

    print_success "Migrations complete!"
}

# Open shell in service
cmd_shell() {
    local service=${1:-"backend"}

    check_docker

    case $service in
        backend)
            print_info "Opening shell in backend service..."
            docker compose -f "$COMPOSE_FILE" exec backend sh
            ;;
        ml|ml-service)
            print_info "Opening shell in ML service..."
            docker compose -f "$COMPOSE_FILE" exec ml-service bash
            ;;
        postgres|db)
            print_info "Opening psql in postgres..."
            docker compose -f "$COMPOSE_FILE" exec postgres psql -U postgres nutri_db
            ;;
        redis)
            print_info "Opening redis-cli..."
            docker compose -f "$COMPOSE_FILE" exec redis redis-cli
            ;;
        *)
            print_error "Unknown service: $service"
            print_info "Available services: backend, ml, postgres, redis"
            exit 1
            ;;
    esac
}

# Show help
cmd_help() {
    echo ""
    echo -e "${BLUE}Nutri Docker Development Environment${NC}"
    echo ""
    echo "Usage: ./scripts/docker-dev.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start              Start all development services"
    echo "  stop               Stop all development services"
    echo "  restart            Restart all services"
    echo "  build              Rebuild and start services"
    echo "  logs [service]     View logs (optionally for specific service)"
    echo "  status             Show status of all services"
    echo "  clean              Stop and remove all containers and volumes"
    echo "  migrate            Run database migrations"
    echo "  shell [service]    Open shell in service (backend/ml/postgres/redis)"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/docker-dev.sh start"
    echo "  ./scripts/docker-dev.sh logs backend"
    echo "  ./scripts/docker-dev.sh shell ml"
    echo ""
    echo "NPM shortcuts:"
    echo "  npm run docker:dev:start"
    echo "  npm run docker:dev:stop"
    echo "  npm run docker:dev:logs"
    echo "  npm run docker:dev:build"
    echo "  npm run docker:dev:clean"
    echo "  npm run docker:dev:status"
    echo "  npm run docker:dev:shell:backend"
    echo "  npm run docker:dev:shell:ml"
    echo ""
}

# Main command router
case "${1:-help}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    build)
        cmd_build
        ;;
    logs)
        cmd_logs "$2"
        ;;
    status)
        cmd_status
        ;;
    clean)
        cmd_clean
        ;;
    migrate)
        cmd_migrate
        ;;
    shell)
        cmd_shell "$2"
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        print_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
