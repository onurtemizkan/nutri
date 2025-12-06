#!/bin/bash

# E2E Test Runner for Nutri App
# This script sets up all required services and runs E2E tests locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
E2E_DIR="$PROJECT_ROOT/e2e"
SERVER_DIR="$PROJECT_ROOT/server"
METRO_PORT=8081
BACKEND_PORT=3000

# Test environment variables
export TEST_USER_EMAIL="${TEST_USER_EMAIL:-test@nutri-e2e.local}"
export TEST_USER_PASSWORD="${TEST_USER_PASSWORD:-TestPass123!}"
export TEST_USER_NAME="${TEST_USER_NAME:-E2E Test User}"

# PIDs for cleanup
METRO_PID=""
BACKEND_PID=""

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"

    if [ -n "$METRO_PID" ] && kill -0 "$METRO_PID" 2>/dev/null; then
        echo "Stopping Metro bundler (PID: $METRO_PID)..."
        kill "$METRO_PID" 2>/dev/null || true
    fi

    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "Stopping backend server (PID: $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null || true
    fi

    # Kill any remaining processes on our ports
    lsof -ti:$METRO_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true

    echo -e "${GREEN}Cleanup complete.${NC}"
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# Print header
print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║           Nutri E2E Test Runner                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    # Check Maestro
    if ! command -v maestro &> /dev/null; then
        echo -e "${RED}Error: Maestro CLI not installed.${NC}"
        echo "Install with: curl -Ls \"https://get.maestro.mobile.dev\" | bash"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Maestro CLI installed"

    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Error: Node.js not installed.${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Node.js installed"

    # Check if iOS simulator is running (macOS only)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! xcrun simctl list devices | grep -q "Booted"; then
            echo -e "${YELLOW}Warning: No iOS simulator running.${NC}"
            echo "Starting iPhone 16 Pro simulator..."
            xcrun simctl boot "iPhone 16 Pro" 2>/dev/null || true
            open -a Simulator
            sleep 5
        fi
        echo -e "  ${GREEN}✓${NC} iOS simulator running"
    fi

    # Check if server dependencies are installed
    if [ ! -d "$SERVER_DIR/node_modules" ]; then
        echo -e "${YELLOW}Installing server dependencies...${NC}"
        (cd "$SERVER_DIR" && npm install)
    fi
    echo -e "  ${GREEN}✓${NC} Server dependencies installed"

    # Check if app dependencies are installed
    if [ ! -d "$PROJECT_ROOT/node_modules" ]; then
        echo -e "${YELLOW}Installing app dependencies...${NC}"
        (cd "$PROJECT_ROOT" && npm install)
    fi
    echo -e "  ${GREEN}✓${NC} App dependencies installed"

    echo ""
}

# Wait for a service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-60}
    local attempt=0

    echo -n "Waiting for $name to be ready"
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo -e " ${RED}✗${NC}"
    echo -e "${RED}Error: $name failed to start after $max_attempts seconds${NC}"
    return 1
}

# Start backend server
start_backend() {
    echo -e "${BLUE}Starting backend server...${NC}"

    # Check if already running
    if curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Backend already running on port $BACKEND_PORT"
        return 0
    fi

    # Start backend in background
    (cd "$SERVER_DIR" && npm run dev > /tmp/nutri-backend.log 2>&1) &
    BACKEND_PID=$!

    # Wait for backend to be ready
    wait_for_service "http://localhost:$BACKEND_PORT/health" "Backend API" 30
}

# Start Metro bundler
start_metro() {
    echo -e "${BLUE}Starting Metro bundler...${NC}"

    # Check if already running
    if curl -s "http://localhost:$METRO_PORT/status" 2>/dev/null | grep -q "packager-status:running"; then
        echo -e "  ${GREEN}✓${NC} Metro already running on port $METRO_PORT"
        return 0
    fi

    # Kill any existing Metro process
    lsof -ti:$METRO_PORT | xargs kill -9 2>/dev/null || true
    sleep 1

    # Start Metro in background
    (cd "$PROJECT_ROOT" && npx expo start --port $METRO_PORT > /tmp/nutri-metro.log 2>&1) &
    METRO_PID=$!

    # Wait for Metro to be ready
    wait_for_service "http://localhost:$METRO_PORT/status" "Metro bundler" 60
}

# Run E2E tests
run_tests() {
    local test_path="${1:-tests/}"
    local extra_args="${@:2}"

    echo -e "${BLUE}Running E2E tests...${NC}"
    echo -e "  Test path: $test_path"
    echo -e "  Test user: $TEST_USER_EMAIL"
    echo ""

    cd "$E2E_DIR"

    maestro test "$test_path" \
        -e TEST_USER_EMAIL="$TEST_USER_EMAIL" \
        -e TEST_USER_PASSWORD="$TEST_USER_PASSWORD" \
        -e TEST_USER_NAME="$TEST_USER_NAME" \
        $extra_args

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}All tests passed!${NC}"
    else
        echo -e "\n${RED}Some tests failed.${NC}"
    fi

    return $exit_code
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS] [TEST_PATH]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -s, --skip-backend  Skip starting the backend server"
    echo "  -m, --skip-metro    Skip starting Metro bundler"
    echo "  -c, --clean         Clean up existing processes before starting"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all tests"
    echo "  $0 tests/auth/sign_in.yaml   # Run specific test"
    echo "  $0 -s tests/auth/            # Run auth tests (skip backend if already running)"
    echo ""
}

# Main
main() {
    local skip_backend=false
    local skip_metro=false
    local clean=false
    local test_path="tests/"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -s|--skip-backend)
                skip_backend=true
                shift
                ;;
            -m|--skip-metro)
                skip_metro=true
                shift
                ;;
            -c|--clean)
                clean=true
                shift
                ;;
            *)
                test_path="$1"
                shift
                ;;
        esac
    done

    print_header

    # Clean if requested
    if [ "$clean" = true ]; then
        echo -e "${YELLOW}Cleaning up existing processes...${NC}"
        lsof -ti:$METRO_PORT | xargs kill -9 2>/dev/null || true
        lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    check_prerequisites

    # Start services
    if [ "$skip_backend" = false ]; then
        start_backend
    fi

    if [ "$skip_metro" = false ]; then
        start_metro
    fi

    echo ""

    # Run tests
    run_tests "$test_path"
}

main "$@"
