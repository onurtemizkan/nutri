#!/bin/bash

# E2E Test Runner Script for Nutri App
# =====================================
# This script automates the E2E testing workflow.
#
# Usage:
#   ./e2e/scripts/run-e2e-tests.sh          # Run all tests
#   ./e2e/scripts/run-e2e-tests.sh auth     # Run auth tests only
#   ./e2e/scripts/run-e2e-tests.sh meals    # Run meal tests only
#   ./e2e/scripts/run-e2e-tests.sh health   # Run health tests only
#   ./e2e/scripts/run-e2e-tests.sh profile  # Run profile tests only
#   ./e2e/scripts/run-e2e-tests.sh --seed   # Seed database before running
#   ./e2e/scripts/run-e2e-tests.sh --parallel # Run tests in parallel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
E2E_DIR="$PROJECT_ROOT/e2e"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SEED_DATABASE=false
RUN_PARALLEL=false
TEST_SUITE="all"
SHUTDOWN_SIMULATOR=true
SIMULATOR_WAS_BOOTED=false

# Cleanup function - runs on script exit
cleanup() {
    local exit_code=$?
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"

    # Shutdown simulator if we started it or if shutdown is requested
    if [ "$SHUTDOWN_SIMULATOR" = true ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo -e "${YELLOW}Shutting down iOS simulator...${NC}"
            xcrun simctl shutdown booted 2>/dev/null || true
            echo -e "${GREEN}✓ Simulator shut down${NC}"
        fi
    fi

    # Kill any lingering Maestro processes
    pkill -f "maestro" 2>/dev/null || true

    echo -e "${GREEN}✓ Cleanup complete${NC}"
    exit $exit_code
}

# Set trap to run cleanup on exit (normal or error)
trap cleanup EXIT INT TERM

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED_DATABASE=true
            shift
            ;;
        --parallel)
            RUN_PARALLEL=true
            shift
            ;;
        --no-shutdown)
            SHUTDOWN_SIMULATOR=false
            shift
            ;;
        auth|meals|health|profile|scan|supplements|common|home)
            TEST_SUITE=$1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [TEST_SUITE]"
            echo ""
            echo "Options:"
            echo "  --seed         Seed the database with test data before running"
            echo "  --parallel     Run tests in parallel (faster but less reliable)"
            echo "  --no-shutdown  Don't shutdown simulator after tests"
            echo "  --help         Show this help message"
            echo ""
            echo "Test Suites:"
            echo "  all          Run all tests (default)"
            echo "  auth         Run authentication tests"
            echo "  meals        Run meal management tests"
            echo "  health       Run health metrics tests"
            echo "  profile      Run profile tests"
            echo "  scan         Run barcode scanner tests"
            echo "  supplements  Run supplements tests"
            echo "  common       Run common/navigation tests"
            echo "  home         Run home/dashboard tests"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Nutri E2E Test Runner                                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if Maestro is installed
if ! command -v maestro &> /dev/null; then
    echo -e "${RED}Error: Maestro is not installed.${NC}"
    echo "Install with: curl -Ls 'https://get.maestro.mobile.dev' | bash"
    exit 1
fi
echo -e "${GREEN}✓ Maestro is installed${NC}"

# Check if simulator is running (for iOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if xcrun simctl list devices booted | grep -q "Booted"; then
        SIMULATOR_WAS_BOOTED=true
        echo -e "${GREEN}✓ iOS simulator is already running${NC}"
    else
        echo -e "${YELLOW}Warning: No iOS simulator is booted.${NC}"
        echo "Starting a simulator..."
        xcrun simctl boot "iPhone 15 Pro" 2>/dev/null || true
        # Open Simulator.app to show it
        open -a Simulator 2>/dev/null || true
        sleep 3  # Give simulator time to fully boot
        echo -e "${GREEN}✓ iOS simulator started${NC}"
    fi
fi

# Seed database if requested
if [ "$SEED_DATABASE" = true ]; then
    echo ""
    echo -e "${YELLOW}Seeding database with test data...${NC}"
    cd "$PROJECT_ROOT/server"
    npm run db:seed
    echo -e "${GREEN}✓ Database seeded${NC}"
fi

# Determine test path
case $TEST_SUITE in
    all)
        TEST_PATH="$E2E_DIR/tests/"
        ;;
    *)
        TEST_PATH="$E2E_DIR/tests/$TEST_SUITE/"
        ;;
esac

# Run tests
echo ""
echo -e "${YELLOW}Running E2E tests: ${TEST_SUITE}${NC}"
echo -e "${BLUE}Test path: ${TEST_PATH}${NC}"
echo ""

cd "$E2E_DIR"

if [ "$RUN_PARALLEL" = true ]; then
    echo -e "${YELLOW}Running in parallel mode...${NC}"
    maestro test --parallel "$TEST_PATH"
else
    maestro test "$TEST_PATH"
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         All E2E tests passed!                                ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║         Some E2E tests failed                                ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
fi

# Exit code will trigger cleanup via trap
exit $EXIT_CODE
