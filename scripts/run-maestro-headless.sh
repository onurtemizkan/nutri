#!/bin/bash

# Maestro Headless Test Runner
# This script runs Maestro tests in headless mode for CI/CD environments

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Maestro Headless Test Runner${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Maestro is installed
if ! command -v maestro &> /dev/null; then
    echo -e "${RED}Error: Maestro is not installed${NC}"
    echo "Install it with: curl -Ls 'https://get.maestro.mobile.dev' | bash"
    exit 1
fi

# Configuration
APP_ID=${APP_ID:-"com.anonymous.nutri"}
PLATFORM=${PLATFORM:-"ios"}
OUTPUT_DIR=${OUTPUT_DIR:-"maestro-results"}
FORMAT=${FORMAT:-"junit"}  # junit, json, or pretty
FLOWS_DIR=${FLOWS_DIR:-".maestro/flows"}

# Test user credentials (can be overridden via environment variables)
TEST_EMAIL=${TEST_EMAIL:-"testuser@example.com"}
TEST_PASSWORD=${TEST_PASSWORD:-"Test123456"}
TEST_NAME=${TEST_NAME:-"Test User"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Configuration:${NC}"
echo "  APP_ID: $APP_ID"
echo "  PLATFORM: $PLATFORM"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  FORMAT: $FORMAT"
echo ""

# Check if simulator/emulator is running
if [ "$PLATFORM" = "ios" ]; then
    if ! xcrun simctl list devices | grep -q "Booted"; then
        echo -e "${YELLOW}No iOS simulator is booted. Booting default simulator...${NC}"
        # Boot the first available simulator
        SIMULATOR_ID=$(xcrun simctl list devices available | grep "iPhone" | head -1 | grep -oE '[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}')
        if [ -z "$SIMULATOR_ID" ]; then
            echo -e "${RED}No iOS simulators found${NC}"
            exit 1
        fi
        xcrun simctl boot "$SIMULATOR_ID"
        sleep 5
    fi
fi

# Run tests based on provided argument
TEST_TARGET=${1:-"smoke"}

case $TEST_TARGET in
    smoke)
        echo -e "${GREEN}Running smoke tests...${NC}"
        maestro test \
            --env APP_ID="$APP_ID" \
            --env TEST_EMAIL="$TEST_EMAIL" \
            --env TEST_PASSWORD="$TEST_PASSWORD" \
            --env TEST_NAME="$TEST_NAME" \
            --format="$FORMAT" \
            --output="$OUTPUT_DIR/smoke-tests.xml" \
            .maestro/flows/suites/smoke-tests.yaml
        ;;

    all)
        echo -e "${GREEN}Running all tests...${NC}"
        maestro test \
            --env APP_ID="$APP_ID" \
            --env TEST_EMAIL="$TEST_EMAIL" \
            --env TEST_PASSWORD="$TEST_PASSWORD" \
            --env TEST_NAME="$TEST_NAME" \
            --format="$FORMAT" \
            --output="$OUTPUT_DIR/all-tests.xml" \
            "$FLOWS_DIR"
        ;;

    auth)
        echo -e "${GREEN}Running authentication tests...${NC}"
        maestro test \
            --env APP_ID="$APP_ID" \
            --env TEST_EMAIL="$TEST_EMAIL" \
            --env TEST_PASSWORD="$TEST_PASSWORD" \
            --env TEST_NAME="$TEST_NAME" \
            --format="$FORMAT" \
            --output="$OUTPUT_DIR/auth-tests.xml" \
            .maestro/flows/01-auth/
        ;;

    meals)
        echo -e "${GREEN}Running meal tests...${NC}"
        maestro test \
            --env APP_ID="$APP_ID" \
            --env TEST_EMAIL="$TEST_EMAIL" \
            --env TEST_PASSWORD="$TEST_PASSWORD" \
            --env TEST_NAME="$TEST_NAME" \
            --format="$FORMAT" \
            --output="$OUTPUT_DIR/meal-tests.xml" \
            .maestro/flows/02-meals/
        ;;

    health)
        echo -e "${GREEN}Running health tests...${NC}"
        maestro test \
            --env APP_ID="$APP_ID" \
            --env TEST_EMAIL="$TEST_EMAIL" \
            --env TEST_PASSWORD="$TEST_PASSWORD" \
            --env TEST_NAME="$TEST_NAME" \
            --format="$FORMAT" \
            --output="$OUTPUT_DIR/health-tests.xml" \
            .maestro/flows/03-health/
        ;;

    activity)
        echo -e "${GREEN}Running activity tests...${NC}"
        maestro test \
            --env APP_ID="$APP_ID" \
            --env TEST_EMAIL="$TEST_EMAIL" \
            --env TEST_PASSWORD="$TEST_PASSWORD" \
            --env TEST_NAME="$TEST_NAME" \
            --format="$FORMAT" \
            --output="$OUTPUT_DIR/activity-tests.xml" \
            .maestro/flows/04-activity/
        ;;

    profile)
        echo -e "${GREEN}Running profile tests...${NC}"
        maestro test \
            --env APP_ID="$APP_ID" \
            --env TEST_EMAIL="$TEST_EMAIL" \
            --env TEST_PASSWORD="$TEST_PASSWORD" \
            --env TEST_NAME="$TEST_NAME" \
            --format="$FORMAT" \
            --output="$OUTPUT_DIR/profile-tests.xml" \
            .maestro/flows/05-profile/
        ;;

    validation)
        echo -e "${GREEN}Running validation tests...${NC}"
        maestro test \
            --env APP_ID="$APP_ID" \
            --env TEST_EMAIL="$TEST_EMAIL" \
            --env TEST_PASSWORD="$TEST_PASSWORD" \
            --env TEST_NAME="$TEST_NAME" \
            --format="$FORMAT" \
            --output="$OUTPUT_DIR/validation-tests.xml" \
            .maestro/flows/06-validation/
        ;;

    *)
        echo -e "${YELLOW}Running custom test: $TEST_TARGET${NC}"
        maestro test \
            --env APP_ID="$APP_ID" \
            --env TEST_EMAIL="$TEST_EMAIL" \
            --env TEST_PASSWORD="$TEST_PASSWORD" \
            --env TEST_NAME="$TEST_NAME" \
            --format="$FORMAT" \
            --output="$OUTPUT_DIR/custom-tests.xml" \
            "$TEST_TARGET"
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Results saved to: $OUTPUT_DIR"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Tests failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo "Check results in: $OUTPUT_DIR"
    exit 1
fi
