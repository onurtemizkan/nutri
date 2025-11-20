#!/bin/bash

# Maestro Headless Test Runner
# This script runs Maestro tests in headless mode for CI/CD environments
#
# Usage: ./scripts/run-maestro-headless.sh [test-target]
#
# Available test targets:
#   smoke      - Quick smoke tests (default)
#   all        - Run all tests
#   auth       - Authentication tests
#   meals      - Meal management tests
#   health     - Health metrics tests
#   activity   - Activity tracking tests
#   navigation - Navigation and routing tests
#   profile    - Profile and goals tests
#   validation - Input validation and error handling tests
#   <custom>   - Custom test file or directory path
#
# Examples:
#   ./scripts/run-maestro-headless.sh smoke
#   ./scripts/run-maestro-headless.sh auth
#   ./scripts/run-maestro-headless.sh .maestro/flows/01-auth/signin.yaml

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

# Workaround: Temporarily move docker-compose.yml to avoid Maestro parsing it
DOCKER_COMPOSE_MOVED=false
if [ -f "docker-compose.yml" ]; then
    mv docker-compose.yml docker-compose.yml.tmp
    DOCKER_COMPOSE_MOVED=true
    echo -e "${YELLOW}Note: Temporarily moved docker-compose.yml to avoid conflicts${NC}"
fi

# Cleanup function to restore docker-compose.yml
cleanup() {
    if [ "$DOCKER_COMPOSE_MOVED" = true ]; then
        mv docker-compose.yml.tmp docker-compose.yml
        echo -e "${YELLOW}Restored docker-compose.yml${NC}"
    fi
}

# Set trap to ensure cleanup happens on exit
trap cleanup EXIT

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

# New user for registration tests
NEW_USER_EMAIL=${NEW_USER_EMAIL:-"newuser$(date +%s)@example.com"}
NEW_USER_PASSWORD=${NEW_USER_PASSWORD:-"NewUser123"}
NEW_USER_NAME=${NEW_USER_NAME:-"New Test User"}

# Test meal data
TEST_MEAL_NAME=${TEST_MEAL_NAME:-"Grilled Chicken Salad"}
TEST_MEAL_CALORIES=${TEST_MEAL_CALORIES:-"450"}
TEST_MEAL_PROTEIN=${TEST_MEAL_PROTEIN:-"35"}
TEST_MEAL_CARBS=${TEST_MEAL_CARBS:-"40"}
TEST_MEAL_FAT=${TEST_MEAL_FAT:-"15"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Configuration:${NC}"
echo "  APP_ID: $APP_ID"
echo "  PLATFORM: $PLATFORM"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  FORMAT: $FORMAT"
echo ""

# Function to run maestro with all environment variables
run_maestro_test() {
    local output_file=$1
    local test_path=$2

    maestro test \
        --env APP_ID="$APP_ID" \
        --env TEST_EMAIL="$TEST_EMAIL" \
        --env TEST_PASSWORD="$TEST_PASSWORD" \
        --env TEST_NAME="$TEST_NAME" \
        --env NEW_USER_EMAIL="$NEW_USER_EMAIL" \
        --env NEW_USER_PASSWORD="$NEW_USER_PASSWORD" \
        --env NEW_USER_NAME="$NEW_USER_NAME" \
        --env TEST_MEAL_NAME="$TEST_MEAL_NAME" \
        --env TEST_MEAL_CALORIES="$TEST_MEAL_CALORIES" \
        --env TEST_MEAL_PROTEIN="$TEST_MEAL_PROTEIN" \
        --env TEST_MEAL_CARBS="$TEST_MEAL_CARBS" \
        --env TEST_MEAL_FAT="$TEST_MEAL_FAT" \
        --format="$FORMAT" \
        --output="$OUTPUT_DIR/$output_file" \
        "$test_path"
}

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
        echo -e "${GREEN}Running smoke tests sequentially...${NC}"
        echo ""

        # Run each smoke test individually in sequence
        SMOKE_TESTS=(
            "01-login-navigation.yaml"
            "02-add-meal.yaml"
            "03-profile-logout.yaml"
            "04-registration.yaml"
        )

        FAILED_TESTS=()
        PASSED_TESTS=()

        for test in "${SMOKE_TESTS[@]}"; do
            test_name=$(basename "$test" .yaml)
            echo -e "${YELLOW}→ Running: $test_name${NC}"

            if run_maestro_test "smoke-$test_name.xml" ".maestro/flows/smoke/$test"; then
                echo -e "${GREEN}✓ Passed: $test_name${NC}"
                PASSED_TESTS+=("$test_name")
            else
                echo -e "${RED}✗ Failed: $test_name${NC}"
                FAILED_TESTS+=("$test_name")
                # Continue with other tests even if one fails
            fi
            echo ""
        done

        # Print summary
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Smoke Tests Summary${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo "Passed: ${#PASSED_TESTS[@]}/${#SMOKE_TESTS[@]}"
        echo "Failed: ${#FAILED_TESTS[@]}/${#SMOKE_TESTS[@]}"

        if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
            echo ""
            echo -e "${RED}Failed tests:${NC}"
            for test in "${FAILED_TESTS[@]}"; do
                echo "  - $test"
            done
            exit 1
        fi
        ;;

    all)
        echo -e "${GREEN}Running all tests...${NC}"
        run_maestro_test "all-tests.xml" "$FLOWS_DIR"
        ;;

    auth)
        echo -e "${GREEN}Running authentication tests...${NC}"
        run_maestro_test "auth-tests.xml" ".maestro/flows/01-auth/"
        ;;

    meals)
        echo -e "${GREEN}Running meal tests...${NC}"
        run_maestro_test "meal-tests.xml" ".maestro/flows/02-meals/"
        ;;

    health)
        echo -e "${GREEN}Running health tests...${NC}"
        run_maestro_test "health-tests.xml" ".maestro/flows/03-health/"
        ;;

    activity)
        echo -e "${GREEN}Running activity tests...${NC}"
        run_maestro_test "activity-tests.xml" ".maestro/flows/04-activity/"
        ;;

    navigation)
        echo -e "${GREEN}Running navigation tests...${NC}"
        run_maestro_test "navigation-tests.xml" ".maestro/flows/04-navigation/"
        ;;

    profile)
        echo -e "${GREEN}Running profile tests...${NC}"
        # Run profile tests from both 03-profile (goals) and 05-profile (update)
        echo -e "${YELLOW}  Testing profile goals...${NC}"
        run_maestro_test "profile-goals-tests.xml" ".maestro/flows/03-profile/"
        if [ $? -eq 0 ]; then
            echo -e "${YELLOW}  Testing profile updates...${NC}"
            run_maestro_test "profile-update-tests.xml" ".maestro/flows/05-profile/"
        fi
        ;;

    validation)
        echo -e "${GREEN}Running validation tests...${NC}"
        run_maestro_test "validation-tests.xml" ".maestro/flows/06-validation/"
        ;;

    *)
        echo -e "${YELLOW}Running custom test: $TEST_TARGET${NC}"
        run_maestro_test "custom-tests.xml" "$TEST_TARGET"
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
