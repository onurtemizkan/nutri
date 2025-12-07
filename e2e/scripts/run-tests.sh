#!/bin/bash

# Nutri E2E Test Runner
# This script runs Maestro E2E tests with various options

set -e

# Default values
PARALLEL=false
SHARDS=4
TEST_DIR="tests"
REPORT_DIR="reports"
ENV_FILE="config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -s|--shards)
            SHARDS="$2"
            shift 2
            ;;
        -t|--test)
            TEST_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Nutri E2E Test Runner"
            echo ""
            echo "Usage: ./run-tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  -p, --parallel     Run tests in parallel using sharding"
            echo "  -s, --shards N     Number of shards for parallel execution (default: 4)"
            echo "  -t, --test PATH    Run specific test file or directory"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run-tests.sh                      # Run all tests sequentially"
            echo "  ./run-tests.sh -p                   # Run all tests in parallel (4 shards)"
            echo "  ./run-tests.sh -p -s 8              # Run in parallel with 8 shards"
            echo "  ./run-tests.sh -t auth              # Run only auth tests"
            echo "  ./run-tests.sh -t tests/auth/sign_in.yaml  # Run specific test"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to e2e directory
cd "$(dirname "$0")/.."

# Create reports directory if it doesn't exist
mkdir -p "$REPORT_DIR"

# Check if Maestro is installed
if ! command -v maestro &> /dev/null; then
    echo -e "${RED}Error: Maestro is not installed.${NC}"
    echo "Install it with: curl -Ls \"https://get.maestro.mobile.dev\" | bash"
    exit 1
fi

echo -e "${GREEN}Starting E2E Tests${NC}"
echo "================================"

# Determine test path
if [ -n "$TEST_PATH" ]; then
    if [ -d "$TEST_DIR/$TEST_PATH" ]; then
        TESTS_TO_RUN="$TEST_DIR/$TEST_PATH"
    elif [ -f "$TEST_PATH" ]; then
        TESTS_TO_RUN="$TEST_PATH"
    elif [ -f "$TEST_DIR/$TEST_PATH" ]; then
        TESTS_TO_RUN="$TEST_DIR/$TEST_PATH"
    else
        echo -e "${RED}Error: Test path not found: $TEST_PATH${NC}"
        exit 1
    fi
else
    TESTS_TO_RUN="$TEST_DIR"
fi

echo "Test path: $TESTS_TO_RUN"
echo "Parallel: $PARALLEL"
if [ "$PARALLEL" = true ]; then
    echo "Shards: $SHARDS"
fi
echo ""

# Run tests
if [ "$PARALLEL" = true ]; then
    echo -e "${YELLOW}Running tests in parallel with $SHARDS shards...${NC}"

    # Get all test files
    TEST_FILES=$(find "$TESTS_TO_RUN" -name "*.yaml" -type f | sort)
    TOTAL_TESTS=$(echo "$TEST_FILES" | wc -l | xargs)

    echo "Found $TOTAL_TESTS test files"

    # Run with sharding
    maestro test "$TESTS_TO_RUN" \
        --format junit \
        --output "$REPORT_DIR/results.xml" \
        --shard-all "$SHARDS" \
        -e TEST_USER_EMAIL="${TEST_USER_EMAIL:-test@nutri-e2e.local}" \
        -e TEST_USER_PASSWORD="${TEST_USER_PASSWORD:-TestPass123!}" \
        -e TEST_USER_NAME="${TEST_USER_NAME:-E2E Test User}"
else
    echo -e "${YELLOW}Running tests sequentially...${NC}"

    maestro test "$TESTS_TO_RUN" \
        --format junit \
        --output "$REPORT_DIR/results.xml" \
        -e TEST_USER_EMAIL="${TEST_USER_EMAIL:-test@nutri-e2e.local}" \
        -e TEST_USER_PASSWORD="${TEST_USER_PASSWORD:-TestPass123!}" \
        -e TEST_USER_NAME="${TEST_USER_NAME:-E2E Test User}"
fi

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo ""
    echo -e "${RED}Some tests failed. Check the reports for details.${NC}"
    exit 1
fi
