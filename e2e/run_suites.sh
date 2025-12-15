#!/bin/bash
# Run all optimized test suites and measure time
# Includes proper cleanup and simulator shutdown

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track exit codes for each suite
MAIN_EXIT=0
PROFILE_EXIT=0
SCAN_EXIT=0
AUTH_EXIT=0

# Cleanup function - runs on script exit
cleanup() {
    local exit_code=$?
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"

    # Shutdown simulator
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${YELLOW}Shutting down iOS simulator...${NC}"
        xcrun simctl shutdown booted 2>/dev/null || true
        echo -e "${GREEN}✓ Simulator shut down${NC}"
    fi

    # Kill any lingering Maestro processes
    pkill -f "maestro" 2>/dev/null || true

    echo -e "${GREEN}✓ Cleanup complete${NC}"

    # Exit with proper code
    if [ $MAIN_EXIT -ne 0 ] || [ $PROFILE_EXIT -ne 0 ] || [ $SCAN_EXIT -ne 0 ] || [ $AUTH_EXIT -ne 0 ]; then
        exit 1
    fi
    exit 0
}

# Set trap to run cleanup on exit (normal or error)
trap cleanup EXIT INT TERM

# Check if simulator is running and ensure it's ready for Maestro
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! xcrun simctl list devices booted | grep -q "Booted"; then
        echo -e "${YELLOW}Starting iOS simulator...${NC}"
        # Try to boot first available iPhone simulator
        SIMULATOR_UDID=$(xcrun simctl list devices available | grep -E "iPhone (14|15|16)" | grep -v "Booted" | head -1 | grep -oE "[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}")
        if [ -n "$SIMULATOR_UDID" ]; then
            xcrun simctl boot "$SIMULATOR_UDID" 2>/dev/null || true
            echo -e "${GREEN}✓ Booted simulator: $SIMULATOR_UDID${NC}"
        else
            echo -e "${RED}✗ No available iPhone simulator found${NC}"
        fi
    else
        echo -e "${GREEN}✓ Simulator already booted${NC}"
    fi

    # Wait for simulator to be fully ready (required for Maestro detection)
    echo -e "${YELLOW}Waiting for simulator to be ready...${NC}"
    sleep 5

    # Open Simulator.app (required for Maestro to detect device)
    open -a Simulator 2>/dev/null || true
    sleep 3

    # Verify simulator is detected
    if xcrun simctl list devices booted | grep -q "Booted"; then
        echo -e "${GREEN}✓ Simulator ready${NC}"
    else
        echo -e "${RED}✗ No simulator detected, tests may fail${NC}"
    fi
fi

START=$(date +%s)
echo -e "${BLUE}=== Running Optimized Test Suites ===${NC}"
echo "Start time: $(date)"
echo ""

echo -e "${BLUE}--- main_suite (dashboard, navigation, meals, health) ---${NC}"
set +e  # Don't exit on error, we want to continue
maestro test tests/suites/main_suite.yaml 2>&1 | tail -10
MAIN_EXIT=${PIPESTATUS[0]}
set -e
MAIN_END=$(date +%s)
if [ $MAIN_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ main_suite PASSED${NC} (time: $((MAIN_END - START))s)"
else
    echo -e "${RED}✗ main_suite FAILED${NC} (time: $((MAIN_END - START))s)"
fi
echo ""

echo -e "${BLUE}--- profile_suite (profile, settings, logout) ---${NC}"
set +e
maestro test tests/suites/profile_suite.yaml 2>&1 | tail -10
PROFILE_EXIT=${PIPESTATUS[0]}
set -e
PROFILE_END=$(date +%s)
if [ $PROFILE_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ profile_suite PASSED${NC} (time: $((PROFILE_END - MAIN_END))s)"
else
    echo -e "${RED}✗ profile_suite FAILED${NC} (time: $((PROFILE_END - MAIN_END))s)"
fi
echo ""

echo -e "${BLUE}--- scan_suite (food scan, barcode scan, AR) ---${NC}"
set +e
maestro test tests/suites/scan_suite.yaml 2>&1 | tail -10
SCAN_EXIT=${PIPESTATUS[0]}
set -e
SCAN_END=$(date +%s)
if [ $SCAN_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ scan_suite PASSED${NC} (time: $((SCAN_END - PROFILE_END))s)"
else
    echo -e "${RED}✗ scan_suite FAILED${NC} (time: $((SCAN_END - PROFILE_END))s)"
fi
echo ""

echo -e "${BLUE}--- auth_suite (sign in, validation, forgot password) ---${NC}"
set +e
maestro test tests/suites/auth_suite.yaml 2>&1 | tail -10
AUTH_EXIT=${PIPESTATUS[0]}
set -e
AUTH_END=$(date +%s)
if [ $AUTH_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ auth_suite PASSED${NC} (time: $((AUTH_END - SCAN_END))s)"
else
    echo -e "${RED}✗ auth_suite FAILED${NC} (time: $((AUTH_END - SCAN_END))s)"
fi
echo ""

END=$(date +%s)
TOTAL=$((END - START))

echo -e "${BLUE}=== Summary ===${NC}"
echo "Total time: ${TOTAL}s (~$((TOTAL / 60))m $((TOTAL % 60))s)"
echo "End time: $(date)"
echo ""

# Count passed/failed
PASSED=0
FAILED=0
if [ $MAIN_EXIT -eq 0 ]; then PASSED=$((PASSED + 1)); else FAILED=$((FAILED + 1)); fi
if [ $PROFILE_EXIT -eq 0 ]; then PASSED=$((PASSED + 1)); else FAILED=$((FAILED + 1)); fi
if [ $SCAN_EXIT -eq 0 ]; then PASSED=$((PASSED + 1)); else FAILED=$((FAILED + 1)); fi
if [ $AUTH_EXIT -eq 0 ]; then PASSED=$((PASSED + 1)); else FAILED=$((FAILED + 1)); fi

echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              All test suites passed!                         ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║              Some test suites failed                         ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
fi

# Cleanup will run via trap
