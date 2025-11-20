#!/bin/bash

# Maestro Test Runner Script
# Convenient wrapper for running different test suites

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✅ ${NC}$1"
}

print_error() {
    echo -e "${RED}❌ ${NC}$1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  ${NC}$1"
}

# Navigate to project root
cd "$(dirname "$0")/../.."

# Check if Maestro is installed
if ! command -v maestro &> /dev/null; then
    print_error "Maestro is not installed!"
    echo "Run: .maestro/scripts/setup-maestro.sh"
    exit 1
fi

# Parse command line arguments
SUITE="smoke"
PLATFORM=""
APP_ID="com.yourcompany.nutri"
REPORT_FORMAT="console"

print_help() {
    echo "Usage: ./run-tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --suite <name>     Test suite to run (smoke, critical, regression, all)"
    echo "                         Default: smoke"
    echo "  -p, --platform <name>  Platform (ios, android)"
    echo "                         If not specified, uses connected device"
    echo "  -a, --app-id <id>      App bundle ID (default: com.yourcompany.nutri)"
    echo "  -f, --format <format>  Report format (console, junit, html)"
    echo "                         Default: console"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run-tests.sh                          # Run smoke tests"
    echo "  ./run-tests.sh -s critical              # Run critical path tests"
    echo "  ./run-tests.sh -s regression -f junit   # Run all tests with JUnit report"
    echo "  ./run-tests.sh -s all -p ios            # Run all test flows on iOS"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--suite)
            SUITE="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -a|--app-id)
            APP_ID="$2"
            shift 2
            ;;
        -f|--format)
            REPORT_FORMAT="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Determine test file(s) to run
case $SUITE in
    smoke)
        TEST_FILES=".maestro/flows/suites/smoke-tests.yaml"
        print_info "Running Smoke Tests (quick validation)"
        ;;
    critical)
        TEST_FILES=".maestro/flows/suites/critical-path.yaml"
        print_info "Running Critical Path Tests (key user journeys)"
        ;;
    regression)
        TEST_FILES=".maestro/flows/suites/regression.yaml"
        print_info "Running Full Regression Suite (all tests)"
        ;;
    all)
        TEST_FILES=".maestro/flows/**/*.yaml"
        print_info "Running All Individual Test Flows"
        ;;
    *)
        print_error "Invalid suite: $SUITE"
        print_help
        exit 1
        ;;
esac

# Build Maestro command
MAESTRO_CMD="maestro test $TEST_FILES --env APP_ID=$APP_ID"

# Add platform if specified
if [ -n "$PLATFORM" ]; then
    MAESTRO_CMD="$MAESTRO_CMD --platform $PLATFORM"
fi

# Add report format
case $REPORT_FORMAT in
    junit)
        mkdir -p test-results
        MAESTRO_CMD="$MAESTRO_CMD --format junit --output test-results/results.xml"
        print_info "JUnit report will be saved to: test-results/results.xml"
        ;;
    html)
        mkdir -p test-results
        MAESTRO_CMD="$MAESTRO_CMD --format html --output test-results/report.html"
        print_info "HTML report will be saved to: test-results/report.html"
        ;;
    console)
        # Default console output
        ;;
    *)
        print_error "Invalid format: $REPORT_FORMAT"
        exit 1
        ;;
esac

# Check if device/emulator is connected
print_info "Checking for connected devices..."
if ! maestro test --dry-run .maestro/flows/01-auth/signin.yaml &> /dev/null; then
    print_warning "No device/emulator detected!"
    echo ""
    echo "Please start a device:"
    echo "  iOS:     Open Simulator or connect device"
    echo "  Android: Start emulator or connect device"
    echo ""
    exit 1
fi

print_success "Device detected"
echo ""

# Run tests
print_info "Executing tests..."
echo "Command: $MAESTRO_CMD"
echo ""

# Execute with proper error handling
if eval $MAESTRO_CMD; then
    echo ""
    print_success "All tests passed! ✨"
    exit 0
else
    echo ""
    print_error "Tests failed!"
    echo ""
    print_info "Debug tips:"
    echo "  • Check screenshots in ~/.maestro/tests/"
    echo "  • Run with Maestro Studio for visual debugging: maestro studio"
    echo "  • Check app logs for errors"
    exit 1
fi
