#!/bin/bash

# Nutri Full Development Stack + iOS Simulator Launcher
# Starts ALL services and launches the mobile app in the iOS Simulator

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SIMULATOR_DEVICE="iPhone 17 Pro"

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

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

print_header "🚀 Starting Nutri Full Development Stack"
echo ""
echo "  📱 Target Simulator: ${YELLOW}$SIMULATOR_DEVICE${NC}"
echo ""

# ============================================================================
# Step 1: Ensure logs directory exists
# ============================================================================
mkdir -p "$PROJECT_ROOT/logs"

# ============================================================================
# Step 2: Start all backend services (Docker + API)
# ============================================================================
print_header "🐳 Starting Backend Services"

print_info "Running start-all.sh to start Docker and Backend API..."
./scripts/start-all.sh

# ============================================================================
# Step 3: Boot the iOS Simulator
# ============================================================================
print_header "📱 Launching iOS Simulator"

# Get simulator UDID
SIMULATOR_UDID=$(xcrun simctl list devices available | grep "$SIMULATOR_DEVICE" | head -1 | grep -oE '[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}')

if [ -z "$SIMULATOR_UDID" ]; then
    print_error "Could not find simulator: $SIMULATOR_DEVICE"
    print_info "Available simulators:"
    xcrun simctl list devices available | grep -i "iphone"
    exit 1
fi

print_info "Found $SIMULATOR_DEVICE (UDID: $SIMULATOR_UDID)"

# Check if simulator is already booted
SIMULATOR_STATE=$(xcrun simctl list devices | grep "$SIMULATOR_UDID" | grep -oE '\(Booted\)|\(Shutdown\)')

if [[ "$SIMULATOR_STATE" == "(Shutdown)" ]]; then
    print_info "Booting $SIMULATOR_DEVICE..."
    xcrun simctl boot "$SIMULATOR_UDID"
    sleep 2
fi

# Open Simulator app
print_info "Opening Simulator app..."
open -a Simulator

# Wait for simulator to be ready
print_info "Waiting for simulator to be ready..."
sleep 3

print_success "$SIMULATOR_DEVICE is running!"

# ============================================================================
# Step 4: Build and Run iOS App (Development Build)
# ============================================================================
print_header "📲 Building & Running iOS App (Development Build)"

print_info "This uses native modules (HealthKit, etc.) - requires Xcode"
print_info "Building for $SIMULATOR_DEVICE..."
echo ""

# Run expo run:ios which builds and runs a development build with native modules
# The --device flag specifies the simulator to use
npx expo run:ios --device "$SIMULATOR_UDID"
