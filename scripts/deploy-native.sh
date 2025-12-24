#!/bin/bash
# Deploy Native iOS Build to USB-connected iPhone
# Usage: ./scripts/deploy-native.sh [--release|--debug] [--clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IOS_DIR="$PROJECT_DIR/ios"
CONFIGURATION="Release"
CLEAN_BUILD=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() { echo -e "${BLUE}==>${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}!${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --release) CONFIGURATION="Release" ;;
        --debug) CONFIGURATION="Debug" ;;
        --clean) CLEAN_BUILD=true ;;
        --help|-h)
            echo "Deploy Native iOS Build to USB-connected iPhone"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --release  Build Release configuration (default)"
            echo "  --debug    Build Debug configuration"
            echo "  --clean    Clean build before building"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *) print_error "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Check prerequisites
print_step "Checking prerequisites..."

if ! command -v xcodebuild &> /dev/null; then
    print_error "Xcode not found. Please install Xcode from the App Store"
    exit 1
fi
print_success "Xcode available"

if ! command -v xcrun &> /dev/null; then
    print_error "xcrun not found. Please install Xcode Command Line Tools"
    exit 1
fi
print_success "Xcode CLI tools available"

# Check for iOS directory
if [ ! -d "$IOS_DIR" ]; then
    print_error "iOS directory not found. Run 'npx expo prebuild --platform ios' first"
    exit 1
fi
print_success "iOS project exists"

# Find connected iPhone
print_step "Detecting connected iPhone..."
DEVICE_INFO=$(xcrun xctrace list devices 2>/dev/null | grep -E "iPhone.*\([0-9]+\.[0-9]+\)" | grep -v Simulator | head -1)

if [ -z "$DEVICE_INFO" ]; then
    print_error "No iPhone connected via USB"
    echo ""
    echo "Please connect your iPhone and trust this computer"
    exit 1
fi

# Extract device name and UDID
DEVICE_NAME=$(echo "$DEVICE_INFO" | sed -E 's/^(.+) \([0-9]+\.[0-9]+\) \(.+\)$/\1/')
DEVICE_UDID=$(echo "$DEVICE_INFO" | sed -E 's/.*\(([A-Fa-f0-9-]+)\)$/\1/')

print_success "Found: $DEVICE_NAME"
print_success "UDID: $DEVICE_UDID"

cd "$IOS_DIR"

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_step "Cleaning build..."
    xcodebuild -workspace nutri.xcworkspace -scheme nutri -configuration "$CONFIGURATION" clean 2>/dev/null || true
    print_success "Build cleaned"
fi

# Build the app
print_step "Building $CONFIGURATION configuration for device..."
echo ""

xcodebuild \
    -workspace nutri.xcworkspace \
    -scheme nutri \
    -destination "id=$DEVICE_UDID" \
    -configuration "$CONFIGURATION" \
    build \
    2>&1 | grep -E "(error:|warning:|BUILD|Signing|Compiling|Linking)" | head -50

# Check if build succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

print_success "Build succeeded"

# Find the built app
DERIVED_DATA=$(xcodebuild -workspace nutri.xcworkspace -scheme nutri -showBuildSettings 2>/dev/null | grep -m 1 "BUILT_PRODUCTS_DIR" | awk '{print $3}')
APP_PATH="$DERIVED_DATA/nutri.app"

if [ ! -d "$APP_PATH" ]; then
    # Fallback to common location
    APP_PATH="$HOME/Library/Developer/Xcode/DerivedData/nutri-*/Build/Products/$CONFIGURATION-iphoneos/nutri.app"
    APP_PATH=$(ls -d $APP_PATH 2>/dev/null | head -1)
fi

if [ ! -d "$APP_PATH" ]; then
    print_error "Could not find built app"
    exit 1
fi

print_success "App bundle: $APP_PATH"

# Install the app
print_step "Installing on $DEVICE_NAME..."
xcrun devicectl device install app --device "$DEVICE_UDID" "$APP_PATH" 2>&1 | grep -E "(App installed|bundleID|Error)"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    print_error "Installation failed"
    exit 1
fi

print_success "App installed"

# Launch the app
print_step "Launching app..."
BUNDLE_ID=$(defaults read "$APP_PATH/Info.plist" CFBundleIdentifier 2>/dev/null || echo "com.anonymous.nutri")
xcrun devicectl device process launch --device "$DEVICE_UDID" "$BUNDLE_ID" 2>&1 | grep -E "(Launched|Error)"

print_success "App launched on $DEVICE_NAME"

echo ""
print_success "Deployment complete!"
echo ""
echo "Configuration: $CONFIGURATION"
echo "Device: $DEVICE_NAME"
echo "Bundle ID: $BUNDLE_ID"
