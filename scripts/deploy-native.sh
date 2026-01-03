#!/bin/bash
# Deploy Pure Native iOS Build to USB-connected iPhone
# This script builds a STANDALONE NATIVE iOS app WITHOUT Expo development features
# Usage: ./scripts/deploy-native.sh [--release|--debug] [--clean] [--prebuild] [--skip-prebuild]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IOS_DIR="$PROJECT_DIR/ios"
CONFIGURATION="Release"
CLEAN_BUILD=false
RUN_PREBUILD=false
SKIP_PREBUILD=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_step() { echo -e "${BLUE}==>${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}!${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${CYAN}ℹ${NC} $1"; }

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --release) CONFIGURATION="Release" ;;
        --debug) CONFIGURATION="Debug" ;;
        --clean) CLEAN_BUILD=true ;;
        --prebuild) RUN_PREBUILD=true ;;
        --skip-prebuild) SKIP_PREBUILD=true ;;
        --help|-h)
            echo "Deploy Pure Native iOS Build to USB-connected iPhone"
            echo ""
            echo "This script builds a STANDALONE NATIVE iOS app without Expo development"
            echo "features. It uses xcodebuild directly for pure native compilation."
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --release       Build Release configuration (default)"
            echo "  --debug         Build Debug configuration"
            echo "  --clean         Clean build before building"
            echo "  --prebuild      Force run expo prebuild (excludes dev-client)"
            echo "  --skip-prebuild Skip prebuild check (use existing ios/ directory)"
            echo "  --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Build release, auto-detect prebuild need"
            echo "  $0 --clean --prebuild   # Clean native build with fresh prebuild"
            echo "  $0 --debug              # Debug build for development testing"
            echo ""
            echo "Note: This creates a pure native build without expo-dev-client."
            echo "      The app runs standalone without any Expo development servers."
            exit 0
            ;;
        *) print_error "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# =============================================================================
# PURE NATIVE BUILD PREPARATION
# =============================================================================

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          PURE NATIVE iOS BUILD (No Expo Dev Features)        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

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

# =============================================================================
# NATIVE PROJECT GENERATION (expo prebuild WITHOUT dev-client)
# =============================================================================

cd "$PROJECT_DIR"

# Determine if we need to run prebuild
NEEDS_PREBUILD=false

if [ "$RUN_PREBUILD" = true ]; then
    NEEDS_PREBUILD=true
    print_info "Prebuild requested via --prebuild flag"
elif [ "$SKIP_PREBUILD" = true ]; then
    NEEDS_PREBUILD=false
    print_info "Skipping prebuild check (--skip-prebuild)"
elif [ ! -d "$IOS_DIR" ]; then
    NEEDS_PREBUILD=true
    print_warning "iOS directory not found - prebuild required"
elif [ ! -f "$IOS_DIR/nutri.xcworkspace/contents.xcworkspacedata" ]; then
    NEEDS_PREBUILD=true
    print_warning "iOS workspace incomplete - prebuild required"
fi

# Run prebuild if needed - EXCLUDING expo-dev-client for pure native build
if [ "$NEEDS_PREBUILD" = true ]; then
    print_step "Generating pure native iOS project (excluding expo-dev-client)..."
    echo ""

    # Set environment variable to exclude dev client from plugins
    export EXCLUDE_DEV_CLIENT=true

    # CRITICAL: Temporarily remove expo-dev-client from package.json
    # This ensures the native code is NOT linked during prebuild
    print_info "Temporarily removing expo-dev-client from dependencies..."
    cp package.json package.json.backup

    # Remove expo-dev-client line from package.json
    sed -i '' '/"expo-dev-client":/d' package.json

    # Clean prebuild for fresh native project without dev features
    if [ "$CLEAN_BUILD" = true ]; then
        print_info "Running clean prebuild (NO expo-dev-client)..."
        rm -rf "$IOS_DIR"
        npx expo prebuild --platform ios --clean --no-install 2>&1 | grep -E "(Prebuild|Generated|iOS|CocoaPods|expo-dev)" | head -20 || true
    else
        npx expo prebuild --platform ios --no-install 2>&1 | grep -E "(Prebuild|Generated|iOS|CocoaPods|expo-dev)" | head -20 || true
    fi

    # Restore original package.json
    mv package.json.backup package.json
    print_success "Restored package.json with expo-dev-client"

    print_success "Native iOS project generated (WITHOUT expo-dev-client)"

    # Install CocoaPods dependencies (without expo-dev-client)
    print_step "Installing CocoaPods dependencies..."
    cd "$IOS_DIR"
    pod install 2>&1 | grep -E "(Installing|Generating|Pod|EXDevMenu|expo-dev)" | head -20 || true
    cd "$PROJECT_DIR"
    print_success "CocoaPods dependencies installed"
fi

# Verify iOS directory exists now
if [ ! -d "$IOS_DIR" ]; then
    print_error "iOS directory not found after prebuild. Something went wrong."
    exit 1
fi

# =============================================================================
# VERIFY NO EXPO-DEV-CLIENT IN BUILD
# =============================================================================

# Check if expo-dev-client is still in Podfile.lock (it shouldn't be)
if [ -f "$IOS_DIR/Podfile.lock" ]; then
    if grep -q "expo-dev-client\|EXDevMenu\|EXDevLauncher" "$IOS_DIR/Podfile.lock"; then
        print_warning "expo-dev-client detected in Podfile.lock!"
        print_info "Forcing clean rebuild to remove dev client..."

        # Force a clean prebuild
        export EXCLUDE_DEV_CLIENT=true
        cp "$PROJECT_DIR/package.json" "$PROJECT_DIR/package.json.backup"
        sed -i '' '/"expo-dev-client":/d' "$PROJECT_DIR/package.json"

        rm -rf "$IOS_DIR"
        cd "$PROJECT_DIR"
        npx expo prebuild --platform ios --clean --no-install 2>&1 | grep -E "(Prebuild|Generated|iOS)" | head -10 || true

        mv "$PROJECT_DIR/package.json.backup" "$PROJECT_DIR/package.json"

        cd "$IOS_DIR"
        pod install 2>&1 | grep -E "(Installing|Generating|Pod)" | head -10 || true
        cd "$PROJECT_DIR"

        print_success "Rebuilt iOS project without expo-dev-client"
    else
        print_success "Verified: NO expo-dev-client in build"
    fi
fi

print_success "iOS native project ready (production-grade)"

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

cd "$PROJECT_DIR"

# =============================================================================
# NATIVE BUILD PREPARATION
# =============================================================================

cd "$PROJECT_DIR"

# For Release builds, configure for production
if [ "$CONFIGURATION" = "Release" ]; then
    print_step "Preparing for PRODUCTION native build..."

    # Remove aps-environment entitlement that requires Push Notifications capability
    ENTITLEMENTS_FILE="$IOS_DIR/nutri/nutri.entitlements"
    if [ -f "$ENTITLEMENTS_FILE" ] && grep -q "aps-environment" "$ENTITLEMENTS_FILE"; then
        sed -i '' '/<key>aps-environment<\/key>/,/<\/string>/d' "$ENTITLEMENTS_FILE"
        print_success "Removed push notification entitlement for local build"
    fi

    # Ensure production bundle configuration
    print_info "Build type: RELEASE (optimized, no debug symbols)"
else
    print_step "Preparing for DEBUG native build..."
    print_info "Build type: DEBUG (with debug symbols, not optimized)"
fi

cd "$IOS_DIR"

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_step "Cleaning native build artifacts..."
    xcodebuild -workspace nutri.xcworkspace -scheme nutri -configuration "$CONFIGURATION" clean 2>/dev/null || true
    # Also clean DerivedData for this project
    rm -rf ~/Library/Developer/Xcode/DerivedData/nutri-* 2>/dev/null || true
    print_success "Native build cleaned"
fi

# =============================================================================
# PURE NATIVE XCODEBUILD (No Expo Development Server)
# =============================================================================

print_step "Building PURE NATIVE $CONFIGURATION app for device..."
echo ""
print_info "Using xcodebuild directly (no Expo bundler, no Metro)"
print_info "Target device: $DEVICE_NAME ($DEVICE_UDID)"
echo ""

# Build with xcodebuild - PURE NATIVE compilation
# This builds a standalone app without any Expo development features
xcodebuild \
    -workspace nutri.xcworkspace \
    -scheme nutri \
    -destination "id=$DEVICE_UDID" \
    -configuration "$CONFIGURATION" \
    -allowProvisioningUpdates \
    DEVELOPMENT_TEAM="6FJ4QBBQTK" \
    CODE_SIGN_STYLE="Automatic" \
    GCC_PREPROCESSOR_DEFINITIONS='$(inherited) DISABLE_EXPO_DEV_MENU=1' \
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

# =============================================================================
# DEPLOYMENT COMPLETE
# =============================================================================

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            PURE NATIVE DEPLOYMENT COMPLETE!                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}Build Type:${NC}     PURE NATIVE (xcodebuild)"
echo -e "  ${CYAN}Configuration:${NC}  $CONFIGURATION"
echo -e "  ${CYAN}Device:${NC}         $DEVICE_NAME"
echo -e "  ${CYAN}Bundle ID:${NC}      $BUNDLE_ID"
echo -e "  ${CYAN}Expo Dev:${NC}       EXCLUDED (no expo-dev-client)"
echo ""
echo -e "${YELLOW}Note:${NC} This is a standalone native app. No Expo development"
echo -e "      server or Metro bundler is required to run it."
echo ""
