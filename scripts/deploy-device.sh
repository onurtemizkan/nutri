#!/bin/bash
# Deploy Development Build to Physical iPhone
# Usage: ./scripts/deploy-device.sh [--local] [--clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
LOCAL_BUILD=false
CLEAN_BUILD=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --local) LOCAL_BUILD=true ;;
        --clean) CLEAN_BUILD=true ;;
        --help|-h)
            echo "Deploy Development Build to Physical iPhone"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --local    Build locally using your machine (requires Xcode)"
            echo "  --clean    Clean build cache before building"
            echo "  --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0              # Build on EAS cloud servers"
            echo "  $0 --local      # Build locally on your Mac"
            echo "  $0 --local --clean  # Clean local build"
            exit 0
            ;;
        *) print_error "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

cd "$PROJECT_DIR"

# Check prerequisites
print_step "Checking prerequisites..."

if ! command -v eas &> /dev/null; then
    print_error "EAS CLI not found. Installing..."
    npm install -g eas-cli
fi

if ! eas whoami &> /dev/null; then
    print_error "Not logged in to EAS. Please run: eas login"
    exit 1
fi
print_success "EAS CLI authenticated"

if [ "$LOCAL_BUILD" = true ]; then
    if ! command -v fastlane &> /dev/null; then
        print_error "Fastlane not found. Please install: brew install fastlane"
        exit 1
    fi
    print_success "Fastlane available"

    if ! command -v xcodebuild &> /dev/null; then
        print_error "Xcode not found. Please install Xcode from the App Store"
        exit 1
    fi
    print_success "Xcode available"
fi

# Get local IP for API
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
print_success "Local IP: $LOCAL_IP"

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_step "Cleaning build cache..."
    rm -rf "$BUILD_DIR"
    rm -rf ios/build
    rm -rf ios/Pods
    npx expo prebuild --clean --platform ios 2>/dev/null || true
    print_success "Build cache cleaned"
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Build the app
if [ "$LOCAL_BUILD" = true ]; then
    print_step "Building locally (this may take 10-20 minutes)..."
    echo ""
    print_warning "You will be prompted for Apple Developer credentials"
    print_warning "Device registration may be required for first-time builds"
    echo ""

    # Local build with EAS
    eas build --profile development-local --platform ios --local --output "$BUILD_DIR/nutri-dev.ipa"

    if [ -f "$BUILD_DIR/nutri-dev.ipa" ]; then
        print_success "Build complete: $BUILD_DIR/nutri-dev.ipa"

        # Check for ios-deploy
        if command -v ios-deploy &> /dev/null; then
            print_step "Installing to connected device..."
            ios-deploy --bundle "$BUILD_DIR/nutri-dev.ipa"
            print_success "App installed!"
        else
            print_warning "ios-deploy not found. Install with: brew install ios-deploy"
            echo ""
            echo "To install manually:"
            echo "1. Open Finder and navigate to: $BUILD_DIR"
            echo "2. AirDrop the .ipa file to your iPhone"
            echo "3. Or use Apple Configurator 2"
        fi
    fi
else
    print_step "Building on EAS cloud servers..."
    echo ""
    print_warning "You will be prompted for Apple Developer credentials"
    print_warning "Device registration may be required for first-time builds"
    echo ""

    # Cloud build
    eas build --profile device --platform ios

    echo ""
    print_success "Build submitted to EAS!"
    echo ""
    echo "Next steps:"
    echo "1. Wait for build to complete (check expo.dev dashboard)"
    echo "2. Scan QR code or click download link from EAS"
    echo "3. Install the app on your iPhone"
fi

echo ""
print_step "Post-installation setup:"
echo ""
echo "1. Start the backend server:"
echo "   cd server && npm run dev"
echo ""
echo "2. Start Metro bundler:"
echo "   API_URL=http://$LOCAL_IP:3000/api npx expo start"
echo ""
echo "3. Open the installed Nutri app on your iPhone"
echo "   It will connect to Metro at: exp://$LOCAL_IP:8081"
echo ""
echo "Test credentials:"
echo "   Email: test@nutri-e2e.local"
echo "   Password: TestPass123"
