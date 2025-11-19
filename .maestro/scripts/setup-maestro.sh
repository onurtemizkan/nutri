#!/bin/bash

# Maestro Setup Script
# This script installs and configures Maestro for local E2E testing

set -e  # Exit on error

echo "🎭 Maestro E2E Testing Setup"
echo "=============================="
echo ""

# Check if Maestro is already installed
if command -v maestro &> /dev/null; then
    echo "✅ Maestro is already installed"
    maestro --version
    echo ""
    read -p "Do you want to reinstall? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping installation..."
        SKIP_INSTALL=true
    fi
fi

# Install Maestro
if [ -z "$SKIP_INSTALL" ]; then
    echo "📥 Installing Maestro CLI..."
    curl -Ls "https://get.maestro.mobile.dev" | bash

    # Add to PATH for this session
    export PATH="$HOME/.maestro/bin:$PATH"

    echo ""
    echo "✅ Maestro installed successfully!"
    maestro --version
    echo ""
fi

# Check system requirements
echo "🔍 Checking system requirements..."
echo ""

# Check Node.js
if command -v node &> /dev/null; then
    echo "✅ Node.js: $(node --version)"
else
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check Yarn
if command -v yarn &> /dev/null; then
    echo "✅ Yarn: $(yarn --version)"
else
    echo "❌ Yarn not found. Please install Yarn"
    exit 1
fi

# Check platform-specific requirements
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected"

    # Check for Xcode (iOS)
    if xcodebuild -version &> /dev/null; then
        echo "✅ Xcode: $(xcodebuild -version | head -n 1)"
    else
        echo "⚠️  Xcode not found (required for iOS testing)"
    fi

    # Check for Android Studio
    if [ -d "$HOME/Library/Android/sdk" ]; then
        echo "✅ Android SDK found"
    else
        echo "⚠️  Android SDK not found (required for Android testing)"
    fi

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux detected"

    # Check for Android SDK
    if [ -d "$HOME/Android/Sdk" ] || [ -d "$ANDROID_HOME" ]; then
        echo "✅ Android SDK found"
    else
        echo "⚠️  Android SDK not found (required for Android testing)"
    fi
fi

echo ""
echo "📱 Setting up test environment..."

# Navigate to project root (assuming script is in .maestro/scripts/)
cd "$(dirname "$0")/../.."

# Check if we're in the right directory
if [ ! -f "app.json" ]; then
    echo "❌ Error: Not in Expo project root directory"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing project dependencies..."
    yarn install
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "🧪 Validating test flows..."

# Validate YAML syntax
YAML_ERROR=false
for yaml_file in .maestro/flows/**/*.yaml; do
    if [ -f "$yaml_file" ]; then
        if ! maestro test --dry-run "$yaml_file" &> /dev/null; then
            echo "❌ Invalid YAML: $yaml_file"
            YAML_ERROR=true
        fi
    fi
done

if [ "$YAML_ERROR" = false ]; then
    echo "✅ All test flows validated"
else
    echo "⚠️  Some test flows have errors (check above)"
fi

echo ""
echo "📋 Test Suite Summary:"
echo "  • Auth flows: $(ls -1 .maestro/flows/01-auth/*.yaml 2>/dev/null | wc -l) files"
echo "  • Meal flows: $(ls -1 .maestro/flows/02-meals/*.yaml 2>/dev/null | wc -l) files"
echo "  • Profile flows: $(ls -1 .maestro/flows/03-profile/*.yaml 2>/dev/null | wc -l) files"
echo "  • Navigation flows: $(ls -1 .maestro/flows/04-navigation/*.yaml 2>/dev/null | wc -l) files"
echo "  • Test suites: $(ls -1 .maestro/flows/suites/*.yaml 2>/dev/null | wc -l) files"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next Steps:"
echo ""
echo "1. Build your app for testing:"
echo "   iOS:     npx expo run:ios"
echo "   Android: npx expo run:android"
echo ""
echo "2. Run your first test:"
echo "   maestro test .maestro/flows/suites/smoke-tests.yaml"
echo ""
echo "3. Or use Maestro Studio (interactive):"
echo "   maestro studio"
echo ""
echo "4. View all available commands:"
echo "   maestro --help"
echo ""
echo "📖 Full documentation: .maestro/README.md"
echo ""
