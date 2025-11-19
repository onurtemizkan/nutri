#!/bin/bash

# Setup Redis - Install and configure local Redis for development/testing

set -e  # Exit on error

echo "🔧 Setting up Redis..."

# Detect OS
OS=$(uname -s)

install_redis_mac() {
    echo "📦 Installing Redis on macOS..."

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    # Check if Redis is already installed
    if command -v redis-server &> /dev/null; then
        echo "✅ Redis is already installed ($(redis-server --version))"
    else
        echo "📥 Installing Redis via Homebrew..."
        brew install redis
        echo "✅ Redis installed successfully"
    fi

    # Configure Redis to not start automatically
    echo "⚙️  Configuring Redis..."
    brew services stop redis 2>/dev/null || true

    echo "✅ Redis setup complete!"
    echo ""
    echo "🚀 To start Redis manually: npm run redis:start"
    echo "🛑 To stop Redis: npm run redis:stop"
    echo "📊 To check Redis status: npm run redis:status"
}

install_redis_linux() {
    echo "📦 Installing Redis on Linux..."

    # Detect Linux distribution
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        echo "📥 Installing Redis via apt..."
        sudo apt-get update
        sudo apt-get install -y redis-server

        # Stop Redis service (we'll start it manually for testing)
        sudo systemctl stop redis-server 2>/dev/null || true
        sudo systemctl disable redis-server 2>/dev/null || true

        echo "✅ Redis installed successfully"
    elif [ -f /etc/redhat-release ]; then
        # RedHat/CentOS/Fedora
        echo "📥 Installing Redis via yum..."
        sudo yum install -y redis

        # Stop Redis service
        sudo systemctl stop redis 2>/dev/null || true
        sudo systemctl disable redis 2>/dev/null || true

        echo "✅ Redis installed successfully"
    else
        echo "❌ Unsupported Linux distribution"
        echo "Please install Redis manually: https://redis.io/docs/getting-started/installation/"
        exit 1
    fi

    echo "✅ Redis setup complete!"
    echo ""
    echo "🚀 To start Redis manually: npm run redis:start"
    echo "🛑 To stop Redis: npm run redis:stop"
    echo "📊 To check Redis status: npm run redis:status"
}

# Install based on OS
case "$OS" in
    Darwin)
        install_redis_mac
        ;;
    Linux)
        install_redis_linux
        ;;
    *)
        echo "❌ Unsupported operating system: $OS"
        echo "Please install Redis manually: https://redis.io/docs/getting-started/installation/"
        exit 1
        ;;
esac

# Test Redis installation
if command -v redis-server &> /dev/null; then
    echo ""
    echo "✅ Redis is ready to use!"
    redis-server --version
else
    echo "❌ Redis installation verification failed"
    exit 1
fi
