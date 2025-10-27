#!/bin/bash
# Install FFmpeg on various platforms
#
# Usage: ./scripts/install_ffmpeg.sh

set -e

echo "Installing FFmpeg..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux"

    # Check for package manager
    if command -v apt-get &> /dev/null; then
        echo "Using apt-get"
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        echo "Using yum"
        sudo yum install -y epel-release
        sudo yum install -y ffmpeg ffmpeg-devel
    elif command -v dnf &> /dev/null; then
        echo "Using dnf"
        sudo dnf install -y ffmpeg ffmpeg-devel
    elif command -v pacman &> /dev/null; then
        echo "Using pacman"
        sudo pacman -Sy --noconfirm ffmpeg
    else
        echo "ERROR: No supported package manager found"
        echo "Please install FFmpeg manually: https://ffmpeg.org/download.html"
        exit 1
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"

    if command -v brew &> /dev/null; then
        echo "Using Homebrew"
        brew install ffmpeg
    else
        echo "ERROR: Homebrew not found"
        echo "Install Homebrew first: https://brew.sh"
        echo "Or install FFmpeg manually: https://ffmpeg.org/download.html"
        exit 1
    fi

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows"
    echo "Please install FFmpeg manually:"
    echo "1. Download from https://ffmpeg.org/download.html#build-windows"
    echo "2. Extract the archive"
    echo "3. Add the bin directory to your PATH"
    exit 1

else
    echo "ERROR: Unsupported OS: $OSTYPE"
    echo "Please install FFmpeg manually: https://ffmpeg.org/download.html"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying FFmpeg installation..."

if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg found: $(ffmpeg -version | head -n1)"
else
    echo "✗ ffmpeg not found"
    exit 1
fi

if command -v ffprobe &> /dev/null; then
    echo "✓ ffprobe found: $(ffprobe -version | head -n1)"
else
    echo "✗ ffprobe not found"
    exit 1
fi

echo ""
echo "FFmpeg installation complete!"
