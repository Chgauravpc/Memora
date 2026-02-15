#!/bin/bash

# Quick Start Script for Memory System Phase 1

set -e

echo "=========================================="
echo "  Memory System Phase 1 - Quick Start"
echo "=========================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker"
    exit 1
fi

echo "✓ Docker found: $(docker --version)"

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose"
    exit 1
fi

echo "✓ Docker Compose found: $(docker-compose --version)"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Start Redis
echo "Starting Redis..."
docker-compose up -d
sleep 2
echo "✓ Redis started"
echo ""

# Check Redis health
if docker-compose ps | grep -q "Up"; then
    echo "✓ Redis is healthy"
else
    echo "❌ Redis failed to start"
    docker-compose logs redis
    exit 1
fi

echo ""
echo "=========================================="
echo "  Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run the demo:  python demo.py"
echo "  2. Check README:  cat README.md"
echo "  3. View logs:     docker-compose logs -f"
echo "  4. Stop Redis:    docker-compose down"
echo ""
