#!/usr/bin/env bash

echo "=================================================="
echo "   Restart Chatbot (Clean Start)"
echo "=================================================="
echo ""

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧹 Cleaning up old processes..."
pkill -f "flwr" 2>/dev/null || true
pkill -f "streamlit" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "ray" 2>/dev/null || true
sleep 2
echo "✓ Old processes killed"
echo ""

echo "🗑️  Removing old Ray cache..."
rm -rf .ray/ 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
echo "✓ Cache cleared"
echo ""

echo "🚀 Starting chatbot..."
echo ""
./start_chatbot.sh

