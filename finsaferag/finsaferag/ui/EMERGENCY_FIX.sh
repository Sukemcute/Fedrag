#!/usr/bin/env bash

echo "=================================================="
echo "   EMERGENCY FIX - Force Clean Everything"
echo "=================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "🛑 Step 1: Kill all processes..."
pkill -9 -f "streamlit" 2>/dev/null && echo "  ✓ Killed Streamlit" || echo "  - No Streamlit"
pkill -9 -f "flwr" 2>/dev/null && echo "  ✓ Killed Flower" || echo "  - No Flower"
pkill -9 -f "uvicorn" 2>/dev/null && echo "  ✓ Killed Uvicorn" || echo "  - No Uvicorn"
pkill -9 -f "ray" 2>/dev/null && echo "  ✓ Killed Ray" || echo "  - No Ray"

echo ""
echo "🗑️  Step 2: Clear all cache..."
rm -rf .ray/ 2>/dev/null && echo "  ✓ Removed .ray/" || true
rm -rf .streamlit/ 2>/dev/null && echo "  ✓ Removed .streamlit/" || true
rm -rf ui/.streamlit/ 2>/dev/null && echo "  ✓ Removed ui/.streamlit/" || true
rm -rf /tmp/ray 2>/dev/null && echo "  ✓ Removed /tmp/ray" || true
find . -name "*.pyc" -delete 2>/dev/null && echo "  ✓ Removed .pyc files" || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null && echo "  ✓ Removed __pycache__" || true

echo ""
echo "⏳ Step 3: Waiting for cleanup..."
sleep 3

echo ""
echo "🚀 Step 4: Starting services..."
./start_chatbot.sh

echo ""
echo "=================================================="
echo "✅ Emergency fix complete!"
echo ""
echo "Next steps:"
echo "  1. Open browser: http://localhost:8501"
echo "  2. Press Ctrl+Shift+R (hard refresh)"
echo "  3. Click 🧹 Clean button in sidebar"
echo "  4. Test with a new question"
echo "=================================================="
