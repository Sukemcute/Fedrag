#!/usr/bin/env bash

echo "=================================================="
echo "   Kill All Chatbot Processes"
echo "=================================================="
echo ""

echo "🛑 Stopping all services..."

# Kill by process name
echo "  - Killing Flower processes..."
pkill -9 -f "flwr" 2>/dev/null && echo "    ✓ Flower killed" || echo "    - No Flower found"

echo "  - Killing Streamlit processes..."
pkill -9 -f "streamlit" 2>/dev/null && echo "    ✓ Streamlit killed" || echo "    - No Streamlit found"

echo "  - Killing Uvicorn/FastAPI processes..."
pkill -9 -f "uvicorn" 2>/dev/null && echo "    ✓ Uvicorn killed" || echo "    - No Uvicorn found"

echo "  - Killing Ray processes..."
pkill -9 -f "ray" 2>/dev/null && echo "    ✓ Ray killed" || echo "    - No Ray found"

echo ""
echo "⏳ Waiting for processes to terminate..."
sleep 2

# Check if any still running
REMAINING=$(ps aux | grep -E "flwr|streamlit|uvicorn|ray" | grep -v grep | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo "✅ All processes stopped successfully!"
else
    echo "⚠️  Warning: $REMAINING process(es) still running:"
    ps aux | grep -E "flwr|streamlit|uvicorn|ray" | grep -v grep
    echo ""
    echo "Try running this script again or kill them manually with:"
    echo "  kill -9 <PID>"
fi

echo ""
echo "=================================================="
echo "✓ Done"
echo "=================================================="

