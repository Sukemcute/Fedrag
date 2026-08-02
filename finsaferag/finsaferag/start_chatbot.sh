#!/usr/bin/env bash

set -u

echo "=================================================="
echo "   Federated RAG Chatbot with Privacy Protection"
echo "=================================================="
echo ""

# Determine script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

echo "✓ Python: $(python3 --version)"
echo ""

# Check project structure
if [ ! -f "config.toml" ]; then
    echo "❌ config.toml not found. Run from project root"
    exit 1
fi

# Setup logs
mkdir -p logs
echo "✓ Logs directory ready"
echo ""

# FIX: Ray path issues with spaces (CRITICAL for path: "AI privace")
export RAY_SCRATCH_DIR="$SCRIPT_DIR/.ray"
export RAY_TMPDIR="$SCRIPT_DIR/.ray"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
mkdir -p "$RAY_SCRATCH_DIR"
echo "✓ Ray scratch dir: $RAY_SCRATCH_DIR"
echo ""

FLOWER_PID=""
STREAMLIT_PID=""

cleanup() {
    echo ""
    echo "=================================================="
    echo "   Shutting down..."
    echo "=================================================="
    echo ""

    if [ -n "$STREAMLIT_PID" ]; then
        kill "$STREAMLIT_PID" 2>/dev/null && echo "✓ Stopped Streamlit" || true
    fi

    if [ -n "$FLOWER_PID" ]; then
        kill "$FLOWER_PID" 2>/dev/null && echo "✓ Stopped Flower" || true
    fi

    pkill -f "flwr run" 2>/dev/null || true
    pkill -f "streamlit" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true

    echo ""
    echo "✓ All services stopped"
    echo ""
}

trap cleanup INT TERM

# ============================================================
# Start Flower
# ============================================================
echo "🌸 Starting Flower Federated Server..."
echo "=================================================="

flwr run . > logs/flower.log 2>&1 &
FLOWER_PID=$!

echo "✓ Flower started (PID: $FLOWER_PID)"
echo "  Logs: logs/flower.log"
echo ""
echo "⏳ Waiting for server & clients..."

max_wait=40
waited=0

while [ $waited -lt $max_wait ]; do
    # Check for server ready message (from server_app.py)
    if grep -q "Federated RAG Server Ready\|Server Ready\|FastAPI started" logs/flower.log 2>/dev/null; then
        echo "✓ Flower & FastAPI ready!"
        sleep 2
        break
    fi
    
    # Also check if process is still alive
    if ! kill -0 "$FLOWER_PID" 2>/dev/null; then
        echo "❌ Flower process died! Check logs/flower.log"
        exit 1
    fi
    
    sleep 1
    ((waited++))
done

if [ $waited -ge $max_wait ]; then
    echo "⚠️  Timeout waiting for server, but continuing..."
    echo "   Check logs/flower.log for details"
fi

echo ""

# ============================================================
# Start Streamlit
# ============================================================
echo "🎨 Starting Streamlit UI..."
echo "=================================================="
echo ""
echo "✅ Open: http://localhost:8501"
echo ""
echo "Shortcuts:"
echo "  Ctrl+C: Stop all"
echo ""
echo "Logs: logs/flower.log"
echo ""
echo "=================================================="
echo ""

cd "$SCRIPT_DIR/ui"
streamlit run streamlit_app.py

cleanup
