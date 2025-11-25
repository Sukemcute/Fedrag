#!/bin/bash

# Start script for RAG Chatbot (Linux/Mac)

echo "=================================================="
echo "   RAG Chatbot with Privacy Protection"
echo "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if in correct directory
if [ ! -f "config.toml" ]; then
    echo "❌ Please run this script from RAGTest directory"
    exit 1
fi

echo "Starting FastAPI Backend..."
echo "=================================================="

# Start FastAPI in background
cd api
python3 main.py > ../logs/api.log 2>&1 &
API_PID=$!
cd ..

echo "✓ FastAPI started (PID: $API_PID)"
echo "  Logs: logs/api.log"
echo "  API Docs: http://localhost:8000/api/docs"
echo ""

# Wait for API to start
echo "Waiting for API to be ready..."
sleep 5

# Check if API is running
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "✓ API is healthy"
else
    echo "⚠️ API may not be ready yet, check logs/api.log"
fi

echo ""
echo "Starting Streamlit UI..."
echo "=================================================="

# Start Streamlit
cd ui
streamlit run streamlit_app.py

# Cleanup when Streamlit exits
echo ""
echo "Stopping API..."
kill $API_PID
echo "✓ Shutdown complete"

