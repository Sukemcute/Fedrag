@echo off
REM Start script for RAG Chatbot (Windows)

echo ==================================================
echo    RAG Chatbot with Privacy Protection
echo ==================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python not found. Please install Python 3.8+
    exit /b 1
)

echo OK Python found
echo.

REM Check if in correct directory
if not exist "config.toml" (
    echo X Please run this script from RAGTest directory
    exit /b 1
)

echo Starting FastAPI Backend...
echo ==================================================

REM Create logs directory if not exists
if not exist "logs" mkdir logs

REM Start FastAPI in background
cd api
start /B python main.py > ..\logs\api.log 2>&1
cd ..

echo OK FastAPI started
echo   Logs: logs\api.log
echo   API Docs: http://localhost:8000/api/docs
echo.

REM Wait for API to start
echo Waiting for API to be ready...
timeout /t 5 /nobreak >nul

REM Check if API is running
curl -s http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    echo ! API may not be ready yet, check logs\api.log
) else (
    echo OK API is healthy
)

echo.
echo Starting Streamlit UI...
echo ==================================================
echo.

REM Start Streamlit
cd ui
streamlit run streamlit_app.py

echo.
echo Chatbot stopped.

