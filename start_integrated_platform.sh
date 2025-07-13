#!/bin/bash

# WeaveHacks 2025 - Integrated Platform Startup Script
# This script starts both the backend API and frontend Streamlit app

echo "🔬 Starting WeaveHacks Lab Automation Platform..."

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found. Please ensure backend/main.py exists."
    exit 1
fi

# Check if Python virtual environment should be activated
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Start backend API in background
echo "🚀 Starting backend API server..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is running on http://localhost:8000"
else
    echo "❌ Backend API failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
pip install -r requirements.txt

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend..."
echo "🌐 Frontend will be available at http://localhost:8501"

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down platform..."
    kill $BACKEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start the integrated Streamlit app
streamlit run integrated_app.py --server.port 8501 --server.address 0.0.0.0

# Cleanup on normal exit
cleanup