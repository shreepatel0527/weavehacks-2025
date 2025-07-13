#!/bin/bash

# Start the Lab AI Assistant with Safety Monitoring
echo "🔬 Starting Lab AI Assistant with Safety Monitoring..."
echo "📍 Working directory: $(pwd)"
echo "🌐 The app will open in your browser automatically"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed. Installing..."
    pip install streamlit
fi

# Start the Streamlit app
echo "🚀 Launching application..."
streamlit run app_unified.py

echo "✅ Application started successfully!"
