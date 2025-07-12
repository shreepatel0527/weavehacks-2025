#!/bin/bash

echo "Starting Claude-Flow Web App..."

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing backend dependencies..."
    npm install
fi

if [ ! -d "client/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd client && npm install && cd ..
fi

# Create necessary directories
mkdir -p scratch uploads visualizations

# Start the application in development mode
echo "Starting development server..."
npm run dev