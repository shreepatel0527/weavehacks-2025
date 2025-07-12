#!/bin/bash

echo "Building Claude-Flow Web App for production..."

# Install dependencies
echo "Installing dependencies..."
npm install
cd client && npm install && cd ..

# Build the React app
echo "Building React frontend..."
cd client && npm run build && cd ..

# Create necessary directories
mkdir -p scratch uploads visualizations

echo "Build complete! To start the production server, run:"
echo "npm start"