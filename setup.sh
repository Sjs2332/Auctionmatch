#!/bin/bash

# AuctionMatch Setup Script
echo "📦 Setting up AuctionMatch..."

# 1. Setup Python Backend
echo "🐍 Configuring Python backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# 2. Setup Frontend
echo "⚛️ Configuring Next.js frontend..."
npm install

echo "✅ Setup complete! Run 'npm run dev' to start the application."
