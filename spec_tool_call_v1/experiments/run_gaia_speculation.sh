#!/bin/bash
# Run speculation experiments on GAIA dataset

set -e

echo "=========================================="
echo "GAIA Speculation Experiment"
echo "=========================================="
echo ""

# Change to parent directory to access .env and python scripts
cd "$(dirname "$0")/.."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please copy .env.example to .env and configure your API keys"
    exit 1
fi

# Run speculation
echo "Running speculation (draft + target models in parallel)..."
echo ""

python run_speculation.py \
    --dataset gaia \
    --num-queries 10

echo ""
echo "✅ Speculation experiment complete!"
echo "Results saved in results/ directory"
