#!/bin/bash
# Run baseline experiments on GAIA dataset

set -e

echo "=========================================="
echo "GAIA Baseline Experiment"
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

# Run baseline
echo "Running baseline (target model only, no speculation)..."
echo ""

python run_baseline.py \
    --dataset gaia \
    --num-queries 10

echo ""
echo "✅ Baseline experiment complete!"
echo "Results saved in results/ directory"
