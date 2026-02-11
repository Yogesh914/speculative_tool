#!/bin/bash
# Run speculation experiments on HotPotQA dataset

set -e

echo "=========================================="
echo "HotPotQA Speculation Experiment"
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

# Check if hotpot_train.csv exists
if [ ! -f ../hotpot_train.csv ]; then
    echo "❌ Error: hotpot_train.csv not found"
    echo "Expected location: ../hotpot_train.csv (parent directory)"
    exit 1
fi

# Run speculation
echo "Running speculation (draft + target models in parallel)..."
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="results/speculation_hotpot_${TIMESTAMP}.json"

python run_speculation.py \
    --dataset hotpot \
    --num-queries 10 \
    --results-file "$RESULT_FILE"

echo ""
echo "Judging accuracy with Gemini..."
python judge_results.py "$RESULT_FILE"

echo ""
echo "✅ Speculation experiment complete!"
echo "Results saved to $RESULT_FILE"
