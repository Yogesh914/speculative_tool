#!/bin/bash
# Run baseline experiments on HotPotQA dataset

set -e

echo "=========================================="
echo "HotPotQA Baseline Experiment"
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

# Run baseline and judge accuracy separately
echo "Running baseline (target model only, no speculation)..."
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="results/baseline_hotpot_${TIMESTAMP}.json"

python run_baseline.py \
    --dataset hotpot \
    --num-queries 10 \
    --results-file "$RESULT_FILE"

echo ""
echo "Judging accuracy with Gemini..."
python judge_results.py "$RESULT_FILE"

echo ""
echo "✅ Baseline experiment complete!"
echo "Results saved to $RESULT_FILE"
