#!/bin/bash
# Compare baseline vs speculation results

set -e

echo "=========================================="
echo "Compare Baseline vs Speculation"
echo "=========================================="
echo ""

# Change to parent directory to access results and python scripts
cd "$(dirname "$0")/.."

# Check if results directory exists
if [ ! -d results ]; then
    echo "❌ Error: results/ directory not found"
    echo "Please run baseline and speculation experiments first"
    exit 1
fi

# Find latest baseline and speculation results
BASELINE=$(ls -t results/baseline_*.json 2>/dev/null | head -1)
SPECULATION=$(ls -t results/speculation_*.json 2>/dev/null | head -1)

if [ -z "$BASELINE" ]; then
    echo "❌ Error: No baseline results found"
    echo "Run: python run_baseline.py first"
    exit 1
fi

if [ -z "$SPECULATION" ]; then
    echo "❌ Error: No speculation results found"
    echo "Run: python run_speculation.py first"
    exit 1
fi

echo "Comparing:"
echo "  Baseline:    $BASELINE"
echo "  Speculation: $SPECULATION"
echo ""

python compare_results.py "$BASELINE" "$SPECULATION"

echo ""
echo "✅ Comparison complete!"
