#!/bin/bash
# Run all GraphAgent examples

set -e

cd "$(dirname "$0")/../../.."
source venv/bin/activate

echo "========================================"
echo "Running All GraphAgent Examples"
echo "========================================"
echo ""

examples=(
    "01_basic"
    "02_conditional_routing"
    "03_enhanced_routing"
    "04_checkpointing"
    "05_interrupts_basic"
    "08_rewind"
    "09_parallel_wait_all"
    "10_parallel_wait_any"
    "14_parallel_rewind"
)

for example in "${examples[@]}"; do
    echo "----------------------------------------"
    echo "Running: $example"
    echo "----------------------------------------"
    python -m "contributing.samples.graph_examples.${example}.agent" 2>&1 | grep -v "UserWarning" || true
    echo ""
done

echo "========================================"
echo "✅ All Examples Complete!"
echo "========================================"
