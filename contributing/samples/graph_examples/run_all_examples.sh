#!/bin/bash
# Run all GraphAgent examples

set -euo pipefail

cd "$(dirname "$0")/../../.."
source .venv/bin/activate

echo "========================================"
echo "Running All GraphAgent Examples"
echo "========================================"
echo ""

examples=(
    "01_basic"
    "02_conditional_routing"
    "03_cyclic_execution"
    "15_enhanced_routing"
    "04_checkpointing"
    "05_interrupts_basic"
    "06_interrupts_reasoning"
    "07_callbacks"
    "08_rewind"
    "09_parallel_wait_all"
    "10_parallel_wait_any"
    "11_parallel_wait_n"
    "12_parallel_checkpointing"
    "13_parallel_interrupts"
    "14_parallel_rewind"
)

failed_examples=()

for example in "${examples[@]}"; do
    echo "----------------------------------------"
    echo "Running: $example"
    echo "----------------------------------------"
    if ! python -m "contributing.samples.graph_examples.${example}.agent" 2>&1 | grep -v "UserWarning"; then
        failed_examples+=("$example")
        echo "FAILED: $example"
    fi
    echo ""
done

echo "========================================"
if [ ${#failed_examples[@]} -eq 0 ]; then
    echo "All Examples Complete!"
else
    echo "FAILED examples: ${failed_examples[*]}"
    exit 1
fi
echo "========================================"
