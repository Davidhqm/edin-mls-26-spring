#!/bin/bash
#
# Student Version Benchmark Script
# Usage: ./benchmark.sh <folder_name> [options]
#
# Examples:
#   ./benchmark.sh glm_asr_cutile_template
#   ./benchmark.sh glm_asr_triton_example
#   ./benchmark.sh glm_asr_scratch
#   ./benchmark.sh glm_asr_cutile_template --audio /path/to/test.wav
#   ./benchmark.sh glm_asr_cutile_template --warmup 2 --runs 5
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_name>"
    echo ""
    echo "Available folders:"
    for dir in "$SCRIPT_DIR"/*/; do
        if [ -d "$dir" ]; then
            dirname=$(basename "$dir")
            if [ "$dirname" != "__pycache__" ]; then
                echo "  - $dirname"
            fi
        fi
    done
    exit 1
fi

FOLDER="$1"

# Check if folder exists
if [ ! -d "$SCRIPT_DIR/$FOLDER" ]; then
    echo "Error: Folder '$FOLDER' not found in $SCRIPT_DIR"
    exit 1
fi

# Run the Python benchmark
cd "$SCRIPT_DIR"
python benchmark_student.py "$FOLDER" "${@:2}"
