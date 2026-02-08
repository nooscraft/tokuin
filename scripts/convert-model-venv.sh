#!/bin/bash
# Wrapper script to run convert-model.py with .venv activated

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Error: .venv directory not found at $VENV_DIR"
    echo "   Please create a virtual environment first:"
    echo "     python3 -m venv .venv"
    echo "     source .venv/bin/activate"
    echo '     pip install "optimum[exporters]" transformers onnx'
    exit 1
fi

echo "🐍 Using virtual environment: $VENV_DIR"
echo ""

# Activate venv and run the script
source "$VENV_DIR/bin/activate"
python3 "$SCRIPT_DIR/convert-model.py"

