#!/bin/bash
# Script to install optimum with all required extras

set -e

echo "📦 Installing optimum with exporters support..."
echo ""

# Check if we're in a venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Activate your venv first: source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Installing packages..."
pip install --upgrade pip
pip install transformers torch onnx onnxruntime sentence-transformers onnxscript

echo ""
echo "✅ Installation complete!"
echo ""
echo "Verifying installation..."
python3 -c "
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    print('✓ transformers and torch available')
    try:
        from sentence_transformers import SentenceTransformer
        print('✓ sentence-transformers available')
    except ImportError:
        print('⚠️  sentence-transformers not available (optional)')
    print('✅ All required packages installed correctly!')
except ImportError as e:
    print(f'❌ Error: {e}')
    print('   Try: pip install transformers torch onnx onnxruntime sentence-transformers')
    exit(1)
"

