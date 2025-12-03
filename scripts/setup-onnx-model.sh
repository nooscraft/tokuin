#!/bin/bash
# Script to download and convert the ONNX model for semantic scoring

set -e

MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR="$HOME/Library/Caches/tokuin/models/all-MiniLM-L6-v2"
ONNX_OUTPUT_DIR="./onnx_model_temp"

echo "🔧 Setting up ONNX model for semantic scoring..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed"
    echo "   Please install Python 3.8+ to continue"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed"
    echo "   Please install pip to continue"
    exit 1
fi

echo "📦 Installing required Python packages..."
pip3 install --quiet --upgrade pip
pip3 install --quiet "optimum[exporters]" onnx

echo ""
echo "🔄 Converting model to ONNX format..."
echo "   This may take a few minutes..."
echo ""

# Convert model to ONNX
optimum-cli export onnx \
    --model "$MODEL_ID" \
    --task feature-extraction \
    "$ONNX_OUTPUT_DIR" 2>&1 | grep -v "^$" || true

echo ""
echo "📁 Creating cache directory..."
mkdir -p "$CACHE_DIR"

echo "📋 Copying model files..."
if [ -f "$ONNX_OUTPUT_DIR/model.onnx" ]; then
    cp "$ONNX_OUTPUT_DIR/model.onnx" "$CACHE_DIR/"
    echo "✓ ONNX model copied to: $CACHE_DIR/model.onnx"
    
    # Get file size
    SIZE=$(du -h "$CACHE_DIR/model.onnx" | cut -f1)
    echo "   File size: $SIZE"
else
    echo "❌ Error: model.onnx not found in output directory"
    exit 1
fi

# Copy tokenizer if it doesn't exist (should already be there)
if [ ! -f "$CACHE_DIR/tokenizer.json" ]; then
    if [ -f "$ONNX_OUTPUT_DIR/tokenizer.json" ]; then
        cp "$ONNX_OUTPUT_DIR/tokenizer.json" "$CACHE_DIR/"
        echo "✓ Tokenizer copied"
    fi
fi

echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf "$ONNX_OUTPUT_DIR"

echo ""
echo "✅ Setup complete!"
echo ""
echo "You can now test with:"
echo "  ./target/release/tokuin compress your-prompt.txt --scoring semantic --model gpt-4"
echo ""
echo "Expected output should show:"
echo "  ✓ ONNX model loaded successfully"

