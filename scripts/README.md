# Helper Scripts for Model Setup

This directory contains helper scripts for manually setting up embedding models when the built-in `tokuin setup models` command fails or when you need more control over the conversion process.

## Quick Start

**For most users**: Use the built-in command instead:
```bash
tokuin setup models
```

**Use these scripts only if**:
- `tokuin setup models` fails to download the ONNX model
- You need to convert the model manually
- You're troubleshooting model setup issues
- You're developing/testing the model conversion process

## Scripts Overview

### `setup-onnx-model.sh`
**Purpose**: Downloads and converts the ONNX model using `optimum-cli` (same method used in GitHub Actions)

**When to use**: When you want to use the same conversion method as CI/CD

**Prerequisites**:
- Python 3.8+
- pip3

**Usage**:
```bash
./scripts/setup-onnx-model.sh
```

**What it does**:
1. Installs `optimum[exporters]` and `onnx` via pip
2. Converts the model using `optimum-cli export onnx`
3. Copies the model to the cache directory (`~/.cache/tokuin/models/all-MiniLM-L6-v2/`)

---

### `convert-model.py`
**Purpose**: Alternative Python script for model conversion with multiple fallback methods

**When to use**: 
- When `optimum-cli` doesn't work
- When you need more control over the conversion process
- As a fallback when other methods fail

**Prerequisites**:
- Python 3.8+
- One of:
  - `optimum[exporters]` and `transformers` (Method 1)
  - `transformers`, `torch`, `onnx`, `onnxruntime`, `sentence-transformers`, `onnxscript` (Method 2)

**Usage**:
```bash
# Direct usage
python3 scripts/convert-model.py

# Or with virtual environment (recommended)
./scripts/convert-model-venv.sh
```

**What it does**:
1. Tries multiple conversion methods:
   - **Method 1**: Uses `optimum.onnxruntime.ORTModelForFeatureExtraction` (optimum < 2.0)
   - **Method 2**: Uses `sentence-transformers` or `transformers` + `torch.onnx.export`
2. Automatically detects available packages and uses the best method
3. Copies the converted model to the cache directory

**Features**:
- Auto-detects virtual environment
- Multiple fallback methods
- Better error messages
- Handles different optimum versions

---

### `convert-model-venv.sh`
**Purpose**: Wrapper script to run `convert-model.py` with a virtual environment

**When to use**: When you want to use a virtual environment for model conversion

**Prerequisites**:
- Virtual environment at `.venv/` (create with `python3 -m venv .venv`)

**Usage**:
```bash
# First, create and activate venv (if not exists)
python3 -m venv .venv
source .venv/bin/activate
pip install "optimum[exporters]" transformers onnx onnxruntime sentence-transformers onnxscript

# Then run the wrapper
./scripts/convert-model-venv.sh
```

**What it does**:
1. Checks for `.venv` directory
2. Activates the virtual environment
3. Runs `convert-model.py` with the venv Python

---

### `install-optimum.sh`
**Purpose**: Installs all required Python packages for model conversion

**When to use**: When setting up your environment for manual model conversion

**Prerequisites**:
- Python 3.8+
- pip
- Virtual environment (recommended, but not required)

**Usage**:
```bash
# In a virtual environment (recommended)
source .venv/bin/activate
./scripts/install-optimum.sh

# Or globally (not recommended)
./scripts/install-optimum.sh
```

**What it does**:
1. Checks if you're in a virtual environment (warns if not)
2. Installs: `transformers`, `torch`, `onnx`, `onnxruntime`, `sentence-transformers`, `onnxscript`
3. Verifies the installation

**Installed packages**:
- `transformers` - HuggingFace transformers library
- `torch` - PyTorch (for model conversion)
- `onnx` - ONNX format support
- `onnxruntime` - ONNX runtime
- `sentence-transformers` - Sentence transformers library
- `onnxscript` - ONNX script support (optional, for newer export methods)

---

## Complete Workflow Examples

### Option 1: Using optimum-cli (Recommended)
```bash
# Install dependencies
pip install "optimum[exporters]" onnx

# Run the script
./scripts/setup-onnx-model.sh
```

### Option 2: Using Python script with venv
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
./scripts/install-optimum.sh

# Convert model
./scripts/convert-model-venv.sh
```

### Option 3: Using Python script directly
```bash
# Install dependencies globally (not recommended)
pip3 install transformers torch onnx onnxruntime sentence-transformers onnxscript

# Run conversion
python3 scripts/convert-model.py
```

---

## Troubleshooting

### "optimum-cli: command not found"
**Solution**: Install optimum with exporters:
```bash
pip install "optimum[exporters]"
```

### "No module named 'torch'"
**Solution**: Install PyTorch:
```bash
pip install torch
```

### "ONNX export failed"
**Solution**: Try the alternative method:
```bash
python3 scripts/convert-model.py
```

### "Permission denied" when running scripts
**Solution**: Make scripts executable:
```bash
chmod +x scripts/*.sh
```

### Model conversion takes too long
**Note**: Model conversion can take 5-10 minutes depending on your system. This is normal.

### "CUDA out of memory" or similar GPU errors
**Solution**: The scripts automatically use CPU for conversion. If you see GPU errors, ensure PyTorch is using CPU:
```python
import torch
torch.set_default_tensor_type('torch.FloatTensor')
```

---

## Model Location

All scripts place the converted model in:
- **macOS/Linux**: `~/.cache/tokuin/models/all-MiniLM-L6-v2/`
- **Windows**: `%LOCALAPPDATA%\tokuin\models\all-MiniLM-L6-v2\`

Files created:
- `tokenizer.json` - Tokenizer file (required)
- `model.onnx` - ONNX model file (optional, for better quality)

---

## Relationship to `tokuin setup models`

The built-in `tokuin setup models` command:
1. **First tries**: Download pre-converted ONNX model from HuggingFace
2. **If that fails**: Suggests manual conversion using these scripts

These scripts are the **manual fallback** when automatic download fails.

**When to use `tokuin setup models`**:
- ✅ First-time setup
- ✅ Normal usage
- ✅ When HuggingFace has pre-converted models

**When to use these scripts**:
- ⚠️ When `tokuin setup models` fails
- ⚠️ When HuggingFace doesn't have pre-converted ONNX model
- ⚠️ When you need custom conversion settings
- ⚠️ For development/testing

---

## Development Notes

These scripts are used in:
- **GitHub Actions**: `setup-onnx-model.sh` approach (via `optimum-cli`)
- **Local Development**: `convert-model.py` for testing different methods
- **CI/CD**: Pre-converting models for release binaries

For contributors: These scripts help test model conversion in different environments and with different package versions.

---

## See Also

- [Main README](../README.md) - General Tokuin documentation
- [Build from Source Guide](../README.md#-build-from-source-power-users) - Building Tokuin from source
- [Using ONNX](../docs/USING_ONNX.md) - ONNX integration details

---

**Last Updated**: 2025-12-02

