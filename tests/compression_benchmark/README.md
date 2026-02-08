# Compression Benchmark Suite

This directory contains tools and scripts for comprehensive benchmarking of the prompt compression feature.

## Quick Start (with uv)

```bash
cd tests/compression_benchmark

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt

# Collect prompts
uv run python collect_prompts.py

# Run benchmarks
uv run python run_benchmark.py

# Generate report
uv run python generate_report.py
```

## Overview

The benchmark suite evaluates compression performance across:
- **Compression Levels**: light, medium, aggressive
- **Scoring Modes**: heuristic, semantic, hybrid
- **Prompt Categories**: instructions, structured, conversations, technical, mixed

**Note**: The benchmark automatically filters out prompts with < 50 tokens, as very short prompts don't benefit from compression due to Hieratic format overhead (~10-15 tokens). This ensures the benchmark focuses on realistic compression scenarios.

## Setup

### 1. Install Python Dependencies

**Option A: Using `uv` (Recommended - Faster)**

```bash
cd tests/compression_benchmark

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv pip install -r requirements.txt
# OR use pyproject.toml (modern approach)
uv sync
```

**Option B: Using `pip` (Traditional)**

```bash
cd tests/compression_benchmark
pip install -r requirements.txt
```

**Option C: Using `uv` with virtual environment (Isolated)**

```bash
cd tests/compression_benchmark

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Build Tokuin with Compression Features

```bash
# From project root
cargo build --release --features compression,compression-embeddings
```

### 3. Collect Test Prompts

**With uv:**
```bash
uv run python collect_prompts.py
```

**With pip:**
```bash
python collect_prompts.py
```

This will:
- Create the prompt directory structure
- Collect real-world prompts from public sources
- Generate metadata with attribution
- Organize prompts by category

## Running Benchmarks

### Run Full Benchmark Suite

**With uv:**
```bash
uv run python run_benchmark.py
```

**With pip:**
```bash
python run_benchmark.py
```

This will:
- Execute compression tests across all scenarios
- Collect metrics (compression ratio, quality, performance)
- Save results to `results/benchmark_results.json`

### Generate Report

**With uv:**
```bash
uv run python generate_report.py
```

**With pip:**
```bash
python generate_report.py
```

This will:
- Process collected results
- Generate comprehensive markdown report
- Save to `COMPRESSION_BENCHMARK_REPORT.md` in project root

### Run Everything

**With uv (recommended):**
```bash
# Collect prompts (if not done already)
uv run python collect_prompts.py

# Run benchmarks
uv run python run_benchmark.py

# Generate report
uv run python generate_report.py
```

**With pip/traditional:**
```bash
# Collect prompts (if not done already)
python collect_prompts.py

# Run benchmarks
python run_benchmark.py

# Generate report
python generate_report.py
```

## Directory Structure

```
compression_benchmark/
├── prompts/              # Test dataset
│   ├── instructions/     # Instruction prompts
│   ├── structured/      # JSON/HTML/table prompts
│   ├── conversations/    # Multi-turn conversations
│   ├── technical/       # Technical documentation
│   ├── mixed/           # Mixed content prompts
│   └── METADATA.md      # Dataset documentation
├── results/             # Benchmark results (JSON)
├── collect_prompts.py   # Prompt collection script
├── run_benchmark.py     # Benchmark runner
├── collect_metrics.py   # Metrics aggregation module
├── generate_report.py   # Report generator
├── requirements.txt     # Python dependencies (pip)
├── pyproject.toml      # Python project config (uv/pip)
├── .python-version      # Python version for uv
└── README.md           # This file
```

## Test Matrix

For each prompt, the benchmark tests:
- 3 compression levels × 3 scoring modes = 9 scenarios
- Total: `number_of_prompts × 9` test runs

## Metrics Collected

### Compression Metrics
- Original token count
- Compressed token count
- Compression ratio (%)
- Tokens saved
- Compression time (ms)

### Quality Metrics
- Overall quality score
- Semantic similarity
- Critical instruction preservation
- Information retention
- Structural integrity

### Performance Metrics
- Compression execution time
- Memory usage

## Report Sections

The generated report includes:
1. **Executive Summary** - Key findings and overall metrics
2. **Test Configuration** - Test matrix and environment
3. **Compression Performance** - Tables by level, scoring mode, category
4. **Quality Analysis** - Quality metrics breakdown
5. **Performance Analysis** - Speed and resource usage
6. **Scenario Analysis** - Best/worst performing scenarios
7. **Recommendations** - Optimal settings for different use cases
8. **Success Thresholds** - Derived criteria based on results

## Customization

### Adding More Prompts

1. Add prompt files to appropriate category directories
2. Update `METADATA.md` with source attribution
3. Re-run benchmarks

### Modifying Test Matrix

Edit `run_benchmark.py`:
- `COMPRESSION_LEVELS` - Add/remove levels
- `SCORING_MODES` - Add/remove scoring modes
- `CATEGORIES` - Add/remove categories

### Custom Report Sections

Edit `generate_report.py` to add custom analysis sections.

## Troubleshooting

### "tokuin binary not found"

Ensure tokuin is built:
```bash
cargo build --release --features compression,compression-embeddings
```

Or specify path in `run_benchmark.py`:
```python
TOKUIN_BINARY = "./target/release/tokuin"
```

### "No prompts found"

Run prompt collection:
```bash
python collect_prompts.py
```

### Missing Python Dependencies

Install requirements:

**With uv (faster):**
```bash
uv pip install -r requirements.txt
# OR
uv sync
```

**With pip (traditional):**
```bash
pip install -r requirements.txt
```

### Quality Metrics Not Available

Ensure embeddings are set up:
```bash
tokuin setup models
```

## Output Files

- `results/benchmark_results.json` - Raw benchmark results
- `COMPRESSION_BENCHMARK_REPORT.md` - Generated report (project root)

## License

Test prompts are collected from public sources with proper attribution.
See `prompts/METADATA.md` for details.

