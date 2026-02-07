# 🧮 Tokuin

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/nooscraft/tokuin)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![CI](https://github.com/nooscraft/tokuin/workflows/CI/badge.svg)](https://github.com/nooscraft/tokuin/actions)

A fast, CLI-based tool to estimate **token usage** and **API cost** for prompts targeting various LLM providers (OpenAI, Claude, Mistral, etc.). Built in Rust for performance, portability, and safety.

## ✨ Features

- **Token Count Estimation**: Analyze prompts and count tokens for selected models (e.g., `gpt-4`, `gpt-3.5-turbo`)
- **Cost Estimation**: Calculate API costs based on token pricing per model
- **Multi-Model Comparison**: Compare token usage and cost across multiple providers
- **Role-Based Breakdown**: Show token count by system/user/assistant role messages
- **Multiple Input Formats**: Support plain text and JSON chat formats
- **Flexible Output**: Human-readable text or JSON output for scripting
- **Prompt Compression** (requires `--features compression`): Compress prompts by 70-90% using the Hieratic format - LLM-parseable, semantic-preserving
- **Prompt Library Analysis**: Scan directories of prompts, detect duplicates, show token distribution, and estimate costs at scale
- **Load Testing** (requires `--features load-test`): Run concurrent load tests against LLM APIs with real-time metrics, progress bars, and cost estimation

## 🚀 Installation

### Quick Install (macOS & Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/nooscraft/tokuin/main/install.sh | bash
```

The script detects your platform, downloads the latest release, verifies its checksum, and installs `tokuin` to `/usr/local/bin` (or `~/.local/bin` if root access is unavailable).

**What you get**: The release binary includes **all features** enabled:
- ✅ Token counting and cost estimation
- ✅ Prompt compression with Hieratic format
- ✅ Semantic scoring with ONNX embeddings
- ✅ Quality metrics
- ✅ Load testing capabilities
- ✅ Markdown support
- ✅ Watch mode
- ✅ Gemini tokenization

**Note**: Release binaries include embedding models bundled, so compression features work immediately without running `tokuin setup models`. The script optionally runs `tokuin setup models` after installation (set `TOKUIN_SETUP_MODELS=1` to enable, or use `--skip-models` to disable).

### Quick Install (Windows PowerShell)

```powershell
irm https://raw.githubusercontent.com/nooscraft/tokuin/main/install.ps1 | iex
```

By default the binary is placed in `%LOCALAPPDATA%\Programs\tokuin`. To customize the destination, download the script first (`irm ... -OutFile install.ps1`) and invoke `.\install.ps1 -InstallDir "C:\Tools"`.

**What you get**: The release binary includes **all features** enabled:
- ✅ Token counting and cost estimation
- ✅ Prompt compression with Hieratic format
- ✅ Semantic scoring with ONNX embeddings
- ✅ Quality metrics
- ✅ Load testing capabilities
- ✅ Markdown support
- ✅ Watch mode
- ✅ Gemini tokenization

**Note**: Release binaries include embedding models bundled, so compression features work immediately without running `tokuin setup models`. The script optionally runs `tokuin setup models` after installation (set `$env:TOKUIN_SETUP_MODELS=1` to enable, or use `-SkipModels` to disable).

### From Source

**Basic build (token counting only):**
```bash
git clone https://github.com/nooscraft/tokuin.git
cd tokuin
cargo build --release
```

The binary will be available at `target/release/tokuin`.

**Build with all features (recommended):**
```bash
cargo build --release --features all
```

This includes compression, embeddings, load testing, markdown support, watch mode, and Gemini tokenization.

**After building, setup embedding models (if using compression features):**
```bash
./target/release/tokuin setup models
```

For more details, see the [Build from Source](#-build-from-source-power-users) section below.

### From Releases

Release archives are published for each tag at [GitHub Releases](https://github.com/nooscraft/tokuin/releases). Download the archive matching your OS/architecture, verify it against `checksums.txt`, and place the `tokuin` binary somewhere on your `PATH` (e.g., `/usr/local/bin` or `%LOCALAPPDATA%\Programs\tokuin`).

**What you get**: Release binaries are built with `--features all`, so they include **all features**:
- ✅ Token counting and cost estimation
- ✅ Prompt compression with Hieratic format
- ✅ Semantic scoring with ONNX embeddings
- ✅ Quality metrics
- ✅ Load testing capabilities
- ✅ Markdown support
- ✅ Watch mode
- ✅ Gemini tokenization

**Note**: Release binaries include embedding models bundled, so they work immediately without running `tokuin setup models`. This is the easiest way to get started with all features.

## 📖 Usage

### Basic Token Counting

```bash
echo "Hello, world!" | tokuin --model gpt-4
```

Output:
```
Model: gpt-4
Tokens: 4
```

### With Cost Estimation

```bash
echo "Hello, world!" | tokuin --model gpt-4 --price
```

Output:
```
Model: gpt-4
Tokens: 4
Cost: $0.0001 (input)
```

### With Role Breakdown

```bash
echo '[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello!"}]' | \
  tokuin --model gpt-4 --breakdown --price
```

Output:
```
Model: gpt-4
Tokens: 15

System:     8 tokens
User:       2 tokens
Assistant:  0 tokens
------------------------------
Total:      15 tokens
Cost: $0.0005 (input)
```

### Multi-Model Comparison

```bash
echo "Hello, world!" | tokuin --compare gpt-4 gpt-3.5-turbo --price
```

Output:
```
Model              Tokens    Cost
-----------------------------------------------
gpt-4              4         $0.0001
gpt-3.5-turbo      4         $0.0000
```

### JSON Output

```bash
echo "Hello, world!" | tokuin --model gpt-4 --format json
```

Output:
```json
{
  "model": "gpt-4",
  "tokens": 4,
  "input_cost": null,
  "output_cost": null,
  "breakdown": null
}
```

### Markdown Output (requires `--features markdown`)

```bash
echo "Hello, world!" | tokuin --model gpt-4 --format markdown --price
```

Output:
```markdown
## Token Analysis: gpt-4

**Total Tokens:** 4

### Cost Estimation

- **Input Cost:** $0.0001
- **Output Cost:** $0.0002
```

### Minify Markdown (requires `--features markdown`)

Strip markdown formatting to see token savings:

```bash
echo "# Title\n\n**Bold** text" | tokuin --model gpt-4 --minify
```

### Diff Mode

Compare two prompts to see token differences:

```bash
tokuin prompt.txt --model gpt-4 --diff prompt-v2.txt --price
```

Output:
```
Model: gpt-4
Original: 100 tokens
Modified: 85 tokens
Difference: -15 tokens
Cost difference: $0.0005
```

### Watch Mode (requires `--features watch`)

Automatically re-run analysis when file changes:

```bash
tokuin prompt.txt --model gpt-4 --watch
```

### Reading from File

```bash
tokuin prompt.txt --model gpt-4 --price
```

### Reading from Stdin

```bash
cat prompts.txt | tokuin --model gpt-4
```

### Load Testing (requires `--features load-test`)

Run load tests against LLM APIs to measure performance, latency, and costs:

> Pass credentials via dedicated flags (`--openrouter-api-key`, `--openai-api-key`, `--anthropic-api-key`), `--api-key`, or matching environment variables. The CLI will auto-detect the provider from the model name or endpoint, and you can override it explicitly with `--provider {openai|openrouter|anthropic|generic}`.

```bash
# Basic load test with OpenAI
export OPENAI_API_KEY="sk-openai-..."
echo "What is 2+2?" | tokuin load-test \
  --model gpt-4 \
  --runs 100 \
  --concurrency 10 \
  --openai-api-key "$OPENAI_API_KEY"

# With OpenRouter (access to 400+ models)
export OPENROUTER_API_KEY="sk-or-..."
echo "Hello!" | tokuin load-test \
  --model openai/gpt-4 \
  --runs 50 \
  --concurrency 5 \
  --provider openrouter \
  --openrouter-api-key "$OPENROUTER_API_KEY"

# With Anthropic (direct API)
export ANTHROPIC_API_KEY="sk-ant-..."
echo "Draft a product overview" | tokuin load-test \
  --model claude-3-sonnet \
  --provider anthropic \
  --runs 25 \
  --concurrency 5 \
  --anthropic-api-key "$ANTHROPIC_API_KEY"

# With think time between requests
tokuin load-test --model gpt-4 --runs 200 --concurrency 20 --think-time "250-750ms" --prompt-file prompts.txt

# Dry run to estimate costs without making API calls
echo "Test prompt" | tokuin load-test --model gpt-4 --runs 1000 --concurrency 50 --dry-run --estimate-cost

# Generic provider (bring-your-own endpoint)
echo "Ping" | tokuin load-test \
  --model lambda-1 \
  --provider generic \
  --endpoint https://example.com/api/infer \
  --api-key "token" \
  --runs 10 \
  --concurrency 2

# With retry and cost estimation
tokuin load-test \
  --model gpt-4 \
  --runs 100 \
  --concurrency 10 \
  --retry 3 \
  --estimate-cost \
  --output-format json \
  --openai-api-key "$OPENAI_API_KEY"
```

### Custom Pricing Overrides

Pricing for a handful of popular OpenAI/Gemini models ships in-tree, but rates move quickly. Supply your own TOML file to keep cost projections accurate:

```bash
cp PRICING_TEMPLATE.toml pricing.toml
# edit pricing.toml with the latest numbers

tokuin --pricing-file pricing.toml --model gpt-4 --price prompt.txt

# or via environment variable
export TOKUIN_PRICING_FILE=/path/to/pricing.toml
tokuin load-test --model openrouter/anthropic-sonnet --runs 10 --estimate-cost
```

Keys follow the pattern `[provider.model]` (for example `[openrouter.anthropic-sonnet]`). At runtime the CLI merges overrides with built-in defaults; if a model is missing, cost estimates fall back to any bundled price or simply show token counts.

Output:
```
Starting load test: 100 requests with concurrency 10
⠋ [00:05.234] [████████████████████░░░░] 80/100 Success: 78 | Failed: 2 | Avg Latency: 1250ms | Throughput: 15.2 req/s

=== Load Test Results ===
Total Requests: 100
Successful: 98 (98.0%)
Failed: 2 (2.0%)

Latency (ms):
  Average: 1234.56
  p50: 1200
  p95: 1850

Cost Estimation:
  Input tokens: 5000
  Output tokens: 12000
  Input cost: $0.150000
  Output cost: $0.720000
  Total cost: $0.870000
```

## 🗜️ Prompt Compression with Hieratic Format (requires `--features compression`)

Tokuin includes a powerful prompt compression system that uses the **Hieratic format** - a structured, LLM-parseable format that reduces token usage by 70-90% while remaining directly usable with any LLM.

### Why Hieratic?

Named after ancient Egypt's compressed cursive writing (a practical simplification of hieroglyphics), Hieratic preserves prompt semantics while dramatically reducing costs and latency.

### Workflow

1. **Setup embedding models** (for semantic scoring): `tokuin setup models`
2. **Extract reusable patterns** from your prompt library (optional)
3. **Compress prompts** using those patterns + extractive compression
4. **Use compressed prompts** directly with LLMs (they understand the format natively)

### Setup Embedding Models (for Semantic Scoring)

Before using semantic or hybrid scoring, download the embedding models:

```bash
# Download tokenizer (required for semantic scoring)
tokuin setup models

# Also download ONNX model (optional, for better quality)
tokuin setup models --onnx
```

**Note**: Release binaries include models bundled, so setup is only needed when building from source.

**Troubleshooting**: If `tokuin setup models` fails, see [`scripts/README.md`](scripts/README.md) for manual conversion scripts.

### Extract Context Patterns

Scan a directory of prompts to identify reusable patterns:

```bash
tokuin extract-context ./prompts --output contexts.toml --model gpt-4
```

This creates a `contexts.toml` file containing frequently-used role descriptions, examples, and constraints.

### Compress a Prompt

```bash
tokuin compress my-prompt.txt --level medium --output compressed.hieratic
```

Options:
- `--level`: `light` (30-50%), `medium` (50-70%), `aggressive` (70-90%)
- `--structured`: Enable structured document mode (better for JSON, code, tables, technical docs)
- `--context-lib`: Path to context library (default: `contexts.toml`)
- `--inline`: Force inline mode (no context references)
- `--format`: Output format (`hieratic`, `expanded`, `json`)
- `--quality`: Calculate and display quality metrics (semantic similarity, critical instruction preservation, etc.)
- `--scoring`: Scoring mode (`heuristic`, `semantic`, `hybrid`) - requires `compression-embeddings` feature for semantic/hybrid

Output:
```
Compressing: my-prompt.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compression Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original:  2,450 tokens
Compressed: 420 tokens
Reduction: 82.9% (2,030 tokens saved)

Output: my-prompt.hieratic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

With `--quality` flag:
```
Compressing: my-prompt.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compression Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original:  2,450 tokens
Compressed: 420 tokens
Reduction: 82.9% (2,030 tokens saved)

Output: my-prompt.hieratic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Calculating quality metrics...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quality Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Score: 82.3% (Good)
  ├─ Semantic Similarity: 85.1%
  ├─ Critical Instructions: 3/3 preserved (100.0%)
  ├─ Information Retention: 78.2%
  └─ Structural Integrity: 100.0%

✅ Quality is acceptable (>= 70%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Hieratic Format Example

**Original (850 tokens):**
```
You are an expert programmer with 10 years of experience...
[full verbose role description]

Example 1: Bug Fix in Authentication
[detailed example]
...
```

**Hieratic (285 tokens, 66.5% reduction):**
```hieratic
@HIERATIC v1.0

@ROLE[inline]
"Expert engineer: 15y full-stack, Python/JS/Go, architecture, patterns"

@EXAMPLES[inline]
1. Auth bug: session token bypass → HMAC signing → 94% bot reduction
2. DB perf: 2.3s queries → pooling+cache → 0.1s, 10x capacity

@TASK
Analyze code and provide recommendations

@FOCUS: performance, security, maintainability
@STYLE: concise, actionable
```

### Structured Document Mode

For complex technical documents (JSON-heavy prompts, HTML tables, BNF grammars, code blocks), use `--structured`:

```bash
tokuin compress technical-prompt.txt --structured --level medium
```

**Benefits:**
- ✅ Preserves JSON document structure
- ✅ Keeps HTML tables intact
- ✅ Detects and consolidates repetitive instruction patterns
- ✅ Segments by logical sections (definitions, examples, formats)
- ✅ Structure-aware importance scoring

**When to use `--structured`:**
- LLM extraction/parsing instructions with JSON documents
- Prompts with HTML tables or code blocks
- Technical specifications with repeated formatting rules
- Documents with clear sections (Definition:, Location:, Response Format:)

**When to use default mode:**
- Conversational prompts
- Natural language instructions
- Simple role/task descriptions

### Limitations and Best Practices

**Minimum Prompt Size:**
- Prompts with **< 50 tokens** won't benefit from compression
- Hieratic format adds ~10-15 tokens of overhead
- For very short prompts, compressed version may be larger than original
- Compression ratio will be 0% or negative for prompts < 50 tokens
- **Recommendation**: Use compression for prompts ≥ 50 tokens for best results

**When Compression is Most Effective:**
- ✅ Longer prompts (100+ tokens) - see 70-90% reduction
- ✅ Prompts with repetitive patterns (role descriptions, examples)
- ✅ Technical documentation with structured sections
- ✅ Multi-turn conversations (use `--incremental`)

**When Compression May Not Help:**
- ❌ Very short prompts (< 50 tokens) - format overhead exceeds savings
- ❌ Already highly compressed text (code, URLs, minimal prose)
- ❌ Single-sentence instructions without context

### Incremental Compression (Factory.ai-Inspired)

For multi-turn conversations or continuously growing documents, use `--incremental` to avoid re-compressing already processed content:

```bash
# First compression (auto-creates conversation-turn1.txt.state.json)
tokuin compress conversation-turn1.txt --incremental --anchor-threshold 1000 --retention-threshold 500

# Subsequent turns — state file is auto-detected
tokuin compress conversation-turn2.txt --incremental
tokuin compress conversation-turn3.txt --incremental
```

**How it works:**
- Creates **anchors** at regular intervals (every `--anchor-threshold` tokens)
- Each anchor stores a compressed summary of content up to that point
- Keeps recent content uncompressed (last `--retention-threshold` tokens)
- Only compresses the delta between last anchor and new content

**Benefits:**
- ✅ Much faster for long conversations (no re-compression)
- ✅ Lower cost per compression operation
- ✅ Maintains context across multiple turns
- ✅ Ideal for agent workflows and chat sessions

**Options:**
- `--anchor-threshold`: Token count before an anchor summary is emitted (default: 1000)
- `--retention-threshold`: Recent tokens (per run) to keep uncompressed (default: 500)
- `--previous <PATH>`: Override the default `<input>.state.json` file if you need multiple independent states

> **Note:** Incremental mode assumes each invocation only includes *new* content (e.g., the next chat turn or newly appended document section). Feed deltas, not the entire conversation, to get real token savings.

**Example output:**
```
Compression Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original:  15834 tokens
Compressed: 7800 tokens
Reduction: 50.7% (8034 tokens saved)

Incremental Mode:
  Anchors: 3
  Anchor tokens: 7300
  Retained tokens: 500

State saved to: conversation.hieratic.json (use --previous for next compression)
```

### Expand Compressed Prompts

```bash
tokuin expand compressed.hieratic --output expanded.txt
```

Or pipe directly to an LLM:

```bash
tokuin expand compressed.hieratic | your-llm-tool
```

### LLM-as-a-Judge Evaluation (requires `--features load-test`)

Evaluate compression quality by comparing outputs from original and compressed prompts using an LLM judge. This provides the most accurate assessment of compression quality by testing whether compressed prompts produce equivalent results.

**Setup:**

1. Get an OpenRouter API key: https://openrouter.ai
2. Set environment variable: `export OPENROUTER_API_KEY="sk-or-..."`

**Basic Usage:**

```bash
# Enable LLM judge evaluation
tokuin compress prompt.txt --quality --llm-judge

# Specify evaluation and judge models (defaults to Claude 3 Opus)
tokuin compress prompt.txt --quality --llm-judge \
  --evaluation-model anthropic/claude-3-sonnet \
  --judge-model anthropic/claude-3-opus

# Use custom API key
tokuin compress prompt.txt --quality --llm-judge \
  --judge-api-key "sk-or-..." \
  --judge-provider openrouter
```

**How It Works:**

1. **Step 1**: Sends original prompt to evaluation model → gets output A
2. **Step 2**: Sends compressed prompt to evaluation model → gets output B
3. **Step 3**: Uses judge model to compare outputs A and B

**Output Example:**

```
LLM Judge Evaluation (Output Comparison):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provider: OpenRouter
Evaluation Model: anthropic/claude-3-opus
Judge Model: anthropic/claude-3-opus

Output Equivalence: 92/100
Instruction Compliance: 95/100
Information Completeness: 88/100
Quality Preservation: 90/100
Overall Fidelity: 91/100 (Excellent)

Justification: The compressed prompt produces outputs that are semantically
equivalent to the original, with all critical instructions followed correctly.

Evaluation Cost: $0.012
```

**Cost Considerations:**

- **Per evaluation**: 3 API calls (2 outputs + 1 judge)
- **Typical cost**: $0.01-0.05 per evaluation (depending on models and prompt/output lengths)
- **For 100 prompt benchmark**: ~$1.00-5.00
- **Optimization tips**:
  - Use cheaper models for evaluation (`anthropic/claude-3-sonnet` or `openai/gpt-3.5-turbo`) while keeping high-quality judge (`anthropic/claude-3-opus`)
  - Default models use Anthropic's Claude 3 Opus for both evaluation and judging (best quality)
  - OpenRouter provides unified pricing and access to 400+ models
  - Cache outputs for identical prompt pairs (future optimization)

**Model Format:**

OpenRouter uses `provider/model` format:
- `anthropic/claude-3-opus` (default, best quality)
- `anthropic/claude-3-sonnet` (good balance)
- `anthropic/claude-3-haiku` (fastest, lower cost)
- `openai/gpt-4`
- `openai/gpt-3.5-turbo`

Simple model names (e.g., `gpt-4`) are automatically converted to OpenRouter format.

**Options:**

- `--llm-judge`: Enable LLM judge evaluation
- `--evaluation-model <MODEL>`: Model to generate outputs (default: same as `--model` or `anthropic/claude-3-opus`)
- `--judge-model <MODEL>`: Model to judge outputs (default: `anthropic/claude-3-opus`)
- `--judge-api-key <KEY>`: API key (or use `OPENROUTER_API_KEY` env var)
- `--judge-provider <PROVIDER>`: Provider (default: `openrouter`, options: `openai`, `anthropic`, `openrouter`)

### Use with Load Testing

Measure cost savings:

```bash
# Test original prompt
echo "$(cat original-prompt.txt)" | tokuin load-test --model gpt-4 --runs 100 --estimate-cost

# Test compressed prompt
tokuin expand compressed.hieratic | tokuin load-test --model gpt-4 --runs 100 --estimate-cost
```

### Format Specification

See [`HIERATIC_FORMAT.md`](HIERATIC_FORMAT.md) for the complete format specification, design rationale, and advanced examples.

## 📋 Command Line Options

### Estimate Command (Default)

```
USAGE:
    tokuin [OPTIONS] [FILE|TEXT]
    tokuin estimate [OPTIONS] [FILE|TEXT]

ARGS:
    <FILE|TEXT>    Input file path (use '-' for stdin or omit for direct text input)

OPTIONS:
    -m, --model <MODEL>        Model to use for tokenization (e.g., gpt-4, gpt-3.5-turbo)
    -c, --compare <MODELS>...   Compare multiple models
    -b, --breakdown             Show token breakdown by role (system/user/assistant)
    -f, --format <FORMAT>       Output format [default: text] 
                                [possible values: text, json, markdown]
    -p, --price                 Show pricing information
        --pricing-file <FILE>    Path to a pricing overrides TOML file (or set TOKUIN_PRICING_FILE)
    --minify                    Strip markdown formatting (requires markdown feature)
    --diff <FILE>               Compare with another prompt file
    -w, --watch                 Watch file for changes and re-run (requires watch feature)
    -h, --help                  Print help
    -V, --version               Print version
```

### Load Test Command (requires `--features load-test`)

```
USAGE:
    tokuin load-test [OPTIONS] --model <MODEL> --runs <RUNS>

OPTIONS:
    -m, --model <MODEL>              Model to use (e.g., gpt-4, openai/gpt-4, claude-2)
        --endpoint <ENDPOINT>         API endpoint URL (optional, uses provider default)
        --api-key <API_KEY>           API key (or use environment variable)
        --openai-api-key <KEY>        OpenAI API key
        --anthropic-api-key <KEY>     Anthropic API key
        --openrouter-api-key <KEY>    OpenRouter API key
    -c, --concurrency <CONCURRENCY>  Number of concurrent requests [default: 10]
    -r, --runs <RUNS>                Total number of requests to make
    -p, --prompt-file <FILE>          Prompt file (or use stdin)
        --think-time <TIME>           Think time between requests (e.g., "250-750ms" or "500ms")
        --retry <RETRY>               Retry count on failure [default: 3]
    -f, --output-format <FORMAT>      Output format [default: text]
                                      [possible values: text, json, csv, prometheus, markdown]
        --dry-run                     Estimate costs without making API calls
        --max-cost <COST>             Maximum cost threshold (stop if exceeded)
    -e, --estimate-cost              Show cost estimation in results
        --pricing-file <FILE>         Path to a pricing overrides TOML file (or set TOKUIN_PRICING_FILE)
    -h, --help                        Print help
```

**Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENROUTER_API_KEY` - OpenRouter API key
- `API_KEY` - Generic API key (provider auto-detected unless `--provider` is set)

## 🔧 Features

### Optional Features

Build with additional features:

```bash
# With markdown support
cargo build --release --features markdown

# With watch mode
cargo build --release --features watch

# With Gemini support
cargo build --release --features gemini

# With load testing capabilities
cargo build --release --features load-test

# With all features
cargo build --release --features all
```

**Feature Details:**
- `markdown`: Markdown output format and minify functionality
- `watch`: File watching for automatic re-analysis
- `gemini`: Google Gemini model support (uses approximation without CMake)
- `load-test`: Load testing with progress bars, metrics, and cost estimation
- `compression`: Prompt compression with Hieratic format
- `compression-embeddings`: Semantic scoring using ONNX embeddings (includes `compression`)
- `all`: Enables all optional features (includes `compression-embeddings`)

## 🛠️ Build from Source (Power Users)

### Prerequisites

- **Rust**: 1.70+ (install via [rustup](https://rustup.rs/))
- **Cargo**: Comes with Rust installation
- **Python 3.10+** (optional, only needed for ONNX model conversion if building from scratch)
- **Git**: For cloning the repository

### Quick Build

**Minimal build (token counting only):**
```bash
git clone https://github.com/nooscraft/tokuin.git
cd tokuin
cargo build --release
```

**Full-featured build (recommended):**
```bash
cargo build --release --features all
```

This includes:
- ✅ Token counting and cost estimation
- ✅ Prompt compression with Hieratic format
- ✅ Semantic scoring with ONNX embeddings
- ✅ Quality metrics
- ✅ Load testing capabilities
- ✅ Markdown support
- ✅ Watch mode
- ✅ Gemini tokenization

### Feature-Specific Builds

**Compression only (no embeddings):**
```bash
cargo build --release --features compression
```
- ✅ Basic compression, quality metrics, structured mode
- ❌ No semantic scoring

**Compression with embeddings (semantic scoring):**
```bash
cargo build --release --features compression-embeddings
```
- ✅ All compression features
- ✅ Semantic and hybrid scoring
- ✅ ONNX embeddings
- ✅ Automatic model download support

**Individual features:**
```bash
# Markdown support
cargo build --release --features markdown

# Watch mode
cargo build --release --features watch

# Gemini tokenization
cargo build --release --features gemini

# Load testing
cargo build --release --features load-test

# Combine features
cargo build --release --features compression,load-test,markdown
```

### Post-Build Setup

**1. Setup Embedding Models (for semantic scoring):**

If you built with `compression-embeddings` or `all`, download the embedding models:

```bash
# Download tokenizer (required for semantic scoring)
./target/release/tokuin setup models

# Also download ONNX model (optional, better quality)
./target/release/tokuin setup models --onnx

# Force re-download if needed
./target/release/tokuin setup models --force
```

**What this does:**
- Downloads the `all-MiniLM-L6-v2` tokenizer from HuggingFace
- Optionally converts and downloads the ONNX model for inference
- Stores models in `~/.cache/tokuin/models/` (or platform-specific cache directory)
- Models are cached and reused across sessions

**2. Verify Installation:**

```bash
# Test basic functionality
./target/release/tokuin --version

# Test compression with semantic scoring
echo "Test prompt" | ./target/release/tokuin compress - --scoring hybrid --quality

# Test load testing (if built with load-test feature)
echo "Hello" | ./target/release/tokuin load-test --model gpt-4 --runs 1 --dry-run
```

### Installation Options

**Option 1: Install to system PATH**
```bash
# Linux/macOS
sudo cp target/release/tokuin /usr/local/bin/

# Or to user directory (no sudo needed)
mkdir -p ~/.local/bin
cp target/release/tokuin ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"  # Add to ~/.bashrc or ~/.zshrc
```

**Option 2: Use directly from build directory**
```bash
# Add alias to your shell config
alias tokuin="$HOME/Projects/tokuin/target/release/tokuin"
```

**Option 3: Use install script (after building)**
```bash
# The install script can also install from a local build
./install.sh --local target/release/tokuin
```

### Troubleshooting

**Build fails with "linker not found":**
- **Linux**: Install `build-essential` or equivalent
- **macOS**: Install Xcode Command Line Tools: `xcode-select --install`
- **Windows**: Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)

**ONNX model download fails:**
- Check internet connection
- Verify Python 3.10+ is installed: `python3 --version`
- Try manual download: `tokuin setup models --force`
- If automatic download fails, use manual conversion scripts: See [`scripts/README.md`](scripts/README.md) for helper scripts
- Models are optional - compression works without them (using heuristic scoring)

**"No such file or directory" when running:**
- Ensure you're running from the correct path: `./target/release/tokuin`
- Check file permissions: `chmod +x target/release/tokuin`
- Verify build succeeded: `ls -lh target/release/tokuin`

**Feature not available after build:**
- Verify you built with the correct feature flag
- Check feature availability: `./target/release/tokuin --help`
- Rebuild with the feature: `cargo build --release --features <feature-name>`

### Development Build

For development with debug symbols and faster compilation:

```bash
# Debug build (faster compilation, larger binary)
cargo build

# Run tests
cargo test

# Run with output
cargo test -- --nocapture

# Format code
cargo fmt

# Lint code
cargo clippy -- -D warnings
```

### Feature Matrix

| Feature | Flag | Build Command | Post-Build Setup |
|---------|------|---------------|------------------|
| Basic token counting | (default) | `cargo build --release` | None |
| Compression | `compression` | `--features compression` | None |
| Semantic scoring | `compression-embeddings` | `--features compression-embeddings` | `tokuin setup models` |
| Load testing | `load-test` | `--features load-test` | API keys (env vars) |
| Markdown | `markdown` | `--features markdown` | None |
| Watch mode | `watch` | `--features watch` | None |
| Gemini | `gemini` | `--features gemini` | None |
| Everything | `all` | `--features all` | `tokuin setup models` |

### Performance Tips

- **Release builds**: Always use `--release` for production (10-100x faster)
- **Parallel compilation**: Cargo uses all CPU cores by default
- **Incremental builds**: Cargo caches intermediate artifacts
- **Feature selection**: Only build features you need to reduce compile time

### Next Steps

After building and setting up:
1. Read the [Usage](#-usage) section for common commands
2. Try [Prompt Compression](#️-prompt-compression-with-hieratic-format-requires-features-compression) with `--quality` flag
3. Explore [Load Testing](#load-testing-requires---features-load-test) capabilities
4. Check [Command Line Options](#-command-line-options) for advanced usage

## 🎯 Supported Models

### OpenAI
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- `gpt-3.5-turbo-16k`

### Google Gemini (requires `--features gemini`)
- `gemini-pro`
- `gemini-2.5-pro`
- `gemini-2.5-flash`

> **Note**: Gemini tokenizer requires the SentencePiece model file. See [ADDING_MODELS_GUIDE.md](ADDING_MODELS_GUIDE.md) for details.

### OpenRouter (requires `--features load-test`)

OpenRouter provides access to 400+ models from various providers through a unified API. Use the `provider/model` format:

- `openai/gpt-4`
- `openai/gpt-3.5-turbo`
- `anthropic/claude-3-haiku`
- `meta-llama/llama-2-70b-chat`
- `google/gemini-pro`
- And hundreds more — see the [OpenRouter catalog](https://openrouter.ai/models).

### Anthropic (requires `--features load-test`)

- `claude-3-opus`
- `claude-3-sonnet`
- `claude-3-haiku`

Call Anthropic directly with `--provider anthropic` plus an API key.

### Generic Endpoint (requires `--features load-test`)

Use `--provider generic` and supply `--endpoint` (and optional `--api-key` / extra headers via `--header` soon) to load test your own gateway or proxy. Responses should return a top-level `content`, `response`, `result`, `output`, or `choices[*].message.content` field.

### Planned Providers
- Mistral AI
- Cohere
- AI21 Labs
- Meta LLaMA

See [PROVIDERS_PLAN.md](PROVIDERS_PLAN.md) for the full provider roadmap.

## 🏗️ Architecture

The project follows a modular architecture:

- **tokenizers/**: Tokenizer implementations for different providers
- **models/**: Model registry and pricing configuration
- **parsers/**: Input format parsers (text, JSON)
- **output/**: Output formatters (text, JSON, Markdown)
- **http/**: HTTP client layer for load testing (requires `load-test` feature)
  - **providers/**: Provider-specific API clients (OpenAI, OpenRouter, Anthropic)
- **simulator/**: Load testing simulator with concurrency control and metrics
- **cli.rs**: Command-line interface
- **error.rs**: Error type definitions

## 🧪 Testing

Run the test suite:

```bash
cargo test
```

Run with output:

```bash
cargo test -- --nocapture
```

### Minimum Supported Rust Version (MSRV)

The project currently targets Rust 1.70+ as indicated by the badge. However, due to dependency compatibility issues (Cargo.lock version 4 and newer dependency requirements), we do not currently test against a specific MSRV in CI. The project is tested on stable and beta Rust versions.

If you need to support an older Rust version, you may need to pin dependencies to compatible versions. See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

> Vibe coding welcome — if you collaborate with an AI pair programmer, skim `AGENTS.md` for a quick project brief and mention the session in your PR so everyone can follow the flow.

### 🎉 Contributors

Thanks to everyone who has dived in—bugs, docs, and feature requests all help shape Tokuin. Want to join them?

[![Contributors](https://contrib.rocks/image?repo=nooscraft/tokuin)](https://github.com/nooscraft/tokuin/graphs/contributors)

If you ship a noteworthy feature, open a PR and add yourself to `CONTRIBUTORS.md` (or include a small note in your PR and we’ll update it). Shoutouts go out in release notes.

### Adding New Models

See [ADDING_MODELS_GUIDE.md](ADDING_MODELS_GUIDE.md) for instructions on adding support for new models.

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🙏 Acknowledgments

- Built with [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) for OpenAI tokenization
- Inspired by the need for accurate token estimation in LLM development workflows

