# 🧮 Tokuin

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/nooscraft/tokuin)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Crates.io](https://img.shields.io/crates/v/tokuin.svg)](https://crates.io/crates/tokuin)
[![Documentation](https://docs.rs/tokuin/badge.svg)](https://docs.rs/tokuin)
[![CI](https://github.com/nooscraft/tokuin/workflows/CI/badge.svg)](https://github.com/nooscraft/tokuin/actions)

A fast, CLI-based tool to estimate **token usage** and **API cost** for prompts targeting various LLM providers (OpenAI, Claude, Mistral, etc.). Built in Rust for performance, portability, and safety.

## ✨ Features

- **Token Count Estimation**: Analyze prompts and count tokens for selected models (e.g., `gpt-4`, `gpt-3.5-turbo`)
- **Cost Estimation**: Calculate API costs based on token pricing per model
- **Multi-Model Comparison**: Compare token usage and cost across multiple providers
- **Role-Based Breakdown**: Show token count by system/user/assistant role messages
- **Multiple Input Formats**: Support plain text and JSON chat formats
- **Flexible Output**: Human-readable text or JSON output for scripting
- **Load Testing** (requires `--features load-test`): Run concurrent load tests against LLM APIs with real-time metrics, progress bars, and cost estimation

## 🚀 Installation

### From Source

```bash
git clone https://github.com/nooscraft/tokuin.git
cd tokuin
cargo build --release
```

The binary will be available at `target/release/tokuin`.

### Using Cargo

```bash
cargo install tokuin
```

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

```bash
# Basic load test with OpenAI
export OPENAI_API_KEY="sk-..."
echo "What is 2+2?" | tokuin load-test --model gpt-4 --runs 100 --concurrency 10

# With OpenRouter (access to 400+ models)
export OPENROUTER_API_KEY="sk-or-..."
echo "Hello!" | tokuin load-test --model openai/gpt-4 --runs 50 --concurrency 5

# With think time between requests
tokuin load-test --model gpt-4 --runs 200 --concurrency 20 --think-time "250-750ms" --prompt-file prompts.txt

# Dry run to estimate costs without making API calls
echo "Test prompt" | tokuin load-test --model gpt-4 --runs 1000 --concurrency 50 --dry-run --estimate-cost

# With retry and cost estimation
tokuin load-test \
  --model gpt-4 \
  --runs 100 \
  --concurrency 10 \
  --retry 3 \
  --estimate-cost \
  --output-format json
```

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
    -h, --help                        Print help
```

**Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENROUTER_API_KEY` - OpenRouter API key
- `API_KEY` - Generic API key (provider auto-detected)

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
- `all`: Enables all optional features

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
- `anthropic/claude-2`
- `anthropic/claude-3-opus`
- `meta-llama/llama-2-70b-chat`
- `google/gemini-pro`
- And 400+ more models...

See [OpenRouter Models](https://openrouter.ai/models) for the full list.

### Planned Providers
- Anthropic Claude (direct API support)
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

