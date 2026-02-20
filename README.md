# 🧮 Tokuin

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/nooscraft/tokuin)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![CI](https://github.com/nooscraft/tokuin/workflows/CI/badge.svg)](https://github.com/nooscraft/tokuin/actions)

A fast, CLI tool to **estimate token usage**, **control API costs**, and **compress prompts** for LLM providers. Built in Rust for performance, portability, and safety.

## ✨ What's in v0.2.0

- **Prompt Compression** — Reduce prompt tokens by 70-90% using the [Hieratic format](#️-prompt-compression)
- **Quality Metrics** — Measure compression fidelity (semantic similarity, critical instruction preservation, structural integrity)
- **LLM-as-a-Judge** — Compare outputs from original vs compressed prompts using an LLM judge
- **Incremental Compression** — Compress long conversations turn-by-turn without re-processing history
- **Structured Mode** — Compression-aware handling of JSON, code blocks, HTML tables, and technical docs
- Everything from v0.1.x: token counting, cost estimation, multi-model comparison, load testing

---

## 🚀 Installation

### Quick Install (macOS & Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/nooscraft/tokuin/main/install.sh | bash
```

Detects your platform, downloads the latest release, verifies its checksum, and installs `tokuin` to `/usr/local/bin` (or `~/.local/bin` if root access is unavailable).

### What's included per platform

Release binaries are built with `--features all`. Embedding-based semantic scoring (`compression-embeddings`) requires the ONNX Runtime, which only has compatible prebuilt binaries for some platforms:

| Platform | Token counting | Compression | Embedding scoring | LLM judge |
|---|---|---|---|---|
| Linux x86_64 | ✅ | ✅ | ✅ | ✅ |
| macOS Apple Silicon (aarch64) | ✅ | ✅ | ✅ | ✅ |
| macOS Intel (x86_64) | ✅ | ✅ | ❌ heuristic only | ✅ |
| Linux aarch64 | ✅ | ✅ | ❌ heuristic only | ✅ |
| Windows x86_64 | ✅ | ✅ | ❌ heuristic only | ✅ |

Platforms without embedding scoring still have full compression — quality metrics use heuristic scoring instead of embedding-based semantic similarity.

For full embedding support on any platform, build from source with `--features all,compression-embeddings`.

### Quick Install (Windows PowerShell)

```powershell
irm https://raw.githubusercontent.com/nooscraft/tokuin/main/install.ps1 | iex
```

Binary is placed in `%LOCALAPPDATA%\Programs\tokuin`. To customize: download the script first and invoke `.\install.ps1 -InstallDir "C:\Tools"`.

### From Releases

Release archives for all platforms are at [GitHub Releases](https://github.com/nooscraft/tokuin/releases). Download the archive for your OS/architecture, verify against `checksums.txt`, and place `tokuin` on your `PATH`.

### From Source

**Minimal build (token counting only):**
```bash
git clone https://github.com/nooscraft/tokuin.git
cd tokuin
cargo build --release
```

**Build with all features:**
```bash
cargo build --release --features all
```

**Build with embedding support (semantic scoring):**
```bash
cargo build --release --features all,compression-embeddings
# Then set up models:
./target/release/tokuin setup models
```

---

## 📖 Usage

### Token Counting

```bash
echo "Hello, world!" | tokuin --model gpt-4
tokuin prompt.txt --model gpt-4 --price
```

### Multi-Model Comparison

```bash
echo "Hello, world!" | tokuin --compare gpt-4 gpt-3.5-turbo --price
```

### JSON Output / Breakdown

```bash
echo "Hello, world!" | tokuin --model gpt-4 --format json
echo '[{"role":"system","content":"You are helpful"},{"role":"user","content":"Hi!"}]' | \
  tokuin --model gpt-4 --breakdown --price
```

### Diff & Watch

```bash
tokuin prompt.txt --model gpt-4 --diff prompt-v2.txt --price
tokuin prompt.txt --model gpt-4 --watch
```

---

## 🗜️ Prompt Compression

Tokuin includes a prompt compression system using the **Hieratic format** — a structured, LLM-parseable representation that reduces token usage by 70-90% while preserving semantic meaning.

### Why Hieratic?

Named after ancient Egypt's compressed cursive script (a practical simplification of hieroglyphics), Hieratic represents the same information in far fewer tokens. Any modern LLM reads it directly without decoding.

### Quick Start

```bash
# Compress a prompt (medium level by default)
tokuin compress my-prompt.txt

# With quality metrics
tokuin compress my-prompt.txt --quality

# Aggressive compression
tokuin compress my-prompt.txt --level aggressive --quality

# Structured mode (JSON, code blocks, HTML tables)
tokuin compress api-spec.txt --structured --level medium
```

### Compression Levels

| Level | Token Reduction | Quality Trade-off |
|---|---|---|
| `light` | 30–50% | Minimal — safe for most prompts |
| `medium` | 50–70% | Balanced — good default |
| `aggressive` | 70–90% | Maximum — verify with `--quality` |

### Hieratic Format Example

**Original (850 tokens):**
```
You are an expert programmer with 10 years of experience in building
distributed systems, microservices, and cloud-native applications...
[full verbose role description]

Example 1: Authentication Bug Fix
Given a service that was experiencing session token bypass attacks...
[detailed example spanning 200 tokens]
```

**Hieratic (285 tokens — 66% reduction):**
```
@HIERATIC v1.0

@ROLE[inline]
"Expert engineer: 10y distributed systems, microservices, cloud-native"

@EXAMPLES[inline]
1. Auth bug: session bypass → HMAC signing → 94% bot reduction
2. DB perf: 2.3s queries → pooling+cache → 0.1s, 10x capacity

@TASK
Analyze code and provide recommendations

@FOCUS: performance, security, maintainability
@STYLE: concise, actionable
```

LLMs read Hieratic natively — no expansion step needed when sending to an API.

### Quality Metrics

Use `--quality` to measure compression fidelity:

```bash
tokuin compress prompt.txt --quality
```

```
Quality Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Score: 82.3% (Good)
  ├─ Semantic Similarity:      85.1%
  ├─ Critical Instructions:    3/3 preserved (100.0%)
  ├─ Information Retention:    78.2%
  └─ Structural Integrity:     100.0%

✅ Quality is acceptable (>= 70%)
```

**Scoring modes:**

| Mode | Speed | Accuracy | Requires |
|---|---|---|---|
| `heuristic` | Fast | Keyword-based | Nothing (default) |
| `semantic` | Slower | Embedding-based | `compression-embeddings` |
| `hybrid` | Moderate | Best of both | `compression-embeddings` |

```bash
# Use semantic scoring (on supported platforms)
tokuin compress prompt.txt --scoring semantic --quality

# Heuristic scoring (always available)
tokuin compress prompt.txt --scoring heuristic --quality
```

### Structured Document Mode

For prompts with JSON, HTML tables, BNF grammars, or code blocks:

```bash
tokuin compress technical-spec.txt --structured --level medium
```

Structured mode:
- Preserves JSON document structure
- Keeps HTML tables intact
- Detects and consolidates repetitive instruction patterns
- Segments by logical sections (definitions, examples, format specs)
- Applies structure-aware importance scoring

Use structured mode for: extraction prompts, API documentation, data processing specs.  
Use default mode for: conversational prompts, natural language instructions, role descriptions.

### Incremental Compression

For multi-turn conversations or continuously growing documents:

```bash
# First turn — creates conversation-turn1.txt.state.json
tokuin compress conversation-turn1.txt --incremental

# Subsequent turns — state file is auto-detected
tokuin compress conversation-turn2.txt --incremental
tokuin compress conversation-turn3.txt --incremental
```

Only the delta since the last anchor is compressed — no re-processing of history.

**Options:**
- `--anchor-threshold <N>`: Tokens before creating an anchor summary (default: 1000)
- `--retention-threshold <N>`: Recent tokens kept uncompressed (default: 500)
- `--previous <PATH>`: Override default state file path

```
Incremental Mode:
  Anchors: 3
  Anchor tokens: 7300
  Retained tokens: 500
```

### LLM-as-a-Judge Evaluation

Test whether compressed prompts produce equivalent outputs to the original — the most reliable quality signal:

```bash
export OPENROUTER_API_KEY="sk-or-..."

tokuin compress prompt.txt --quality --llm-judge
```

**How it works:**
1. Sends the original prompt to the evaluation model → output A
2. Sends the compressed prompt → output B
3. Judge model scores A vs B on equivalence, instruction compliance, information completeness

```
LLM Judge Evaluation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output Equivalence:       92/100
Instruction Compliance:   95/100
Information Completeness: 88/100
Quality Preservation:     90/100
Overall Fidelity:         91/100 (Excellent)
Evaluation Cost: $0.012
```

**Options:**
- `--llm-judge`: Enable LLM judge (requires `load-test` feature + API key)
- `--evaluation-model <MODEL>`: Model for generating outputs (default: `anthropic/claude-3-opus`)
- `--judge-model <MODEL>`: Model for scoring (default: `anthropic/claude-3-opus`)
- `--judge-api-key <KEY>`: API key (or use `OPENROUTER_API_KEY` env var)
- `--judge-provider <PROVIDER>`: `openrouter` (default), `openai`, `anthropic`

**Cost per evaluation:** ~$0.01–0.05 (2 output calls + 1 judge call). Use cheaper models like `anthropic/claude-3-haiku` for evaluation and keep the high-quality model for judging.

### Expand Compressed Prompts

```bash
tokuin expand compressed.hieratic --output expanded.txt

# Pipe directly to an LLM tool
tokuin expand compressed.hieratic | your-llm-tool
```

### Limitations and Known Drawbacks

**Where compression works best:**
- ✅ Prompts ≥ 100 tokens (70–90% reduction)
- ✅ Prompts with repetitive role descriptions, examples, or constraints
- ✅ Technical docs with clear sections (JSON extraction, API specs)
- ✅ Multi-turn conversations (use `--incremental`)

**Where it may not help or hurt:**
- ❌ Very short prompts (< 50 tokens) — Hieratic header overhead exceeds savings
- ❌ Already dense text (URLs, code, minimal prose) — little redundancy to remove
- ❌ Single-sentence instructions — format overhead not worth it
- ❌ Prompts where exact wording is critical (legal, medical) — always verify with `--quality`

**Scoring limitations (without `compression-embeddings`):**
- Heuristic scoring uses keyword matching and position weighting — it can miss semantic drift
- Always use `--quality --scoring semantic` (or `--llm-judge`) on prompts where accuracy matters
- Platforms without embedding support: Linux aarch64, macOS Intel, Windows

**General caveats:**
- Compression quality varies by prompt type and content; results are not guaranteed
- `aggressive` level may lose nuance; validate outputs on your specific use case
- Hieratic is a Tokuin-specific format — LLMs understand it, but it is not a standard
- The compressed output is not human-readable without familiarity with the format

---

## 🧪 Load Testing

Run concurrent load tests against LLM APIs:

```bash
# Basic load test
export OPENAI_API_KEY="sk-openai-..."
echo "What is 2+2?" | tokuin load-test \
  --model gpt-4 \
  --runs 100 \
  --concurrency 10

# With OpenRouter (400+ models)
export OPENROUTER_API_KEY="sk-or-..."
echo "Hello!" | tokuin load-test \
  --model openai/gpt-4 \
  --runs 50 \
  --concurrency 5 \
  --provider openrouter

# Dry run — cost estimate only, no API calls
echo "Test prompt" | tokuin load-test \
  --model gpt-4 \
  --runs 1000 \
  --concurrency 50 \
  --dry-run \
  --estimate-cost

# With think time and retries
tokuin load-test \
  --model gpt-4 \
  --runs 200 \
  --concurrency 20 \
  --think-time "250-750ms" \
  --retry 3 \
  --prompt-file prompts.txt
```

**Output:**
```
=== Load Test Results ===
Total Requests: 100
Successful: 98 (98.0%)
Failed: 2 (2.0%)

Latency (ms):
  Average: 1234.56
  p50:     1200
  p95:     1850

Cost Estimation:
  Input tokens:   5000
  Output tokens: 12000
  Total cost:    $0.870000
```

**Environment Variables:**
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `API_KEY` — generic (provider auto-detected unless `--provider` is set)

---

## 📋 Command Reference

### Token Estimate (default)

```
tokuin [OPTIONS] [FILE|TEXT]
tokuin estimate [OPTIONS] [FILE|TEXT]

Options:
  -m, --model <MODEL>          Model for tokenization (e.g. gpt-4)
  -c, --compare <MODELS>...    Compare across multiple models
  -b, --breakdown              Show breakdown by role
  -f, --format <FORMAT>        text | json | markdown  [default: text]
  -p, --price                  Show cost estimate
      --pricing-file <FILE>    Custom pricing TOML (or TOKUIN_PRICING_FILE)
      --minify                 Strip markdown (requires markdown feature)
      --diff <FILE>            Compare with another file
  -w, --watch                  Re-run on file change (requires watch feature)
```

### Compress

```
tokuin compress <FILE> [OPTIONS]

Options:
  -o, --output <FILE>                   Output file [default: <input>.hieratic]
  -l, --level <LEVEL>                   light | medium | aggressive  [default: medium]
      --structured                       Enable structured document mode
      --inline                           Force inline mode
      --incremental                      Incremental compression (delta only)
      --previous <FILE>                  Custom state file path
      --anchor-threshold <N>             Anchor interval in tokens [default: 1000]
      --retention-threshold <N>          Recent tokens kept uncompressed [default: 500]
  -m, --model <MODEL>                   Tokenizer model [default: gpt-4]
  -f, --format <FORMAT>                 hieratic | expanded | json  [default: hieratic]
      --scoring <MODE>                   heuristic | semantic | hybrid  [default: heuristic]
      --quality                          Show quality metrics
      --llm-judge                        LLM judge evaluation (requires load-test + API key)
      --evaluation-model <MODEL>         Model for LLM judge output generation
      --judge-model <MODEL>              Model for judging [default: anthropic/claude-3-opus]
      --judge-api-key <KEY>              API key for judge
      --judge-provider <PROVIDER>        openrouter | openai | anthropic  [default: openrouter]
      --pricing-file <FILE>              Custom pricing TOML
```

### Load Test

```
tokuin load-test [OPTIONS] --model <MODEL> --runs <RUNS>

Options:
  -m, --model <MODEL>              Model name
      --endpoint <URL>             API endpoint (optional)
      --api-key <KEY>              Generic API key
      --openai-api-key <KEY>       OpenAI API key
      --anthropic-api-key <KEY>    Anthropic API key
      --openrouter-api-key <KEY>   OpenRouter API key
  -c, --concurrency <N>            Concurrent requests [default: 10]
  -r, --runs <N>                   Total requests
  -p, --prompt-file <FILE>         Prompt file (or stdin)
      --think-time <TIME>          Think time e.g. "250-750ms"
      --retry <N>                  Retries on failure [default: 3]
  -f, --output-format <FORMAT>     text | json | csv | prometheus | markdown
      --dry-run                    Estimate cost, no API calls
      --estimate-cost              Show cost in results
      --pricing-file <FILE>        Custom pricing TOML
```

---

## 🔧 Features & Build Flags

| Feature | Flag | What it adds |
|---|---|---|
| Token counting | *(default)* | Core estimation, cost, comparison |
| Compression | `compression` | Hieratic compress/expand, heuristic quality metrics |
| Embedding scoring | `compression-embeddings` | Semantic + hybrid scoring via ONNX |
| Load testing | `load-test` | Concurrent API testing, LLM judge |
| Markdown | `markdown` | Markdown output format, `--minify` |
| Watch mode | `watch` | `--watch` file change detection |
| Gemini | `gemini` | Google Gemini tokenization |
| All (release) | `all` | Everything above except `compression-embeddings`* |

\* `compression-embeddings` is excluded from `all` because ONNX Runtime prebuilt binaries are not available for all cross-compiled release targets. Add it explicitly with `--features all,compression-embeddings` on a native host.

```bash
# Standard release build
cargo build --release --features all

# With embedding support (native host only)
cargo build --release --features all,compression-embeddings

# Minimal compression (no embeddings)
cargo build --release --features compression

# Load testing only
cargo build --release --features load-test
```

### Post-Build: Set Up Embedding Models

Only needed when building with `compression-embeddings`:

```bash
./target/release/tokuin setup models        # download tokenizer
./target/release/tokuin setup models --onnx # also download ONNX model
./target/release/tokuin setup models --force # re-download
```

Models are stored in `~/.cache/tokuin/models/` and reused across sessions.

---

## 🎯 Supported Models

### OpenAI
- `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`

### Google Gemini (requires `gemini` feature)
- `gemini-pro`, `gemini-2.5-pro`, `gemini-2.5-flash`

### OpenRouter (requires `load-test` feature)
Unified access to 400+ models via `provider/model` format:
- `anthropic/claude-3-opus`, `openai/gpt-4`, `meta-llama/llama-2-70b-chat`, `google/gemini-pro`, and hundreds more

### Anthropic (requires `load-test` feature)
- `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

### Generic Endpoint
Use `--provider generic` with `--endpoint` to target any OpenAI-compatible API.

### Planned
Mistral, Cohere, AI21 Labs, Meta LLaMA. See [PROVIDERS_PLAN.md](PROVIDERS_PLAN.md).

---

## 🏗️ Architecture

```
src/
├── cli.rs                  # Argument parsing and command dispatch
├── error.rs                # Central error type (thiserror)
├── tokenizers/             # OpenAI, Gemini tokenizer implementations
├── models/                 # Model registry and pricing tables
├── parsers/                # Input parsing (plain text, JSON chat)
├── output/                 # Formatters (text, JSON, Markdown)
├── http/                   # HTTP client layer (load-test feature)
│   └── providers/          # OpenAI, OpenRouter, Anthropic, stubs
├── simulator/              # Load test engine, concurrency, metrics
├── compression/            # Hieratic compression pipeline
│   ├── compressor.rs       # Orchestration and compression levels
│   ├── parser.rs           # Hieratic format parser
│   ├── hieratic_encoder.rs # Encode to Hieratic
│   ├── hieratic_decoder.rs # Expand Hieratic back to text
│   ├── quality.rs          # Quality metrics
│   ├── embeddings.rs       # ONNX embedding provider
│   ├── llm_judge.rs        # LLM-as-a-judge evaluation
│   ├── pattern_extractor.rs
│   └── context_library.rs
└── utils/                  # Shared helpers
```

---

## 🧪 Testing

```bash
cargo test
cargo test --features all
cargo test -- --nocapture  # with output
cargo fmt
cargo clippy -- -D warnings
```

### MSRV

Targets Rust 1.70+. CI runs on stable and beta. Due to dependency constraints, a specific MSRV is not enforced. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

> Vibe coding welcome — if you collaborate with an AI pair programmer, skim `AGENTS.md` for a quick project brief and mention the session in your PR.

[![Contributors](https://contrib.rocks/image?repo=nooscraft/tokuin)](https://github.com/nooscraft/tokuin/graphs/contributors)

---

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🙏 Acknowledgments

- Built with [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) for OpenAI tokenization
- Inspired by the need for accurate token estimation and cost control in LLM development
- Hieratic format named after the ancient Egyptian script that compressed hieroglyphics into practical everyday writing
