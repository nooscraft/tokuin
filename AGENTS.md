# AGENTS GUIDE

This document gives AI coding agents, pair-programming assistants, and vibe coders the context they need to work effectively on **Tokuin**.

## Project Snapshot

- **Name:** Tokuin
- **Type:** Rust CLI for LLM token estimation and load testing
- **Repo:** https://github.com/nooscraft/tokuin
- **Primary Commands:** `estimate` (default) and `load-test`
- **License:** MIT OR Apache-2.0

## Key References

- General overview and usage: `README.md`
- Contribution guidelines (AI-friendly): `CONTRIBUTING.md`
- Coding standards: `.cursor/rules/clean-code.mdc` and section below
- Distribution plans: `docs/DISTRIBUTION_PLAN.md`
- Pre-push expectations: `docs/PRE_PUSH_CHECKLIST.md`
- Load testing design notes: `llm_load_simulator.md`

## Architecture Overview

```
src/
├── cli.rs            # Argument parsing, command dispatch
├── error.rs          # Central error enum using thiserror
├── http/             # HTTP client layer (load-test feature)
│   ├── client.rs     # LlmClient trait, runtime dispatcher
│   └── providers/    # Provider-specific clients (OpenAI, OpenRouter, stubs)
├── models/           # Model registry and pricing tables
├── output/           # Formatters (text, json, markdown, etc.)
├── parsers/          # Input parsing utilities
├── simulator/        # Load test engine and config handling
├── tokenizers/       # Tokenization implementations (OpenAI, Gemini)
└── utils/            # Shared helpers (markdown, diff, file IO)
```

## Features & Flags

- `markdown`: Markdown output and minify support
- `watch`: File watching (`notify` crate)
- `gemini`: Gemini tokenization (approximation without SentencePiece)
- `gemini-sentencepiece`: Exact Gemini tokenizer requiring CMake
- `load-test`: Async HTTP clients, simulator, progress bars
- `all`: Convenience flag enabling every optional feature

Build examples:

```bash
cargo build --release
cargo build --release --features load-test
cargo build --release --features all
```

## CLI Summary

### Estimate (default)
- Supports positional `FILE|TEXT`, or stdin
- Flags: `--model`, `--compare`, `--format`, `--price`, `--breakdown`, `--minify`, `--diff`, `--watch`

### Load Test (feature `load-test`)
- Subcommand: `tokuin load-test`
- Key options: `--model`, `--runs`, `--concurrency`, `--prompt-file`, `--think-time`, `--retry`, `--estimate-cost`, `--dry-run`, `--output-format`
- Environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `API_KEY`

## Coding Standards

- Follow Rust naming conventions: snake_case for functions/variables, PascalCase for types
- Use `Result<T, AppError>` for fallible operations; prefer `?` instead of `.unwrap()` in non-test code
- Add doc comments (`///`) for all public APIs
- Keep modules shallow and cohesive; avoid deeply nested paths
- Run `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test` before submitting
- Document feature-specific behaviours (e.g., load-test) in README or inline docs

## Testing & Tooling

- Unit tests in modules via `#[cfg(test)]`
- Integration tests planned but not yet implemented for load-test
- CI (`.github/workflows/ci.yml`) runs fmt, clippy, tests on stable/beta and multiple OS targets
- MSRV is currently advertised as 1.70+, but CI does not enforce it due to dependency constraints

## Load Test Stack

- Async runtime: `tokio`
- HTTP client: `reqwest`
- Progress bars: `indicatif`
- Metrics: atomic counters plus `hdrhistogram`
- Config parsing: `serde_yaml`

## Working Style & Vibe Coding

- Vibe coders are welcome—capture the intent of your session in commit/PR descriptions
- Reference this file and `CONTRIBUTING.md` before you begin an AI-assisted session
- Keep commits small, annotate AI-generated chunks, and flag TODOs for follow-up work
- Coordinate with maintainers via GitHub Discussions or Issues if you start experimental branches

## Pending Enhancements (High-Level)

- Add README documentation for advanced load-test usage (partially done; monitor for gaps)
- Add integration tests covering the `load-test` command
- Expand provider support (Anthropic direct, Mistral, Cohere, AI21, LLaMA)
- Refine CI to exercise feature matrices more explicitly

## Build & Run Checklist

1. `cargo fmt`
2. `cargo clippy -- -D warnings`
3. `cargo test`
4. (Optional) `cargo test --features all`
5. Update `docs/PRE_PUSH_CHECKLIST.md` if scope warrants it

## Helpful Artifacts

- Pricing & model data live in `models/` and config files under `data/` (if present)
- Distribution plans captured in `docs/DISTRIBUTION_PLAN.md`
- Load-test plan and background: `llm_load_simulator.md`
- Use `.env` or shell exports for API keys during load testing

## Contact & Support

- Open a GitHub Discussion for questions or design proposals
- Label issues with `good first issue` for newcomers and vibe coders
- Mention maintainers in PRs when AI agents submit automated changes

Happy building! Keep the vibes high and the code clean. ✨

