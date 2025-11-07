# Contributing to Tokuin

Thank you for your interest in contributing! This guide will help you get started.

> Quick orientation? Skim `AGENTS.md` for an agent-friendly project brief before you dive in.

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Git
- Basic familiarity with Rust (for code contributions)

### Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tokuin.git
   cd tokuin
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/nooscraft/tokuin.git
   ```
4. **Install dependencies**:
   ```bash
   cargo build
   ```
5. **Run tests**:
   ```bash
   cargo test
   ```

### AI Agents & Vibe Coding

- We love seeing creativity through "vibe coding" sessions. If you are pairing with AI tools (Cursor, Copilot, Claude, etc.), please capture the intent of the session in your PR description so others can follow the flow.
- Before you start, skim `AGENTS.md` for a condensed project briefing tailored to AI editors and pair-programming assistants. It highlights coding conventions, feature flags, and project history to keep vibes aligned with the roadmap.
- When using an AI editor, prefer small commits with clear messages so reviewers can trace automated changes. Mention which agent(s) were involved and any manual tweaks you applied.
- If your session produces follow-up ideas or TODOs, open issues or leave PR comments so the next contributor can pick up the thread.

## 📋 How to Contribute

### Reporting Bugs

- Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template
- Include steps to reproduce
- Provide sample prompts/text that trigger the issue
- Include your Rust version and OS

### Suggesting Features

- Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template
- Explain the use case
- Describe expected behavior
- Consider backwards compatibility

### Adding New Models

- Use the [Model Support](.github/ISSUE_TEMPLATE/model_support.md) template
- Follow the [ADDING_MODELS.md](docs/ADDING_MODELS.md) guide
- Include tokenizer implementation with tests
- Update model registry and pricing config

### Code Contributions

1. **Find an issue** to work on (or create one first)
   - Look for `good first issue` label for beginners
   - Comment on the issue to claim it
2. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```
3. **Make your changes**:
   - Follow Rust style guidelines
   - Write tests for new functionality
   - Update documentation
4. **Test your changes**:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```
5. **Commit your changes**:
   ```bash
   git commit -m "feat: add support for model X"
   ```
   Use conventional commit format:
   - `feat:` New features
   - `fix:` Bug fixes
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `chore:` Maintenance tasks
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**:
   - Use the PR template
   - Link to related issues
   - Describe your changes clearly
   - Wait for review and address feedback

## 🧪 Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run integration tests
cargo test --test integration
```

### Writing Tests

- **Unit tests**: Place in the same file with `#[cfg(test)]`
- **Integration tests**: Place in `tests/` directory
- **Test fixtures**: Use `tests/fixtures/` for sample data

Example:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let tokenizer = OpenAITokenizer::new("gpt-4");
        let count = tokenizer.count_tokens("Hello, world!").unwrap();
        assert_eq!(count, 4);
    }
}
```

## 📝 Code Style

### Rust Best Practices

This project follows comprehensive Rust best practices. **Please read the Rust guidance in `AGENTS.md` and the shared repository rules (`.cursor/rules/clean-code.mdc`) for detailed expectations.**

### Quick Reference

- Follow official [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/)
- **Naming**: `snake_case` for functions/variables, `PascalCase` for types
- **Error Handling**: Use `Result<T, E>` with `thiserror` for all fallible operations
- **Use `?` operator** instead of `.unwrap()` or `.expect()` in library code
- **Never panic** in library code (except in test harnesses)
- **Immutability First**: Prefer `let` over `let mut`, use functional patterns
- **Zero-Copy**: Use references (`&str`, `&[T]`) instead of owned values when possible
- Run `cargo fmt` before committing
- Fix all `clippy` warnings (run `cargo clippy -- -D warnings`)
- All public items must have `///` doc comments with examples

### Documentation

- All public functions/types must have doc comments (`///`)
- Include usage examples in doc comments
- Update relevant documentation when making changes

Example:
```rust
/// Counts tokens in the given text.
///
/// # Arguments
///
/// * `text` - The text to tokenize
///
/// # Returns
///
/// The number of tokens in the text
///
/// # Example
///
/// ```rust
/// let tokenizer = OpenAITokenizer::new("gpt-4");
/// let count = tokenizer.count_tokens("Hello, world!")?;
/// ```
pub fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
    // ...
}
```

## 🏗️ Project Structure

```
src/
├── tokenizers/     # Tokenizer implementations
├── models/         # Model registry and pricing
├── parsers/        # Input format parsers
├── analysis/      # Token analysis and optimization
├── output/         # Output formatters
└── utils/          # Utility functions
```

## 🐛 Debugging

### Common Issues

**Build fails:**
- Ensure Rust version is 1.70+
- Run `cargo clean` and rebuild
- Check `Cargo.toml` for dependency issues

**Tests fail:**
- Check if test fixtures exist
- Verify tokenizer implementations are correct
- Run tests individually to isolate issues

**Clippy errors:**
- Read the error message carefully
- Apply suggested fixes
- For complex cases, ask in GitHub Discussions

## 💬 Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code review and technical questions

## 📜 Code of Conduct

We follow the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Be respectful, inclusive, and constructive in all interactions.

## ✅ Checklist Before Submitting PR

- [ ] Code follows style guidelines (`cargo fmt`, `cargo clippy`)
- [ ] All tests pass (`cargo test`)
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Commit messages follow conventional format
- [ ] PR description is clear and links to related issues

## 🎉 Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing! 🎊

