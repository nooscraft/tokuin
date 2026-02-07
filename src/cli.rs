use crate::error::AppError;
use crate::models::ModelRegistry;
#[cfg(feature = "markdown")]
use crate::output::MarkdownFormatter;
use crate::output::{Formatter, JsonFormatter, TextFormatter, TokenBreakdown, TokenResult};
use crate::parsers::{JsonParser, Parser as InputParser, TextParser};
use crate::tokenizers::Tokenizer;
#[cfg(feature = "markdown")]
use crate::utils::markdown;
/// CLI argument parsing and command execution.
use clap::{Parser, Subcommand, ValueEnum};
use std::io::{self, Read};
use std::path::Path;
#[cfg(feature = "watch")]
use std::path::PathBuf;
#[cfg(feature = "watch")]
use std::time::Duration;

/// Tokuin - Estimate token usage and costs for LLM prompts.
#[derive(Parser, Debug)]
#[command(name = "tokuin")]
#[command(about = "A fast CLI tool to estimate token usage and API costs for LLM prompts")]
#[command(
    long_about = r#"Tokuin - A fast CLI tool for LLM prompt analysis and optimization

CORE FEATURES:
  • Token Counting: Accurate token estimation for OpenAI, Claude, Mistral, and more
  • Cost Estimation: Calculate API costs based on token pricing per model
  • Multi-Model Comparison: Compare token usage and costs across multiple providers
  • Role-Based Breakdown: Show token count by system/user/assistant role messages
  • Multiple Input Formats: Support plain text, JSON chat formats, and stdin
  • Flexible Output: Human-readable text, JSON, or Markdown output

COMPRESSION FEATURES (--features compression):
  • Prompt Compression: Compress prompts by 70-90% using Hieratic format
  • Semantic Scoring: Use ONNX embeddings for better compression quality
  • Quality Metrics: Evaluate compression fidelity and information retention
  • Incremental Compression: Optimize multi-turn conversations efficiently
  • Context Library: Extract and reuse common patterns across prompts

ADDITIONAL FEATURES:
  • Load Testing (--features load-test): Run concurrent load tests with real-time metrics
  • Prompt Library Analysis: Scan directories, detect duplicates, estimate costs at scale
  • Markdown Support (--features markdown): Parse and minify markdown content
  • Watch Mode (--features watch): Automatically re-run on file changes
  • Gemini Tokenization (--features gemini): Support for Google Gemini models

EXAMPLES:
  # Basic token counting
  tokuin prompt.txt --model gpt-4

  # Compare multiple models
  tokuin prompt.txt --compare gpt-4 claude-2 mistral-large

  # Compress a prompt
  tokuin compress prompt.txt --level aggressive --quality

  # Analyze a prompt library
  tokuin analyze-prompts ./prompts --top-n 20

  # Run load tests
  tokuin load-test --model gpt-4 --runs 100 --concurrency 10

For more information, visit: https://github.com/nooscraft/tokuin"#
)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    // Legacy flat arguments for backward compatibility
    /// Input file path (use '-' for stdin or omit for direct text input)
    #[arg(value_name = "FILE|TEXT")]
    pub input: Option<String>,

    /// Model to use for tokenization (e.g., gpt-4, gpt-3.5-turbo)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Compare multiple models
    #[arg(short, long, num_args = 1..)]
    pub compare: Vec<String>,

    /// Show token breakdown by role (system/user/assistant)
    #[arg(short, long)]
    pub breakdown: bool,

    /// Output format
    #[arg(short, long, value_enum, default_value = "text")]
    pub format: OutputFormat,

    /// Show pricing information
    #[arg(short, long)]
    pub price: bool,

    /// Strip markdown formatting to show token savings
    #[arg(long)]
    #[cfg(feature = "markdown")]
    pub minify: bool,

    /// Compare two prompts and show token differences
    #[arg(long)]
    pub diff: Option<String>,

    /// Watch file for changes and re-run automatically
    #[arg(short, long)]
    #[cfg(feature = "watch")]
    pub watch: bool,

    /// Path to a custom pricing configuration (TOML).
    #[arg(long, value_name = "FILE", global = true)]
    pub pricing_file: Option<String>,
}

/// Available commands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Estimate tokens and costs (default behavior)
    Estimate {
        /// Input file path (use '-' for stdin or omit for direct text input)
        #[arg(value_name = "FILE|TEXT")]
        input: Option<String>,

        /// Model to use for tokenization (e.g., gpt-4, gpt-3.5-turbo)
        #[arg(short, long)]
        model: Option<String>,

        /// Compare multiple models
        #[arg(short, long, num_args = 1..)]
        compare: Vec<String>,

        /// Show token breakdown by role (system/user/assistant)
        #[arg(short, long)]
        breakdown: bool,

        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        format: OutputFormat,

        /// Show pricing information
        #[arg(short, long)]
        price: bool,

        /// Strip markdown formatting to show token savings
        #[arg(long)]
        #[cfg(feature = "markdown")]
        minify: bool,

        /// Compare two prompts and show token differences
        #[arg(long)]
        diff: Option<String>,

        /// Watch file for changes and re-run automatically
        #[arg(short, long)]
        #[cfg(feature = "watch")]
        watch: bool,
    },

    /// Run load tests against LLM APIs
    #[cfg(feature = "load-test")]
    #[command(name = "load-test")]
    LoadTest {
        /// Model to use (e.g., gpt-4, claude-2)
        #[arg(short, long)]
        model: String,

        /// API endpoint URL (optional, uses provider default if not specified)
        #[arg(long)]
        endpoint: Option<String>,

        /// API key (or use environment variable)
        #[arg(long)]
        api_key: Option<String>,

        /// OpenAI API key (alternative to --api-key)
        #[arg(long)]
        openai_api_key: Option<String>,

        /// Anthropic API key
        #[arg(long)]
        anthropic_api_key: Option<String>,

        /// OpenRouter API key
        #[arg(long)]
        openrouter_api_key: Option<String>,

        /// Target provider (defaults to automatic detection)
        #[arg(long, value_enum)]
        provider: Option<Provider>,

        /// Number of concurrent requests
        #[arg(short, long, default_value = "10")]
        concurrency: usize,

        /// Total number of requests to make
        #[arg(short, long)]
        runs: usize,

        /// Prompt file (JSONL, YAML, or text) or use stdin
        #[arg(short, long)]
        prompt_file: Option<String>,

        /// Think time between requests (e.g., "250-750ms" or "500ms")
        #[arg(long)]
        think_time: Option<String>,

        /// Retry count on failure
        #[arg(long, default_value = "3")]
        retry: u32,

        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        output_format: LoadTestOutputFormat,

        /// Estimate costs (dry run without making API calls)
        #[arg(long)]
        dry_run: bool,

        /// Maximum cost threshold (stop if exceeded)
        #[arg(long)]
        max_cost: Option<f64>,

        /// Show cost estimation
        #[arg(short, long)]
        estimate_cost: bool,
    },

    /// Analyze a prompt library directory
    #[command(name = "analyze-prompts")]
    Analyze {
        /// Directory to analyze
        #[arg(value_name = "DIR")]
        folder: String,

        /// Model to use for tokenization (default: gpt-4)
        #[arg(short, long, default_value = "gpt-4")]
        model: String,

        /// Top N most expensive prompts to show
        #[arg(long, default_value = "10")]
        top_n: usize,

        /// Monthly invocation count for cost projection
        #[arg(long, default_value = "1000")]
        monthly_invocations: u64,

        /// Model context window limit to check against
        #[arg(long)]
        context_limit: Option<usize>,

        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Extract reusable context patterns from a prompt library
    #[cfg(feature = "compression")]
    #[command(
        name = "extract-context",
        long_about = r#"Extract reusable context patterns from a directory of prompts to build a context library.

This command analyzes multiple prompt files to identify common patterns, instructions, and reusable content. These patterns are saved to a context library (TOML format) that can be referenced during compression, reducing token usage by reusing common elements.

The context library enables:
  • Cross-prompt pattern reuse
  • Reduced token usage through reference-based compression
  • Consistent instruction formatting across prompts

EXAMPLES:
  # Extract patterns from a prompt directory
  tokuin extract-context ./prompts

  # Custom output file and thresholds
  tokuin extract-context ./prompts --output my_contexts.toml --min-frequency 3 --min-similarity 0.9

  # Use with compression
  tokuin compress prompt.txt --context-lib my_contexts.toml

The extracted context library can be used with the compress command via --context-lib flag."#
    )]
    ExtractContext {
        /// Directory containing prompt files
        #[arg(value_name = "DIR")]
        directory: String,

        /// Output file for context library (default: contexts.toml)
        #[arg(short, long, default_value = "contexts.toml")]
        output: String,

        /// Minimum frequency for a pattern (default: 2)
        #[arg(long, default_value = "2")]
        min_frequency: usize,

        /// Minimum similarity threshold (0.0-1.0, default: 0.85)
        #[arg(long, default_value = "0.85")]
        min_similarity: f64,

        /// Model to use for tokenization (default: gpt-4)
        #[arg(short, long, default_value = "gpt-4")]
        model: String,
    },

    /// Compress a prompt to Hieratic format
    #[cfg(feature = "compression")]
    #[command(
        long_about = r#"Compress prompts using the Hieratic format - an LLM-parseable compression format that preserves semantic meaning while reducing token count by 70-90%.

The Hieratic format uses structured sections (role, examples, constraints, task) and context references to achieve high compression ratios while maintaining quality. Quality metrics help ensure critical information is preserved.

EXAMPLES:
  # Basic compression with default settings
  tokuin compress prompt.txt

  # Aggressive compression with quality metrics
  tokuin compress prompt.txt --level aggressive --quality

  # Use semantic scoring for better quality
  tokuin compress prompt.txt --scoring semantic --quality

  # Compress structured content (JSON, code, tables)
  tokuin compress api_docs.txt --structured --level medium

  # Incremental compression for multi-turn conversations
  tokuin compress conversation.txt --incremental

  # Output as JSON for programmatic use
  tokuin compress prompt.txt --format json --output results.json

COMPRESSION LEVELS:
  • light:      30-50% reduction, best quality preservation
  • medium:     50-70% reduction, balanced quality and compression
  • aggressive: 70-90% reduction, maximum compression

SCORING MODES:
  • heuristic: Fast keyword-based scoring (default)
  • semantic:  Embedding-based scoring for better quality (requires compression-embeddings)
  • hybrid:    Combines both approaches for optimal results

For more information, see: https://github.com/nooscraft/tokuin"#
    )]
    Compress {
        /// Input prompt file
        #[arg(value_name = "FILE")]
        input: String,

        /// Output file (default: <input>.hieratic)
        #[arg(short, long)]
        output: Option<String>,

        /// Compression level
        #[arg(short, long, value_enum, default_value = "medium")]
        level: CompressionLevelArg,

        /// Context library path (default: contexts.toml)
        #[arg(long)]
        context_lib: Option<String>,

        /// Force inline mode (no context references)
        #[arg(long)]
        inline: bool,

        /// Enable structured document mode (better for JSON, code, tables, technical docs)
        #[arg(long)]
        structured: bool,

        /// Enable incremental compression (only compress new content)
        #[arg(long)]
        incremental: bool,

        /// Custom incremental state file path (default: <input>.state.json)
        #[arg(long, requires = "incremental")]
        previous: Option<String>,

        /// Token threshold for creating anchors (default: 1000)
        #[arg(long, default_value = "1000")]
        anchor_threshold: usize,

        /// Token retention threshold (default: 500)
        #[arg(long, default_value = "500")]
        retention_threshold: usize,

        /// Model to use for tokenization (default: gpt-4)
        #[arg(short, long, default_value = "gpt-4")]
        model: String,

        /// Output format
        #[arg(short, long, value_enum, default_value = "hieratic")]
        format: CompressionOutputFormat,

        /// Scoring mode (heuristic, semantic, or hybrid)
        #[arg(long, value_enum, default_value = "heuristic")]
        scoring: CompressionScoringMode,

        /// Calculate and display quality metrics
        #[arg(long)]
        quality: bool,

        /// Enable LLM-as-a-judge evaluation (requires load-test feature)
        #[cfg(all(feature = "compression", feature = "load-test"))]
        #[arg(long)]
        llm_judge: bool,

        /// Model to use for generating outputs (default: same as --model or anthropic/claude-3-opus)
        #[cfg(all(feature = "compression", feature = "load-test"))]
        #[arg(long)]
        evaluation_model: Option<String>,

        /// Model to use for judging outputs (default: anthropic/claude-3-opus)
        #[cfg(all(feature = "compression", feature = "load-test"))]
        #[arg(long, default_value = "anthropic/claude-3-opus")]
        judge_model: String,

        /// API key for judge/evaluation (or use OPENROUTER_API_KEY env var)
        #[cfg(all(feature = "compression", feature = "load-test"))]
        #[arg(long)]
        judge_api_key: Option<String>,

        /// Provider for judge/evaluation (default: openrouter)
        #[cfg(all(feature = "compression", feature = "load-test"))]
        #[arg(long, value_enum, default_value = "openrouter")]
        judge_provider: JudgeProvider,
    },

    /// Expand a Hieratic file back to full prompt
    #[cfg(feature = "compression")]
    #[command(
        long_about = r#"Expand a compressed Hieratic file back to its full prompt text.

The expand command reconstructs the original prompt from the Hieratic format, resolving context references and reconstructing all sections. This is useful for:
  • Verifying compression quality
  • Using compressed prompts with LLMs that don't support Hieratic format
  • Debugging compression issues

EXAMPLES:
  # Expand to stdout
  tokuin expand prompt.hieratic

  # Expand to a file
  tokuin expand prompt.hieratic --output expanded.txt

  # Use custom context library
  tokuin expand prompt.hieratic --context-lib custom.toml

The expanded output should closely match the original prompt, with quality metrics indicating how well information was preserved."#
    )]
    Expand {
        /// Hieratic file to expand
        #[arg(value_name = "FILE")]
        input: String,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Context library path (default: contexts.toml)
        #[arg(long)]
        context_lib: Option<String>,
    },

    /// Setup and configure Tokuin
    #[cfg(feature = "compression-embeddings")]
    Setup {
        /// Setup embedding models
        #[command(subcommand)]
        subcommand: SetupSubcommand,
    },
}

/// Setup subcommands
#[cfg(feature = "compression-embeddings")]
#[derive(Subcommand, Debug)]
pub enum SetupSubcommand {
    /// Setup embedding models
    Models {
        /// Force re-download even if models exist
        #[arg(long)]
        force: bool,

        /// Also download/convert ONNX model
        #[arg(long)]
        onnx: bool,
    },
}

/// Output format options.
#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text output
    Text,
    /// JSON output for scripting
    Json,
    /// Markdown report format
    #[cfg(feature = "markdown")]
    Markdown,
}

/// Compression level options.
#[cfg(feature = "compression")]
#[derive(Debug, Clone, ValueEnum)]
pub enum CompressionLevelArg {
    /// Light compression (30-50% reduction)
    Light,
    /// Medium compression (50-70% reduction)
    Medium,
    /// Aggressive compression (70-90% reduction)
    Aggressive,
}

/// Compression output format options.
#[cfg(feature = "compression")]
#[derive(Debug, Clone, PartialEq, Eq, ValueEnum)]
pub enum CompressionOutputFormat {
    /// Hieratic format (.hieratic)
    Hieratic,
    /// Expanded full text
    Expanded,
    /// JSON format
    Json,
}

/// Compression scoring mode options.
#[cfg(feature = "compression")]
#[derive(Debug, Clone, ValueEnum)]
pub enum CompressionScoringMode {
    /// Heuristic-based scoring (keyword matching, position-based)
    Heuristic,
    /// Embedding-based semantic scoring (requires compression-embeddings feature)
    Semantic,
    /// Hybrid: combine embeddings and heuristics (best of both)
    Hybrid,
}

/// Judge provider options for LLM-as-a-judge evaluation.
#[cfg(all(feature = "compression", feature = "load-test"))]
#[derive(Debug, Clone, ValueEnum)]
pub enum JudgeProvider {
    /// OpenAI API
    Openai,
    /// OpenRouter API (default, unified access to 400+ models)
    Openrouter,
    /// Anthropic API
    Anthropic,
}

/// Load test output format options.
#[cfg(feature = "load-test")]
#[derive(Debug, Clone, ValueEnum)]
pub enum LoadTestOutputFormat {
    /// Human-readable text output
    Text,
    /// JSON output for scripting
    Json,
    /// CSV output
    Csv,
    /// Prometheus metrics format
    Prometheus,
    /// Markdown report format
    Markdown,
}

/// Supported LLM providers for load testing.
#[cfg(feature = "load-test")]
#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
pub enum Provider {
    Openai,
    Openrouter,
    Anthropic,
    Generic,
}

#[cfg(feature = "load-test")]
impl Provider {
    fn as_str(&self) -> &'static str {
        match self {
            Provider::Openai => "openai",
            Provider::Openrouter => "openrouter",
            Provider::Anthropic => "anthropic",
            Provider::Generic => "generic",
        }
    }
}

impl Cli {
    /// Execute the CLI command.
    pub fn run(self) -> Result<(), AppError> {
        match self.command {
            Some(Command::Estimate {
                input,
                model,
                compare,
                breakdown,
                format,
                price,
                #[cfg(feature = "markdown")]
                minify,
                diff,
                #[cfg(feature = "watch")]
                watch,
            }) => {
                // Use subcommand args, but fall back to top-level args if not provided
                let estimate_args = EstimateArgs {
                    input: input.or(self.input),
                    model: model.or(self.model),
                    compare: if !compare.is_empty() {
                        compare
                    } else {
                        self.compare
                    },
                    breakdown: breakdown || self.breakdown,
                    format,
                    price: price || self.price,
                    #[cfg(feature = "markdown")]
                    minify: minify || self.minify,
                    diff: diff.or(self.diff),
                    #[cfg(feature = "watch")]
                    watch: watch || self.watch,
                    pricing_file: self.pricing_file.clone(),
                };
                Self::run_estimate(estimate_args)
            }
            #[cfg(feature = "load-test")]
            Some(Command::LoadTest {
                model,
                endpoint,
                api_key,
                openai_api_key,
                anthropic_api_key,
                openrouter_api_key,
                provider,
                concurrency,
                runs,
                prompt_file,
                think_time,
                retry,
                output_format,
                dry_run,
                max_cost,
                estimate_cost,
            }) => {
                let load_args = LoadTestArgs {
                    model,
                    endpoint,
                    api_key,
                    openai_api_key,
                    anthropic_api_key,
                    openrouter_api_key,
                    provider,
                    concurrency,
                    runs,
                    prompt_file,
                    think_time,
                    retry,
                    output_format,
                    dry_run,
                    max_cost,
                    estimate_cost,
                    pricing_file: self.pricing_file.clone(),
                };
                Self::run_load_test(load_args)
            }
            Some(Command::Analyze {
                folder,
                model,
                top_n,
                monthly_invocations,
                context_limit,
                format,
            }) => Self::run_analyze(
                folder,
                model,
                top_n,
                monthly_invocations,
                context_limit,
                format,
            ),
            #[cfg(feature = "compression")]
            Some(Command::ExtractContext {
                directory,
                output,
                min_frequency,
                min_similarity,
                model,
            }) => {
                Self::run_extract_context(directory, output, min_frequency, min_similarity, model)
            }
            #[cfg(feature = "compression")]
            Some(Command::Compress {
                input,
                output,
                level,
                context_lib,
                inline,
                structured,
                incremental,
                previous,
                anchor_threshold,
                retention_threshold,
                model,
                format,
                scoring,
                quality,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                llm_judge,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                evaluation_model,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                judge_model,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                judge_api_key,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                judge_provider,
            }) => Self::run_compress(
                input,
                output,
                level,
                context_lib,
                inline,
                structured,
                incremental,
                previous,
                anchor_threshold,
                retention_threshold,
                model,
                format,
                scoring,
                quality,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                llm_judge,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                evaluation_model,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                judge_model,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                judge_api_key,
                #[cfg(all(feature = "compression", feature = "load-test"))]
                judge_provider,
            ),
            #[cfg(feature = "compression")]
            Some(Command::Expand {
                input,
                output,
                context_lib,
            }) => Self::run_expand(input, output, context_lib),
            #[cfg(feature = "compression-embeddings")]
            Some(Command::Setup { subcommand }) => {
                use crate::cli::SetupSubcommand;
                match subcommand {
                    SetupSubcommand::Models { force, onnx } => Self::run_setup_models(force, onnx),
                }
            }
            None => {
                // Backward compatibility: use flat structure
                let estimate_args = EstimateArgs {
                    input: self.input,
                    model: self.model,
                    compare: self.compare,
                    breakdown: self.breakdown,
                    format: self.format,
                    price: self.price,
                    #[cfg(feature = "markdown")]
                    minify: self.minify,
                    diff: self.diff,
                    #[cfg(feature = "watch")]
                    watch: self.watch,
                    pricing_file: self.pricing_file,
                };
                Self::run_estimate(estimate_args)
            }
        }
    }

    /// Run estimate command (existing functionality).
    fn run_estimate(args: EstimateArgs) -> Result<(), AppError> {
        #[cfg(feature = "watch")]
        if args.watch {
            return Self::run_watch(&args);
        }

        // Handle diff mode
        if let Some(ref diff_file) = args.diff {
            return Self::run_diff(&args, diff_file);
        }

        let registry = ModelRegistry::new_with_pricing(args.pricing_file.as_deref())
            .map_err(AppError::Model)?;

        // Determine input
        let input = Self::get_input(&args.input)?;

        // Apply minify if requested
        #[cfg(feature = "markdown")]
        let original_input = if args.minify {
            let stripped = markdown::strip_markdown(&input);
            let savings = markdown::calculate_savings(&input, &stripped);
            eprintln!(
                "Markdown stripped: {} characters saved (~{} tokens)",
                savings,
                savings / 4
            );
            stripped
        } else {
            input.clone()
        };

        #[cfg(not(feature = "markdown"))]
        let original_input = input.clone();

        let breakdown = args.breakdown;
        let price = args.price;

        // Determine models to use
        let models = if !args.compare.is_empty() {
            args.compare.clone()
        } else if let Some(model) = &args.model {
            vec![model.clone()]
        } else {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "No model specified. Use --model or --compare".to_string(),
            )));
        };

        // Parse input
        let parser: Box<dyn InputParser> = if original_input.trim_start().starts_with('{')
            || original_input.trim_start().starts_with('[')
        {
            Box::new(JsonParser::new())
        } else {
            Box::new(TextParser::new())
        };

        let messages = parser.parse(&original_input)?;

        // Process each model
        let mut results = Vec::new();
        for model_name in &models {
            let tokenizer = registry.get_tokenizer(model_name)?;
            let tokenizer_name = tokenizer.name().to_string();
            let pricing_override = if price {
                registry
                    .pricing_for(model_name)
                    .or_else(|| registry.pricing_for(&tokenizer_name))
            } else {
                None
            };
            let result = Self::count_tokens(
                &*tokenizer,
                &messages,
                &tokenizer_name,
                breakdown,
                price,
                pricing_override,
            )?;
            results.push(result);
        }

        // Format and print output
        let formatter: Box<dyn Formatter> = match args.format {
            OutputFormat::Text => Box::new(TextFormatter::new(args.breakdown)),
            OutputFormat::Json => Box::new(JsonFormatter::new()),
            #[cfg(feature = "markdown")]
            OutputFormat::Markdown => Box::new(MarkdownFormatter::new(args.breakdown)),
        };

        if results.len() == 1 {
            println!("{}", formatter.format_result(&results[0]));
        } else {
            println!("{}", formatter.format_comparison(&results));
        }

        Ok(())
    }

    /// Run load test command.
    #[cfg(feature = "load-test")]
    fn run_load_test(args: LoadTestArgs) -> Result<(), AppError> {
        use crate::http::client::ClientConfig;
        use crate::http::providers::anthropic::AnthropicClient;
        use crate::http::providers::generic::GenericClient;
        use crate::http::providers::openai::OpenAIClient;
        use crate::http::providers::openrouter::OpenRouterClient;
        use crate::simulator::config::SimulatorConfig;
        use crate::simulator::simulator::Simulator;
        use std::sync::Arc;
        use std::time::Duration;

        let provider = args.resolve_provider();

        let api_key = Cli::resolve_api_key(&args, &provider)?;

        // Get prompt
        let prompt = if let Some(ref prompt_file) = args.prompt_file {
            std::fs::read_to_string(prompt_file).map_err(|e| {
                AppError::Io(std::io::Error::other(format!(
                    "Failed to read prompt file: {}",
                    e
                )))
            })?
        } else {
            // Read from stdin
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            buffer
        };

        if prompt.trim().is_empty() {
            return Err(AppError::Config("Prompt cannot be empty".to_string()));
        }

        // Build simulator config
        let mut sim_config = SimulatorConfig::new(args.concurrency, args.runs);
        sim_config.retry = args.retry;
        sim_config.dry_run = args.dry_run;
        sim_config.max_cost = args.max_cost;
        sim_config.timeout = Duration::from_secs(60);

        if let Some(ref think_time_str) = args.think_time {
            sim_config.think_time =
                Some(SimulatorConfig::parse_think_time(think_time_str).map_err(AppError::Config)?);
        }

        // Create HTTP client config
        let endpoint = args.endpoint.clone().unwrap_or_default();
        let model = args.model.clone();
        let estimate_cost = args.estimate_cost;
        let pricing_registry = if estimate_cost {
            Some(
                ModelRegistry::new_with_pricing(args.pricing_file.as_deref())
                    .map_err(AppError::Model)?,
            )
        } else {
            None
        };

        if provider == Provider::Generic && endpoint.is_empty() && !args.dry_run {
            return Err(AppError::Config(
                "Generic provider requires --endpoint to be specified".to_string(),
            ));
        }

        let client_config = ClientConfig {
            endpoint,
            api_key: api_key.clone(),
            timeout: Duration::from_secs(60),
            headers: Vec::new(),
        };

        // Create client based on detected provider
        use crate::http::client::LlmClientEnum;
        let client = match provider {
            Provider::Openrouter => Arc::new(LlmClientEnum::OpenRouter(OpenRouterClient::new(
                client_config,
            )?)),
            Provider::Openai => Arc::new(LlmClientEnum::OpenAI(OpenAIClient::new(client_config)?)),
            Provider::Anthropic => Arc::new(LlmClientEnum::Anthropic(AnthropicClient::new(
                client_config,
            )?)),
            Provider::Generic => {
                Arc::new(LlmClientEnum::Generic(GenericClient::new(client_config)?))
            }
        };

        if args.dry_run {
            eprintln!(
                "Dry run mode: No API calls will be made (provider: {})",
                provider.as_str()
            );
        } else {
            eprintln!(
                "Starting load test using provider '{}' with {} requests at concurrency {}",
                provider.as_str(),
                args.runs,
                args.concurrency
            );
        }

        let simulator = Simulator::new(sim_config);

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AppError::Config(format!("Failed to create async runtime: {}", e)))?;

        // Create progress bar
        let progress_bar = if !args.dry_run {
            let pb = indicatif::ProgressBar::new(args.runs as u64);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} {msg}")
                    .expect("valid progress bar template")
                    .progress_chars("#>-"),
            );
            pb.set_message("Starting load test...");
            Some(Arc::new(pb))
        } else {
            None
        };

        let results = if let Some(ref pb) = progress_bar {
            rt.block_on(simulator.run_with_progress(client, &prompt, &model, Some(pb.clone())))?
        } else {
            rt.block_on(simulator.run(client, &prompt, &model))?
        };

        // Calculate and display metrics
        Self::display_load_test_results(
            &results,
            &model,
            &args.output_format,
            estimate_cost,
            pricing_registry.as_ref(),
        )?;

        Ok(())
    }

    #[cfg(feature = "load-test")]
    fn resolve_api_key(args: &LoadTestArgs, provider: &Provider) -> Result<String, AppError> {
        let key = match provider {
            Provider::Openai => args
                .openai_api_key
                .clone()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .or_else(|| args.api_key.clone())
                .or_else(|| std::env::var("API_KEY").ok()),
            Provider::Openrouter => args
                .openrouter_api_key
                .clone()
                .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
                .or_else(|| args.api_key.clone())
                .or_else(|| std::env::var("API_KEY").ok()),
            Provider::Anthropic => args
                .anthropic_api_key
                .clone()
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                .or_else(|| args.api_key.clone())
                .or_else(|| std::env::var("API_KEY").ok()),
            Provider::Generic => args
                .api_key
                .clone()
                .or_else(|| std::env::var("API_KEY").ok()),
        };

        match (provider, key) {
            (Provider::Generic, Some(key)) => Ok(key),
            (Provider::Generic, None) => Ok(String::new()),
            (_, Some(key)) => Ok(key),
            (_, None) if args.dry_run => Ok("dummy-key-for-dry-run".to_string()),
            (Provider::Openai, None) => Err(AppError::Config(
                "API key required for OpenAI. Use --openai-api-key, --api-key, or set OPENAI_API_KEY/API_KEY."
                    .to_string(),
            )),
            (Provider::Openrouter, None) => Err(AppError::Config(
                "API key required for OpenRouter. Use --openrouter-api-key, --api-key, or set OPENROUTER_API_KEY/API_KEY."
                    .to_string(),
            )),
            (Provider::Anthropic, None) => Err(AppError::Config(
                "API key required for Anthropic. Use --anthropic-api-key, --api-key, or set ANTHROPIC_API_KEY/API_KEY."
                    .to_string(),
            )),
        }
    }

    /// Display load test results.
    #[cfg(feature = "load-test")]
    fn display_load_test_results(
        results: &[crate::simulator::simulator::RequestResult],
        model: &str,
        output_format: &LoadTestOutputFormat,
        estimate_cost: bool,
        pricing_registry: Option<&ModelRegistry>,
    ) -> Result<(), AppError> {
        let total_requests = results.len();
        let successful = results.iter().filter(|r| r.success).count();
        let failed = total_requests.saturating_sub(successful);
        let success_rate = if total_requests > 0 {
            (successful as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };
        let failure_rate = if total_requests > 0 {
            100.0 - success_rate
        } else {
            0.0
        };

        let latencies: Vec<u64> = results
            .iter()
            .filter_map(|r| if r.success { Some(r.latency_ms) } else { None })
            .collect();

        let total_tokens_reported: usize = results.iter().filter_map(|r| r.total_tokens).sum();
        let input_tokens: usize = results.iter().filter_map(|r| r.input_tokens).sum();
        let output_tokens: usize = results.iter().filter_map(|r| r.output_tokens).sum();

        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<u64>() as f64 / latencies.len() as f64
        } else {
            0.0
        };

        let p50 = if !latencies.is_empty() {
            let mut sorted = latencies.clone();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        } else {
            0
        };

        let p95 = if !latencies.is_empty() {
            let mut sorted = latencies.clone();
            sorted.sort_unstable();
            let index = ((sorted.len() as f64) * 0.95).ceil() as usize;
            let index = index.clamp(0, sorted.len().saturating_sub(1));
            sorted[index]
        } else {
            0
        };

        let (input_cost, output_cost, total_cost) = if estimate_cost {
            if let Some(registry) = pricing_registry {
                if let Some((input_rate, output_rate)) = registry
                    .pricing_for(model)
                    .or_else(|| registry.pricing_for(model.rsplit('/').next().unwrap_or(model)))
                {
                    let input_cost = (input_tokens as f64 / 1000.0) * input_rate;
                    let output_cost = (output_tokens as f64 / 1000.0) * output_rate;
                    (
                        Some(input_cost),
                        Some(output_cost),
                        Some(input_cost + output_cost),
                    )
                } else {
                    (None, None, None)
                }
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

        match output_format {
            LoadTestOutputFormat::Text => {
                println!("\n=== Load Test Results ===");
                println!("Total Requests: {}", total_requests);
                println!("Successful: {} ({:.1}%)", successful, success_rate);
                println!("Failed: {} ({:.1}%)", failed, failure_rate);
                println!("\nLatency (ms):");
                println!("  Average: {:.2}", avg_latency);
                println!("  p50: {}", p50);
                println!("  p95: {}", p95);
                if total_tokens_reported > 0 {
                    println!("  Total tokens reported: {}", total_tokens_reported);
                }

                if let Some(sample) = results
                    .iter()
                    .find(|r| r.success)
                    .and_then(|r| r.content.as_ref())
                {
                    let preview = sample.chars().take(80).collect::<String>();
                    println!("  Example response: {}", preview);
                }

                if let Some(err) = results
                    .iter()
                    .find(|r| !r.success)
                    .and_then(|r| r.error.as_ref())
                {
                    println!("  Last error: {}", err);
                }

                if estimate_cost {
                    println!("\nToken Usage:");
                    println!("  Input tokens: {}", input_tokens);
                    println!("  Output tokens: {}", output_tokens);

                    if let (Some(in_cost), Some(out_cost), Some(total)) =
                        (input_cost, output_cost, total_cost)
                    {
                        println!("\nCost Estimation:");
                        println!("  Input cost: ${:.6}", in_cost);
                        println!("  Output cost: ${:.6}", out_cost);
                        println!("  Total cost: ${:.6}", total);
                    }
                }
            }
            LoadTestOutputFormat::Json => {
                let json = serde_json::json!({
                    "total_requests": total_requests,
                    "successful": successful,
                    "failed": failed,
                    "success_rate": success_rate,
                    "latency_ms": {
                        "average": avg_latency,
                        "p50": p50,
                        "p95": p95,
                    },
                    "tokens": {
                        "reported": total_tokens_reported,
                        "input": input_tokens,
                        "output": output_tokens,
                    },
                    "cost_usd": {
                        "input": input_cost,
                        "output": output_cost,
                        "total": total_cost,
                    },
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json).map_err(AppError::Json)?
                );
            }
            LoadTestOutputFormat::Csv => {
                let format_opt = |value: Option<f64>| -> String {
                    value.map(|v| format!("{:.6}", v)).unwrap_or_default()
                };

                println!("total_requests,successful,failed,success_rate,avg_latency_ms,p50_latency_ms,p95_latency_ms,total_tokens,input_tokens,output_tokens,input_cost_usd,output_cost_usd,total_cost_usd");
                println!(
                    "{},{},{},{:.4},{:.2},{},{},{},{},{},{},{},{}",
                    total_requests,
                    successful,
                    failed,
                    success_rate,
                    avg_latency,
                    p50,
                    p95,
                    total_tokens_reported,
                    input_tokens,
                    output_tokens,
                    format_opt(input_cost),
                    format_opt(output_cost),
                    format_opt(total_cost)
                );
            }
            _ => {
                println!("Format {:?} not yet implemented", output_format);
            }
        }

        Ok(())
    }

    /// Run analyze-prompts command.
    fn run_analyze(
        folder: String,
        model: String,
        top_n: usize,
        monthly_invocations: u64,
        context_limit: Option<usize>,
        format: OutputFormat,
    ) -> Result<(), AppError> {
        use crate::analyzers::PromptScanner;
        use crate::output::InsightsFormatter;

        let folder_path = Path::new(&folder);
        if !folder_path.exists() {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                format!("Directory does not exist: {}", folder),
            )));
        }
        if !folder_path.is_dir() {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                format!("Path is not a directory: {}", folder),
            )));
        }

        // Create model registry
        let registry = ModelRegistry::new_with_pricing(None).map_err(AppError::Model)?;

        // Create scanner
        let scanner = PromptScanner::new(registry, model.clone(), context_limit);

        eprintln!("Scanning directory: {}", folder);
        let analyses = scanner.scan_directory(folder_path)?;

        if analyses.is_empty() {
            eprintln!("No prompt files found in directory.");
            return Ok(());
        }

        eprintln!("Analyzed {} prompt files", analyses.len());

        // Generate insights
        let insights = PromptScanner::generate_insights(&analyses, top_n, monthly_invocations);

        // Format output
        match format {
            OutputFormat::Text => {
                let output = InsightsFormatter::format_text(&insights, &model, context_limit);
                println!("{}", output);
            }
            OutputFormat::Json => {
                let output = InsightsFormatter::format_json(&insights)
                    .map_err(|e| AppError::Parse(crate::error::ParseError::InvalidJson(e)))?;
                println!("{}", output);
            }
            #[cfg(feature = "markdown")]
            OutputFormat::Markdown => {
                // For now, use text format for markdown
                let output = InsightsFormatter::format_text(&insights, &model, context_limit);
                println!("{}", output);
            }
        }

        Ok(())
    }

    /// Extract context patterns from a prompt library
    #[cfg(feature = "compression")]
    fn run_extract_context(
        directory: String,
        output: String,
        min_frequency: usize,
        min_similarity: f64,
        model: String,
    ) -> Result<(), AppError> {
        use crate::compression::context_library::ContextLibraryManager;
        use crate::compression::pattern_extractor::{ExtractionConfig, PatternExtractor};

        let registry = ModelRegistry::new_with_pricing(None).map_err(AppError::Model)?;
        let tokenizer = registry.get_tokenizer(&model)?;

        let config = ExtractionConfig {
            min_frequency,
            min_similarity,
            ..Default::default()
        };

        eprintln!("Extracting patterns from: {}", directory);
        let extractor = PatternExtractor::new(tokenizer, config);
        let library = extractor.extract_from_directory(Path::new(&directory))?;

        eprintln!("Extracted {} patterns", library.patterns.len());

        let mut manager = ContextLibraryManager::new();
        *manager.library_mut() = library;
        manager.save_to_file(&output)?;

        eprintln!("Context library saved to: {}", output);
        Ok(())
    }

    /// Compress a prompt to Hieratic format
    #[cfg(feature = "compression")]
    #[allow(clippy::too_many_arguments)]
    // Compression has many configuration options
    // Non-load-test version - inline implementation since we can't use LLM judge
    // This avoids the complexity of conditional function signatures
    #[cfg(not(all(feature = "compression", feature = "load-test")))]
    #[allow(clippy::too_many_arguments)]
    fn run_compress(
        input: String,
        output: Option<String>,
        level: CompressionLevelArg,
        context_lib: Option<String>,
        inline: bool,
        structured: bool,
        incremental: bool,
        previous: Option<String>,
        anchor_threshold: usize,
        retention_threshold: usize,
        model: String,
        format: CompressionOutputFormat,
        scoring: CompressionScoringMode,
        quality: bool,
    ) -> Result<(), AppError> {
        // For non-load-test builds, we can't use LLM judge
        // Just call the internal function with None for LLM judge params
        // But we need to handle this differently since the function signature changes
        // For now, we'll duplicate the logic or use a macro
        // Actually, let's just inline it for the non-load-test case
        use crate::compression::compressor::Compressor;
        use crate::compression::context_library::ContextLibraryManager;
        use crate::compression::hieratic_encoder::HieraticEncoder;
        use crate::compression::types::{
            CompressionAnchor, CompressionConfig, CompressionLevel, ScoringMode,
        };
        use std::fs;
        use std::path::Path;

        let registry = ModelRegistry::new_with_pricing(None).map_err(AppError::Model)?;
        let tokenizer = registry.get_tokenizer(&model)?;
        let prompt = fs::read_to_string(&input).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to read input file: {}",
                e
            )))
        })?;

        let mut compressor = Compressor::new(tokenizer);

        if !inline {
            if let Some(ref lib_path) = context_lib {
                if std::path::Path::new(lib_path).exists() {
                    let library = ContextLibraryManager::load_from_file(lib_path)?;
                    compressor = compressor.with_context_library(library);
                }
            } else if std::path::Path::new("contexts.toml").exists() {
                let library = ContextLibraryManager::load_from_file("contexts.toml")?;
                compressor = compressor.with_context_library(library);
            }
        }

        let compression_level = match level {
            CompressionLevelArg::Light => CompressionLevel::Light,
            CompressionLevelArg::Medium => CompressionLevel::Medium,
            CompressionLevelArg::Aggressive => CompressionLevel::Aggressive,
        };

        let scoring_mode = match scoring {
            CompressionScoringMode::Heuristic => ScoringMode::Heuristic,
            CompressionScoringMode::Semantic => ScoringMode::Semantic,
            CompressionScoringMode::Hybrid => ScoringMode::Hybrid,
        };

        // Initialize embeddings if semantic or hybrid mode is selected
        #[cfg(feature = "compression-embeddings")]
        if matches!(scoring_mode, ScoringMode::Semantic | ScoringMode::Hybrid) {
            use crate::compression::embeddings::OnnxEmbeddingProvider;
            eprintln!("Initializing embedding model for semantic scoring...");
            match OnnxEmbeddingProvider::new() {
                Ok(provider) => {
                    eprintln!("✓ Embedding model loaded successfully");
                    compressor = compressor
                        .with_embeddings(Box::new(provider))
                        .map_err(|e| {
                            AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                                "Failed to initialize embeddings: {}",
                                e
                            )))
                        })?;
                }
                Err(e) => {
                    eprintln!("⚠️  Warning: Embeddings not available ({}), falling back to heuristic scoring", e);
                    eprintln!("   Run 'tokuin setup models' to download embedding models.");
                    eprintln!("   Continuing with heuristic scoring...");
                }
            }
        }

        let state_file_path = if incremental {
            Some(
                previous
                    .clone()
                    .unwrap_or_else(|| format!("{}.state.json", input)),
            )
        } else {
            None
        };

        let previous_state = if let Some(ref path) = state_file_path {
            if Path::new(path).exists() {
                eprintln!("Loading incremental state from: {}", path);
                let prev_json = fs::read_to_string(path).map_err(|e| {
                    AppError::Io(std::io::Error::other(format!(
                        "Failed to read state file: {}",
                        e
                    )))
                })?;
                let prev: crate::compression::types::CompressionResult =
                    serde_json::from_str(&prev_json)
                        .map_err(|e| AppError::Parse(crate::error::ParseError::InvalidJson(e)))?;
                Some(Box::new(prev))
            } else {
                None
            }
        } else {
            None
        };

        let config = CompressionConfig {
            level: compression_level,
            context_library_path: context_lib.clone(),
            force_inline: inline,
            structured_mode: structured,
            incremental_mode: incremental,
            anchor_threshold,
            retention_threshold,
            previous_result: previous_state.clone(),
            scoring_mode,
            ..Default::default()
        };

        eprintln!("Compressing: {}", input);
        let use_incremental = incremental && previous_state.is_some();
        if incremental && !use_incremental {
            eprintln!(
                "No incremental state found. Running full compression and seeding state file."
            );
        }

        let mut result = if use_incremental {
            eprintln!("Using incremental compression mode");
            compressor.compress_incremental(&prompt, &config)?
        } else {
            compressor.compress(&prompt, &config)?
        };

        // Calculate quality metrics if requested
        if quality {
            eprintln!("\nCalculating quality metrics...");
            match compressor.calculate_quality_metrics(&prompt, &result) {
                Ok(metrics) => {
                    result.quality_metrics = Some(metrics.clone());
                }
                Err(e) => {
                    eprintln!("⚠️  Warning: Failed to calculate quality metrics: {}", e);
                    eprintln!(
                        "   Compression completed successfully, but quality check was skipped."
                    );
                }
            }
        }

        let hieratic_encoder = HieraticEncoder::new();
        let hieratic_output = hieratic_encoder.encode(&result.document)?;

        let output_content = match format {
            CompressionOutputFormat::Hieratic => hieratic_output.clone(),
            CompressionOutputFormat::Expanded => prompt.clone(),
            CompressionOutputFormat::Json => serde_json::to_string_pretty(&result)
                .map_err(|e| AppError::Parse(crate::error::ParseError::InvalidJson(e)))?,
        };

        let output_path = output.unwrap_or_else(|| format!("{}.hieratic", input));
        fs::write(&output_path, output_content).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to write output: {}",
                e
            )))
        })?;

        eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("Compression Summary:");
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("Original:  {} tokens", result.original_tokens);
        eprintln!("Compressed: {} tokens", result.compressed_tokens);
        eprintln!(
            "Reduction: {:.1}% ({} tokens saved)",
            result.compression_percentage(),
            result.tokens_saved
        );

        if incremental && !result.anchors.is_empty() {
            eprintln!("\nIncremental Mode:");
            eprintln!("  Anchors: {}", result.anchors.len());
            eprintln!(
                "  Anchor tokens: {}",
                result
                    .anchors
                    .iter()
                    .map(|a| a.summary_tokens)
                    .sum::<usize>()
            );
            eprintln!("  Retained tokens: {}", retention_threshold);
        }

        eprintln!("\nOutput: {}", output_path);
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Display quality metrics if calculated
        if let Some(ref metrics) = result.quality_metrics {
            eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            eprintln!("Quality Metrics:");
            eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            eprintln!(
                "Overall Score: {:.1}% ({})",
                metrics.overall_score * 100.0,
                metrics.rating()
            );
            eprintln!(
                "  ├─ Semantic Similarity: {:.1}%",
                metrics.semantic_similarity * 100.0
            );
            eprintln!(
                "  ├─ Critical Instructions: {}/{} preserved ({:.1}%)",
                metrics.critical_patterns_preserved,
                metrics.critical_patterns_found,
                metrics.critical_instruction_preservation * 100.0
            );
            eprintln!(
                "  ├─ Information Retention: {:.1}%",
                metrics.information_retention * 100.0
            );
            eprintln!(
                "  └─ Structural Integrity: {:.1}%",
                metrics.structural_integrity * 100.0
            );

            if metrics.is_acceptable() {
                eprintln!("\n✅ Quality is acceptable (>= 70%)");
            } else {
                eprintln!("\n⚠️  Warning: Quality is below recommended threshold (< 70%)");
                eprintln!("   Consider using --level light or reviewing the compressed output.");
            }
            eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }

        if incremental {
            if let Some(ref path) = state_file_path {
                let mut state_result = result.clone();
                if state_result.anchors.is_empty() {
                    let hieratic_encoder = HieraticEncoder::new();
                    let hieratic_output = hieratic_encoder.encode(&result.document)?;
                    let anchor = CompressionAnchor::from_base_document(
                        hieratic_output,
                        result.compressed_tokens,
                        result.original_tokens,
                    );
                    state_result.anchors.push(anchor);
                }
                let state_json = serde_json::to_string_pretty(&state_result)
                    .map_err(|e| AppError::Parse(crate::error::ParseError::InvalidJson(e)))?;
                fs::write(path, state_json).map_err(|e| {
                    AppError::Io(std::io::Error::other(format!(
                        "Failed to write state file: {}",
                        e
                    )))
                })?;
                eprintln!("\nState saved to: {}", path);
            }
        }

        Ok(())
    }

    #[cfg(all(feature = "compression", feature = "load-test"))]
    #[allow(clippy::too_many_arguments)]
    fn run_compress(
        input: String,
        output: Option<String>,
        level: CompressionLevelArg,
        context_lib: Option<String>,
        inline: bool,
        structured: bool,
        incremental: bool,
        previous: Option<String>,
        anchor_threshold: usize,
        retention_threshold: usize,
        model: String,
        format: CompressionOutputFormat,
        scoring: CompressionScoringMode,
        quality: bool,
        llm_judge: bool,
        evaluation_model: Option<String>,
        judge_model: String,
        judge_api_key: Option<String>,
        judge_provider: JudgeProvider,
    ) -> Result<(), AppError> {
        Self::run_compress_internal(
            input,
            output,
            level,
            context_lib,
            inline,
            structured,
            incremental,
            previous,
            anchor_threshold,
            retention_threshold,
            model,
            format,
            scoring,
            quality,
            llm_judge,
            evaluation_model,
            Some(judge_model),
            judge_api_key,
            Some(judge_provider),
        )
    }

    #[cfg(all(feature = "compression", feature = "load-test"))]
    #[allow(clippy::too_many_arguments)]
    fn run_compress_internal(
        input: String,
        output: Option<String>,
        level: CompressionLevelArg,
        context_lib: Option<String>,
        inline: bool,
        structured: bool,
        incremental: bool,
        previous: Option<String>,
        anchor_threshold: usize,
        retention_threshold: usize,
        model: String,
        format: CompressionOutputFormat,
        scoring: CompressionScoringMode,
        quality: bool,
        llm_judge: bool,
        evaluation_model: Option<String>,
        judge_model: Option<String>,
        judge_api_key: Option<String>,
        judge_provider: Option<JudgeProvider>,
    ) -> Result<(), AppError> {
        use crate::compression::compressor::Compressor;
        use crate::compression::context_library::ContextLibraryManager;
        use crate::compression::hieratic_encoder::HieraticEncoder;
        use crate::compression::types::{
            CompressionAnchor, CompressionConfig, CompressionLevel, ScoringMode,
        };
        use std::fs;

        let registry = ModelRegistry::new_with_pricing(None).map_err(AppError::Model)?;
        let tokenizer = registry.get_tokenizer(&model)?;
        let prompt = fs::read_to_string(&input).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to read input file: {}",
                e
            )))
        })?;

        let mut compressor = Compressor::new(tokenizer);

        if !inline {
            if let Some(ref lib_path) = context_lib {
                if std::path::Path::new(lib_path).exists() {
                    let library = ContextLibraryManager::load_from_file(lib_path)?;
                    compressor = compressor.with_context_library(library);
                }
            } else if std::path::Path::new("contexts.toml").exists() {
                let library = ContextLibraryManager::load_from_file("contexts.toml")?;
                compressor = compressor.with_context_library(library);
            }
        }

        let compression_level = match level {
            CompressionLevelArg::Light => CompressionLevel::Light,
            CompressionLevelArg::Medium => CompressionLevel::Medium,
            CompressionLevelArg::Aggressive => CompressionLevel::Aggressive,
        };

        let scoring_mode = match scoring {
            CompressionScoringMode::Heuristic => ScoringMode::Heuristic,
            CompressionScoringMode::Semantic => ScoringMode::Semantic,
            CompressionScoringMode::Hybrid => ScoringMode::Hybrid,
        };

        // Initialize embeddings if semantic or hybrid mode is selected
        #[cfg(feature = "compression-embeddings")]
        if matches!(scoring_mode, ScoringMode::Semantic | ScoringMode::Hybrid) {
            use crate::compression::embeddings::OnnxEmbeddingProvider;
            eprintln!("Initializing embedding model for semantic scoring...");
            match OnnxEmbeddingProvider::new() {
                Ok(provider) => {
                    eprintln!("✓ Embedding model loaded successfully");
                    compressor = compressor
                        .with_embeddings(Box::new(provider))
                        .map_err(|e| {
                            AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                                "Failed to initialize embeddings: {}",
                                e
                            )))
                        })?;
                }
                Err(e) => {
                    eprintln!("⚠️  Warning: Embeddings not available ({}), falling back to heuristic scoring", e);
                    eprintln!("   Run 'tokuin setup models' to download embedding models.");
                    eprintln!("   Continuing with heuristic scoring...");
                }
            }
        }

        let state_file_path = if incremental {
            Some(
                previous
                    .clone()
                    .unwrap_or_else(|| format!("{}.state.json", input)),
            )
        } else {
            None
        };

        let previous_state = if let Some(ref path) = state_file_path {
            if Path::new(path).exists() {
                eprintln!("Loading incremental state from: {}", path);
                let prev_json = fs::read_to_string(path).map_err(|e| {
                    AppError::Io(std::io::Error::other(format!(
                        "Failed to read state file: {}",
                        e
                    )))
                })?;
                let prev: crate::compression::types::CompressionResult =
                    serde_json::from_str(&prev_json)
                        .map_err(|e| AppError::Parse(crate::error::ParseError::InvalidJson(e)))?;
                Some(Box::new(prev))
            } else {
                None
            }
        } else {
            None
        };

        let config = CompressionConfig {
            level: compression_level,
            context_library_path: context_lib.clone(),
            force_inline: inline,
            structured_mode: structured,
            incremental_mode: incremental,
            anchor_threshold,
            retention_threshold,
            previous_result: previous_state.clone(),
            scoring_mode,
            ..Default::default()
        };

        eprintln!("Compressing: {}", input);
        let use_incremental = incremental && previous_state.is_some();
        if incremental && !use_incremental {
            eprintln!(
                "No incremental state found. Running full compression and seeding state file."
            );
        }

        let mut result = if use_incremental {
            eprintln!("Using incremental compression mode");
            compressor.compress_incremental(&prompt, &config)?
        } else {
            compressor.compress(&prompt, &config)?
        };

        // Calculate quality metrics if requested (before JSON serialization)
        if quality {
            eprintln!("\nCalculating quality metrics...");

            #[cfg(all(feature = "compression", feature = "load-test"))]
            let metrics_result = if llm_judge {
                // Initialize LLM client for judge evaluation
                // Default evaluation model: use provided model or fallback to Claude 3 Opus
                let eval_model = evaluation_model.as_deref().unwrap_or_else(|| {
                    if model.is_empty() || model == "gpt-4" {
                        "anthropic/claude-3-opus"
                    } else {
                        &model
                    }
                });
                // Get judge model (unwrap since it has a default value)
                let judge_model_val = judge_model.as_deref().unwrap_or("anthropic/claude-3-opus");

                // Get API key
                let api_key = judge_api_key
                    .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                    .ok_or_else(|| AppError::Config(
                        "API key required for LLM judge. Set OPENROUTER_API_KEY environment variable or use --judge-api-key".to_string()
                    ))?;

                // Determine provider
                let provider = judge_provider.unwrap_or(JudgeProvider::Openrouter);

                // Create client
                use crate::http::client::{ClientConfig, LlmClientEnum};
                use crate::http::providers::{
                    anthropic::AnthropicClient, openai::OpenAIClient, openrouter::OpenRouterClient,
                };
                use std::time::Duration;

                let client_config = ClientConfig {
                    endpoint: String::new(),
                    api_key: api_key.clone(),
                    timeout: Duration::from_secs(60),
                    headers: Vec::new(),
                };

                let client: Box<dyn crate::http::client::LlmClient> = match provider {
                    JudgeProvider::Openrouter => Box::new(LlmClientEnum::OpenRouter(
                        OpenRouterClient::new(client_config)?,
                    )),
                    JudgeProvider::Openai => {
                        Box::new(LlmClientEnum::OpenAI(OpenAIClient::new(client_config)?))
                    }
                    JudgeProvider::Anthropic => Box::new(LlmClientEnum::Anthropic(
                        AnthropicClient::new(client_config)?,
                    )),
                };

                eprintln!("Generating output from original prompt...");
                eprintln!("Generating output from compressed prompt...");
                eprintln!("Evaluating outputs with LLM judge...");

                // Create tokio runtime for async operations
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    AppError::Config(format!("Failed to create async runtime: {}", e))
                })?;

                // Normalize model names
                use crate::compression::llm_judge::normalize_model_name;
                let eval_model_norm = normalize_model_name(eval_model);
                let judge_model_norm = normalize_model_name(judge_model_val);

                rt.block_on(compressor.calculate_quality_metrics(
                    &prompt,
                    &result,
                    Some(&eval_model_norm),
                    Some(&judge_model_norm),
                    Some(client.as_ref()),
                ))
            } else {
                // Synchronous version without LLM judge
                #[cfg(not(all(feature = "compression", feature = "load-test")))]
                {
                    compressor.calculate_quality_metrics(&prompt, &result)
                }
                #[cfg(all(feature = "compression", feature = "load-test"))]
                {
                    let rt = tokio::runtime::Runtime::new().map_err(|e| {
                        AppError::Config(format!("Failed to create async runtime: {}", e))
                    })?;
                    rt.block_on(
                        compressor.calculate_quality_metrics(&prompt, &result, None, None, None),
                    )
                }
            };

            #[cfg(not(all(feature = "compression", feature = "load-test")))]
            let metrics_result = compressor.calculate_quality_metrics(&prompt, &result);

            match metrics_result {
                Ok(metrics) => {
                    // Store metrics in result for JSON output
                    result.quality_metrics = Some(metrics.clone());
                    // Display will happen later after file write
                }
                Err(e) => {
                    eprintln!("⚠️  Warning: Failed to calculate quality metrics: {}", e);
                    eprintln!(
                        "   Compression completed successfully, but quality check was skipped."
                    );
                }
            }
        }

        let hieratic_encoder = HieraticEncoder::new();
        let hieratic_output = hieratic_encoder.encode(&result.document)?;

        let output_content = match format {
            CompressionOutputFormat::Hieratic => hieratic_output.clone(),
            CompressionOutputFormat::Expanded => {
                // For expanded, we just output the original
                prompt.clone()
            }
            CompressionOutputFormat::Json => serde_json::to_string_pretty(&result)
                .map_err(|e| AppError::Parse(crate::error::ParseError::InvalidJson(e)))?,
        };

        let output_path = output.unwrap_or_else(|| format!("{}.hieratic", input));
        fs::write(&output_path, output_content).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to write output: {}",
                e
            )))
        })?;

        eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("Compression Summary:");
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("Original:  {} tokens", result.original_tokens);
        eprintln!("Compressed: {} tokens", result.compressed_tokens);
        eprintln!(
            "Reduction: {:.1}% ({} tokens saved)",
            result.compression_percentage(),
            result.tokens_saved
        );

        if incremental && !result.anchors.is_empty() {
            eprintln!("\nIncremental Mode:");
            eprintln!("  Anchors: {}", result.anchors.len());
            eprintln!(
                "  Anchor tokens: {}",
                result
                    .anchors
                    .iter()
                    .map(|a| a.summary_tokens)
                    .sum::<usize>()
            );
            eprintln!("  Retained tokens: {}", retention_threshold);
        }

        eprintln!("\nOutput: {}", output_path);
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Display quality metrics if calculated
        if let Some(ref metrics) = result.quality_metrics {
            eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            eprintln!("Quality Metrics:");
            eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            eprintln!(
                "Overall Score: {:.1}% ({})",
                metrics.overall_score * 100.0,
                metrics.rating()
            );
            eprintln!(
                "  ├─ Semantic Similarity: {:.1}%",
                metrics.semantic_similarity * 100.0
            );
            eprintln!(
                "  ├─ Critical Instructions: {}/{} preserved ({:.1}%)",
                metrics.critical_patterns_preserved,
                metrics.critical_patterns_found,
                metrics.critical_instruction_preservation * 100.0
            );
            eprintln!(
                "  ├─ Information Retention: {:.1}%",
                metrics.information_retention * 100.0
            );
            eprintln!(
                "  └─ Structural Integrity: {:.1}%",
                metrics.structural_integrity * 100.0
            );

            if metrics.is_acceptable() {
                eprintln!("\n✅ Quality is acceptable (>= 70%)");
            } else {
                eprintln!("\n⚠️  Warning: Quality is below recommended threshold (< 70%)");
                eprintln!("   Consider using --level light or reviewing the compressed output.");
            }
            eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // Display LLM judge metrics if available
            #[cfg(all(feature = "compression", feature = "load-test"))]
            if let Some(ref llm_judge) = metrics.llm_judge {
                eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                eprintln!("LLM Judge Evaluation (Output Comparison):");
                eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                eprintln!("Provider: OpenRouter");
                eprintln!("Evaluation Model: {}", llm_judge.evaluation_model);
                eprintln!("Judge Model: {}", llm_judge.judge_model);
                eprintln!();
                eprintln!(
                    "Output Equivalence: {:.0}/100",
                    llm_judge.output_equivalence
                );
                eprintln!(
                    "Instruction Compliance: {:.0}/100",
                    llm_judge.instruction_compliance
                );
                eprintln!(
                    "Information Completeness: {:.0}/100",
                    llm_judge.information_completeness
                );
                eprintln!(
                    "Quality Preservation: {:.0}/100",
                    llm_judge.quality_preservation
                );
                eprintln!(
                    "Overall Fidelity: {:.0}/100 ({})",
                    llm_judge.overall_fidelity,
                    llm_judge.rating()
                );
                eprintln!();
                eprintln!("Justification: {}", llm_judge.justification);

                if !llm_judge.key_differences.is_empty() {
                    eprintln!();
                    eprintln!("Key Differences:");
                    for diff in &llm_judge.key_differences {
                        eprintln!("  - {}", diff);
                    }
                }

                if let Some(cost) = llm_judge.evaluation_cost {
                    eprintln!();
                    eprintln!("Evaluation Cost: ${:.4}", cost);
                }

                if llm_judge.is_acceptable() {
                    eprintln!("\n✅ LLM judge evaluation indicates acceptable quality (>= 70%)");
                } else {
                    eprintln!(
                        "\n⚠️  LLM judge evaluation indicates quality below threshold (< 70%)"
                    );
                }
                eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            }
        }

        if incremental {
            if let Some(ref path) = state_file_path {
                let mut state_result = result.clone();
                if state_result.anchors.is_empty() {
                    let anchor = CompressionAnchor::from_base_document(
                        hieratic_output.clone(),
                        result.compressed_tokens,
                        result.original_tokens,
                    );
                    state_result.anchors.push(anchor);
                }
                let json_content = serde_json::to_string_pretty(&state_result)
                    .map_err(|e| AppError::Parse(crate::error::ParseError::InvalidJson(e)))?;
                fs::write(path, json_content).map_err(|e| {
                    AppError::Io(std::io::Error::other(format!(
                        "Failed to write state file: {}",
                        e
                    )))
                })?;
                eprintln!("Incremental state saved to: {}", path);
            }
        } else if format == CompressionOutputFormat::Json {
            let json_path = format!("{}.json", output_path);
            let json_content = serde_json::to_string_pretty(&result)
                .map_err(|e| AppError::Parse(crate::error::ParseError::InvalidJson(e)))?;
            fs::write(&json_path, json_content).map_err(|e| {
                AppError::Io(std::io::Error::other(format!(
                    "Failed to write JSON: {}",
                    e
                )))
            })?;
        }

        Ok(())
    }

    /// Expand a Hieratic file back to full prompt
    #[cfg(feature = "compression")]
    fn run_expand(
        input: String,
        output: Option<String>,
        context_lib: Option<String>,
    ) -> Result<(), AppError> {
        use crate::compression::hieratic_decoder::HieraticDecoder;
        use std::fs;

        let hieratic_content = fs::read_to_string(&input).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to read input file: {}",
                e
            )))
        })?;

        let mut decoder = HieraticDecoder::new()?;

        if let Some(ref lib_path) = context_lib {
            decoder = decoder.load_context_library(lib_path)?;
        } else if std::path::Path::new("contexts.toml").exists() {
            decoder = decoder.load_context_library("contexts.toml")?;
        }

        eprintln!("Expanding: {}", input);
        let expanded = decoder.decode(&hieratic_content)?;

        if let Some(output_path) = output {
            fs::write(&output_path, &expanded).map_err(|e| {
                AppError::Io(std::io::Error::other(format!(
                    "Failed to write output: {}",
                    e
                )))
            })?;
            eprintln!("Expanded prompt saved to: {}", output_path);
        } else {
            println!("{}", expanded);
        }

        Ok(())
    }

    /// Setup embedding models
    #[cfg(feature = "compression-embeddings")]
    fn run_setup_models(force: bool, onnx: bool) -> Result<(), AppError> {
        use crate::compression::embeddings::download_models;
        #[cfg(feature = "load-test")]
        use indicatif::{ProgressBar, ProgressStyle};

        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("Setting up embedding models...");
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        #[cfg(feature = "load-test")]
        let progress_callback: Option<crate::compression::embeddings::ProgressCallback> = {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg}")
                    .expect("valid spinner template"),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));

            Some(Box::new(move |msg: &str| {
                pb.set_message(msg.to_string());
            }))
        };

        #[cfg(not(feature = "load-test"))]
        let progress_callback: Option<Box<dyn Fn(&str) + Send + Sync>> = None;

        match download_models(force, onnx, progress_callback) {
            Ok(model_path) => {
                eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                eprintln!("✓ Models setup complete!");
                eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                eprintln!("Models location: {:?}", model_path);
                eprintln!("\nYou can now use --scoring semantic or --scoring hybrid");
                Ok(())
            }
            Err(e) => {
                eprintln!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                eprintln!("⚠️  Setup failed: {}", e);
                eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                eprintln!("\nTroubleshooting:");
                eprintln!("1. Check your internet connection");
                eprintln!(
                    "2. Ensure hf-hub feature is enabled (--features compression-embeddings)"
                );
                if onnx {
                    eprintln!("3. For ONNX model, you may need to convert manually:");
                    eprintln!("   pip install optimum[exporters]");
                    eprintln!("   optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 ./onnx_model/");
                }
                Err(e)
            }
        }
    }

    /// Get input from file, stdin, or argument.
    fn get_input(input: &Option<String>) -> Result<String, AppError> {
        if let Some(input) = input {
            if input == "-" {
                // Read from stdin
                let mut buffer = String::new();
                io::stdin().read_to_string(&mut buffer)?;
                Ok(buffer)
            } else {
                // Read from file
                std::fs::read_to_string(input).map_err(|e| {
                    AppError::Io(std::io::Error::other(format!(
                        "Failed to read file '{}': {}",
                        input, e
                    )))
                })
            }
        } else {
            // Read from stdin (if no argument provided)
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            Ok(buffer)
        }
    }

    /// Count tokens for messages using the specified tokenizer.
    fn count_tokens(
        tokenizer: &dyn Tokenizer,
        messages: &[crate::parsers::Message],
        model_name: &str,
        breakdown: bool,
        price: bool,
        pricing_override: Option<(f64, f64)>,
    ) -> Result<TokenResult, AppError> {
        let mut total = 0;
        let mut token_breakdown = if breakdown {
            Some(TokenBreakdown::new())
        } else {
            None
        };

        for message in messages {
            let count = tokenizer.count_tokens(&message.content)?;
            total += count;

            if let Some(ref mut bd) = token_breakdown {
                match message.role.as_str() {
                    "system" => bd.system += count,
                    "user" => bd.user += count,
                    "assistant" => bd.assistant += count,
                    _ => {}
                }
            }
        }

        if let Some(ref mut bd) = token_breakdown {
            bd.total = total;
        }

        // Calculate costs
        let override_input = pricing_override.map(|p| p.0);
        let override_output = pricing_override.map(|p| p.1);

        let input_cost = if price {
            override_input
                .or_else(|| tokenizer.input_price_per_1k())
                .map(|rate| (total as f64 / 1000.0) * rate)
        } else {
            None
        };

        let output_cost = if price {
            override_output
                .or_else(|| tokenizer.output_price_per_1k())
                .map(|rate| (total as f64 / 1000.0) * rate)
        } else {
            None
        };

        Ok(TokenResult {
            model: model_name.to_string(),
            tokens: total,
            input_cost,
            output_cost,
            breakdown: token_breakdown,
        })
    }

    /// Run in diff mode, comparing two prompts.
    fn run_diff(args: &EstimateArgs, diff_file: &str) -> Result<(), AppError> {
        let registry = ModelRegistry::new_with_pricing(args.pricing_file.as_deref())
            .map_err(AppError::Model)?;

        // Get both inputs
        let input1 = Self::get_input(&args.input)?;
        let input2 = std::fs::read_to_string(diff_file).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to read diff file '{}': {}",
                diff_file, e
            )))
        })?;

        // Determine model
        let model = args.model.as_ref().ok_or_else(|| {
            AppError::Parse(crate::error::ParseError::InvalidFormat(
                "Model required for diff mode. Use --model".to_string(),
            ))
        })?;

        let tokenizer = registry.get_tokenizer(model)?;

        // Parse both inputs
        let parser1: Box<dyn InputParser> =
            if input1.trim_start().starts_with('{') || input1.trim_start().starts_with('[') {
                Box::new(JsonParser::new())
            } else {
                Box::new(TextParser::new())
            };

        let parser2: Box<dyn InputParser> =
            if input2.trim_start().starts_with('{') || input2.trim_start().starts_with('[') {
                Box::new(JsonParser::new())
            } else {
                Box::new(TextParser::new())
            };

        let messages1 = parser1.parse(&input1)?;
        let messages2 = parser2.parse(&input2)?;

        // Count tokens for both
        let pricing_override = if args.price {
            registry.pricing_for(model)
        } else {
            None
        };

        let result1 = Self::count_tokens(
            &*tokenizer,
            &messages1,
            model,
            false,
            args.price,
            pricing_override,
        )?;
        let result2 = Self::count_tokens(
            &*tokenizer,
            &messages2,
            model,
            false,
            args.price,
            pricing_override,
        )?;

        // Show diff
        let diff = result2.tokens as i64 - result1.tokens as i64;
        println!("Model: {}", model);
        println!("Original: {} tokens", result1.tokens);
        println!("Modified: {} tokens", result2.tokens);
        println!(
            "Difference: {}{} tokens",
            if diff >= 0 { "+" } else { "" },
            diff
        );

        if args.price {
            if let (Some(cost1), Some(cost2)) = (result1.input_cost, result2.input_cost) {
                let cost_diff = cost2 - cost1;
                println!("Cost difference: ${:.4}", cost_diff.abs());
            }
        }

        Ok(())
    }

    /// Run in watch mode, monitoring file for changes.
    #[cfg(feature = "watch")]
    fn run_watch(args: &EstimateArgs) -> Result<(), AppError> {
        use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
        use std::sync::mpsc;

        let input_file = args.input.as_ref().ok_or_else(|| {
            AppError::Parse(crate::error::ParseError::InvalidFormat(
                "File path required for watch mode".to_string(),
            ))
        })?;

        if input_file == "-" {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "Cannot watch stdin. Provide a file path.".to_string(),
            )));
        }

        let path = PathBuf::from(input_file);
        if !path.exists() {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                format!("File not found: {}", input_file),
            )));
        }

        println!(
            "Watching '{}' for changes. Press Ctrl+C to stop.",
            input_file
        );

        // Create channel for file events
        let (tx, rx) = mpsc::channel();

        // Create watcher with config
        let config = Config::default().with_poll_interval(Duration::from_secs(1));
        let mut watcher = RecommendedWatcher::new(tx, config).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to create file watcher: {}",
                e
            )))
        })?;

        watcher
            .watch(&path, RecursiveMode::NonRecursive)
            .map_err(|e| {
                AppError::Io(std::io::Error::other(format!(
                    "Failed to watch file: {}",
                    e
                )))
            })?;

        // Run initial analysis
        Self::run_estimate_once(args)?;

        // Watch for changes
        loop {
            match rx.recv() {
                Ok(Ok(event)) => {
                    if event.kind.is_modify() {
                        println!("\n--- File changed, re-analyzing ---\n");
                        if let Err(e) = Self::run_estimate_once(args) {
                            eprintln!("Error: {}", e);
                        }
                    }
                }
                Ok(Err(e)) => {
                    eprintln!("Watch error: {}", e);
                }
                Err(e) => {
                    eprintln!("Channel error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Run a single analysis (used by watch mode).
    #[cfg(feature = "watch")]
    fn run_estimate_once(args: &EstimateArgs) -> Result<(), AppError> {
        let mut new_args = args.clone();
        new_args.watch = false;
        Self::run_estimate(new_args)
    }
}

/// Estimate command arguments (for internal use).
#[derive(Debug, Clone)]
pub struct EstimateArgs {
    input: Option<String>,
    model: Option<String>,
    compare: Vec<String>,
    breakdown: bool,
    format: OutputFormat,
    price: bool,
    #[cfg(feature = "markdown")]
    minify: bool,
    diff: Option<String>,
    #[cfg(feature = "watch")]
    watch: bool,
    pricing_file: Option<String>,
}

#[cfg(feature = "watch")]
impl Default for EstimateArgs {
    fn default() -> Self {
        Self {
            input: None,
            model: None,
            compare: Vec::new(),
            breakdown: false,
            format: OutputFormat::Text,
            price: false,
            #[cfg(feature = "markdown")]
            minify: false,
            diff: None,
            watch: false,
            pricing_file: None,
        }
    }
}

/// Load test command arguments (for internal use).
#[cfg(feature = "load-test")]
#[derive(Debug, Clone)]
struct LoadTestArgs {
    model: String,
    endpoint: Option<String>,
    api_key: Option<String>,
    openai_api_key: Option<String>,
    anthropic_api_key: Option<String>,
    openrouter_api_key: Option<String>,
    provider: Option<Provider>,
    concurrency: usize,
    runs: usize,
    prompt_file: Option<String>,
    think_time: Option<String>,
    retry: u32,
    output_format: LoadTestOutputFormat,
    dry_run: bool,
    max_cost: Option<f64>,
    estimate_cost: bool,
    pricing_file: Option<String>,
}

#[cfg(feature = "load-test")]
impl From<Command> for LoadTestArgs {
    fn from(cmd: Command) -> Self {
        match cmd {
            Command::LoadTest {
                model,
                endpoint,
                api_key,
                openai_api_key,
                anthropic_api_key,
                openrouter_api_key,
                provider,
                concurrency,
                runs,
                prompt_file,
                think_time,
                retry,
                output_format,
                dry_run,
                max_cost,
                estimate_cost,
            } => Self {
                model,
                endpoint,
                api_key,
                openai_api_key,
                anthropic_api_key,
                openrouter_api_key,
                provider,
                concurrency,
                runs,
                prompt_file,
                think_time,
                retry,
                output_format,
                dry_run,
                max_cost,
                estimate_cost,
                pricing_file: None,
            },
            _ => panic!("Not a LoadTest command"),
        }
    }
}

#[cfg(feature = "load-test")]
impl LoadTestArgs {
    fn resolve_provider(&self) -> Provider {
        if let Some(provider) = &self.provider {
            return provider.clone();
        }

        let model_lower = self.model.to_lowercase();
        if model_lower.starts_with("claude-")
            || model_lower.contains("anthropic/")
            || model_lower.starts_with("anthropic/")
        {
            return Provider::Anthropic;
        }

        if model_lower.contains("openrouter/") {
            return Provider::Openrouter;
        }

        if let Some(endpoint) = &self.endpoint {
            let endpoint_lower = endpoint.to_lowercase();
            if endpoint_lower.contains("anthropic") {
                return Provider::Anthropic;
            }
            if endpoint_lower.contains("openrouter") {
                return Provider::Openrouter;
            }
        }

        if self.openrouter_api_key.is_some() {
            return Provider::Openrouter;
        }
        if self.anthropic_api_key.is_some() {
            return Provider::Anthropic;
        }

        Provider::Openai
    }
}

#[cfg(feature = "load-test")]
impl From<Command> for EstimateArgs {
    fn from(cmd: Command) -> Self {
        match cmd {
            Command::Estimate {
                input,
                model,
                compare,
                breakdown,
                format,
                price,
                #[cfg(feature = "markdown")]
                minify,
                diff,
                #[cfg(feature = "watch")]
                watch,
            } => Self {
                input,
                model,
                compare,
                breakdown,
                format,
                price,
                #[cfg(feature = "markdown")]
                minify,
                diff,
                #[cfg(feature = "watch")]
                watch,
                pricing_file: None,
            },
            _ => panic!("Not an Estimate command"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_legacy_estimate_arguments() {
        let cli = Cli::try_parse_from([
            "tokuin",
            "--model",
            "gpt-4",
            "--price",
            "--format",
            "json",
            "hello world",
        ])
        .expect("CLI args should parse");

        assert!(cli.command.is_none(), "legacy usage keeps flat args");
        assert_eq!(cli.model.as_deref(), Some("gpt-4"));
        assert_eq!(cli.input.as_deref(), Some("hello world"));
        assert!(cli.price);

        match cli.format {
            OutputFormat::Json => {}
            other => panic!("expected JSON format, got {:?}", other),
        }
    }

    #[test]
    fn parse_estimate_subcommand_arguments() {
        let cli = Cli::try_parse_from([
            "tokuin",
            "estimate",
            "--model",
            "gpt-4o-mini",
            "--compare",
            "gpt-4-turbo",
            "--compare",
            "gpt-3.5-turbo",
            "--breakdown",
            "prompt.txt",
        ])
        .expect("subcommand args should parse");

        let command = cli.command.expect("estimate subcommand expected");
        match command {
            Command::Estimate {
                input,
                model,
                compare,
                breakdown,
                format,
                price,
                #[cfg(feature = "markdown")]
                    minify: _,
                diff,
                #[cfg(feature = "watch")]
                    watch: _,
            } => {
                assert_eq!(input.as_deref(), Some("prompt.txt"));
                assert_eq!(model.as_deref(), Some("gpt-4o-mini"));
                assert_eq!(compare, vec!["gpt-4-turbo", "gpt-3.5-turbo"]);
                assert!(breakdown);
                assert!(!price);
                assert!(diff.is_none());
                match format {
                    OutputFormat::Text => {}
                    other => panic!("expected text format default, got {:?}", other),
                }
            }
            #[allow(unreachable_patterns)]
            _ => panic!("unexpected command variant"),
        }
    }

    #[cfg(feature = "load-test")]
    mod load_test_cli {
        use super::*;

        #[test]
        fn parse_load_test_arguments_with_defaults() {
            let cli = Cli::try_parse_from([
                "tokuin",
                "load-test",
                "--model",
                "openai/gpt-4",
                "--runs",
                "5",
                "--concurrency",
                "2",
                "--api-key",
                "sk-test",
                "--dry-run",
            ])
            .expect("load-test args should parse");

            let command = cli.command.expect("load-test subcommand expected");
            match command {
                Command::LoadTest {
                    model,
                    endpoint,
                    api_key,
                    openai_api_key,
                    anthropic_api_key,
                    openrouter_api_key,
                    provider,
                    concurrency,
                    runs,
                    prompt_file,
                    think_time,
                    retry,
                    output_format,
                    dry_run,
                    max_cost,
                    estimate_cost,
                } => {
                    assert_eq!(model, "openai/gpt-4");
                    assert!(endpoint.is_none());
                    assert_eq!(api_key.as_deref(), Some("sk-test"));
                    assert!(openai_api_key.is_none());
                    assert!(anthropic_api_key.is_none());
                    assert!(openrouter_api_key.is_none());
                    assert!(provider.is_none());
                    assert_eq!(concurrency, 2);
                    assert_eq!(runs, 5);
                    assert!(prompt_file.is_none());
                    assert!(think_time.is_none());
                    assert_eq!(retry, 3, "default retry should be 3");
                    assert!(dry_run, "flag should enable dry-run mode");
                    assert!(max_cost.is_none());
                    assert!(!estimate_cost);

                    match output_format {
                        LoadTestOutputFormat::Text => {}
                        other => panic!("expected text output, got {:?}", other),
                    }
                }
                #[allow(unreachable_patterns)]
                _ => panic!("unexpected command variant"),
            }
        }

        #[test]
        fn parse_load_test_arguments_with_overrides() {
            let cli = Cli::try_parse_from([
                "tokuin",
                "load-test",
                "--model",
                "openrouter/anthropic-sonnet",
                "--runs",
                "10",
                "--concurrency",
                "4",
                "--openrouter-api-key",
                "sk-or",
                "--think-time",
                "250-500ms",
                "--retry",
                "5",
                "--output-format",
                "json",
                "--estimate-cost",
            ])
            .expect("load-test args should parse with overrides");

            let command = cli.command.expect("load-test subcommand expected");
            match command {
                Command::LoadTest {
                    model,
                    endpoint,
                    api_key,
                    openai_api_key,
                    anthropic_api_key,
                    openrouter_api_key,
                    provider,
                    concurrency,
                    runs,
                    prompt_file,
                    think_time,
                    retry,
                    output_format,
                    dry_run,
                    max_cost,
                    estimate_cost,
                } => {
                    assert_eq!(model, "openrouter/anthropic-sonnet");
                    assert!(endpoint.is_none());
                    assert!(api_key.is_none());
                    assert!(openai_api_key.is_none());
                    assert!(anthropic_api_key.is_none());
                    assert_eq!(openrouter_api_key.as_deref(), Some("sk-or"));
                    assert!(provider.is_none());
                    assert_eq!(concurrency, 4);
                    assert_eq!(runs, 10);
                    assert!(prompt_file.is_none());
                    assert_eq!(think_time.as_deref(), Some("250-500ms"));
                    assert_eq!(retry, 5);
                    assert!(!dry_run);
                    assert!(max_cost.is_none());
                    assert!(estimate_cost);

                    match output_format {
                        LoadTestOutputFormat::Json => {}
                        other => panic!("expected JSON output, got {:?}", other),
                    }
                }
                #[allow(unreachable_patterns)]
                _ => panic!("unexpected command variant"),
            }
        }

        #[test]
        fn parse_load_test_with_explicit_provider() {
            let cli = Cli::try_parse_from([
                "tokuin",
                "load-test",
                "--model",
                "claude-3-sonnet",
                "--provider",
                "anthropic",
                "--anthropic-api-key",
                "sk-anthropic",
                "--runs",
                "2",
                "--concurrency",
                "1",
            ])
            .expect("load-test args should parse with provider flag");

            let command = cli.command.expect("load-test subcommand expected");
            if let Command::LoadTest { provider, .. } = &command {
                assert_eq!(provider.as_ref(), Some(&Provider::Anthropic));
            } else {
                panic!("unexpected command variant");
            }

            let args = LoadTestArgs::from(command);
            assert_eq!(args.resolve_provider(), Provider::Anthropic);
        }

        #[test]
        fn resolve_provider_from_model_heuristic() {
            let cli = Cli::try_parse_from([
                "tokuin",
                "load-test",
                "--model",
                "anthropic/claude-3-5-sonnet",
                "--runs",
                "1",
                "--concurrency",
                "1",
                "--anthropic-api-key",
                "sk-anthropic",
            ])
            .expect("load-test args should parse");

            let command = cli.command.expect("load-test subcommand expected");
            let args = LoadTestArgs::from(command);
            assert_eq!(args.resolve_provider(), Provider::Anthropic);
        }
    }
}
