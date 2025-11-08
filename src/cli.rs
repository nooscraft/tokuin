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
#[cfg(feature = "watch")]
use std::path::PathBuf;
#[cfg(feature = "watch")]
use std::time::Duration;

/// Tokuin - Estimate token usage and costs for LLM prompts.
#[derive(Parser, Debug)]
#[command(name = "tokuin")]
#[command(about = "A fast CLI tool to estimate token usage and API costs for LLM prompts")]
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
                };
                Self::run_load_test(load_args)
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

        let registry = ModelRegistry::new();

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
            let result =
                Self::count_tokens(&*tokenizer, &messages, &tokenizer_name, breakdown, price)?;
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
                    .unwrap()
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
        Self::display_load_test_results(&results, &model, &args.output_format, estimate_cost)?;

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
    ) -> Result<(), AppError> {
        let successful = results.iter().filter(|r| r.success).count();
        let failed = results.len() - successful;
        let success_rate = (successful as f64 / results.len() as f64) * 100.0;

        let latencies: Vec<u64> = results
            .iter()
            .filter_map(|r| if r.success { Some(r.latency_ms) } else { None })
            .collect();

        let total_tokens_reported: usize = results.iter().filter_map(|r| r.total_tokens).sum();

        let avg_latency = if !latencies.is_empty() {
            latencies.iter().sum::<u64>() as f64 / latencies.len() as f64
        } else {
            0.0
        };

        let p50 = if !latencies.is_empty() {
            let mut sorted = latencies.clone();
            sorted.sort();
            sorted[sorted.len() / 2]
        } else {
            0
        };

        let p95 = if !latencies.is_empty() {
            let mut sorted = latencies.clone();
            sorted.sort();
            sorted[(sorted.len() as f64 * 0.95) as usize]
        } else {
            0
        };

        match output_format {
            LoadTestOutputFormat::Text => {
                println!("\n=== Load Test Results ===");
                println!("Total Requests: {}", results.len());
                println!("Successful: {} ({:.1}%)", successful, success_rate);
                println!("Failed: {} ({:.1}%)", failed, 100.0 - success_rate);
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
                    let registry = ModelRegistry::new();
                    if let Ok(tokenizer) = registry.get_tokenizer(model) {
                        let input_tokens: usize =
                            results.iter().filter_map(|r| r.input_tokens).sum();
                        let output_tokens: usize =
                            results.iter().filter_map(|r| r.output_tokens).sum();

                        let input_cost = tokenizer
                            .input_price_per_1k()
                            .map(|price| (input_tokens as f64 / 1000.0) * price);
                        let output_cost = tokenizer
                            .output_price_per_1k()
                            .map(|price| (output_tokens as f64 / 1000.0) * price);

                        if let (Some(in_cost), Some(out_cost)) = (input_cost, output_cost) {
                            let total_cost = in_cost + out_cost;
                            println!("\nCost Estimation:");
                            println!("  Input tokens: {}", input_tokens);
                            println!("  Output tokens: {}", output_tokens);
                            println!("  Input cost: ${:.6}", in_cost);
                            println!("  Output cost: ${:.6}", out_cost);
                            println!("  Total cost: ${:.6}", total_cost);
                        }
                    }
                }
            }
            LoadTestOutputFormat::Json => {
                let json = serde_json::json!({
                    "total_requests": results.len(),
                    "successful": successful,
                    "failed": failed,
                    "success_rate": success_rate,
                    "latency_ms": {
                        "average": avg_latency,
                        "p50": p50,
                        "p95": p95
                    },
                    "tokens": {
                        "total_reported": total_tokens_reported
                    }
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json).map_err(AppError::Json)?
                );
            }
            _ => {
                println!("Format {:?} not yet implemented", output_format);
            }
        }

        Ok(())
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
        let input_cost = if price {
            tokenizer
                .input_price_per_1k()
                .map(|price| (total as f64 / 1000.0) * price)
        } else {
            None
        };

        let output_cost = if price {
            tokenizer
                .output_price_per_1k()
                .map(|price| (total as f64 / 1000.0) * price)
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
        let registry = ModelRegistry::new();

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
        let result1 = Self::count_tokens(&*tokenizer, &messages1, model, false, args.price)?;
        let result2 = Self::count_tokens(&*tokenizer, &messages2, model, false, args.price)?;

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
