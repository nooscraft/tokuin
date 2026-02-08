/// Tokuin - Token usage and cost estimator for LLM prompts.
///
/// A fast CLI tool to estimate token usage and API costs for LLM prompts.
mod analyzers;
mod cli;
mod error;
mod models;
mod output;
mod parsers;
mod tokenizers;
mod utils;

#[cfg(feature = "compression")]
mod compression;

#[cfg(feature = "load-test")]
mod http;
#[cfg(feature = "load-test")]
mod simulator;

use clap::Parser;
use cli::Cli;

fn main() {
    let cli = Cli::parse();

    if let Err(e) = cli.run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
