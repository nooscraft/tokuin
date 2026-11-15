/// Tokenizer implementations for various LLM providers.
pub mod trait_impl;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "gemini")]
pub mod gemini;

pub use trait_impl::Tokenizer;

/// Look up per-1K input/output prices for a model without a built-in rate.
///
/// Uses the `llm-prices` table (USD per 1M tokens) when the `llm-prices` feature is enabled.
#[cfg(feature = "llm-prices")]
#[allow(dead_code)]
pub(crate) fn table_price_per_1k(model: &str) -> (Option<f64>, Option<f64>) {
    match llm_prices::lookup(model) {
        Some(p) => (p.input.map(|v| v / 1000.0), p.output.map(|v| v / 1000.0)),
        None => (None, None),
    }
}

/// Without the `llm-prices` feature, models without a built-in rate have no price.
#[cfg(not(feature = "llm-prices"))]
#[allow(dead_code)]
pub(crate) fn table_price_per_1k(_model: &str) -> (Option<f64>, Option<f64>) {
    (None, None)
}

#[cfg(feature = "openai")]
pub use openai::OpenAITokenizer;

#[cfg(feature = "gemini")]
pub use gemini::GeminiTokenizer;
