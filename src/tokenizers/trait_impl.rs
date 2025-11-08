/// Core trait for tokenizer implementations.
use crate::error::TokenizerError;

/// Trait for tokenizing text into tokens and counting tokens.
///
/// All tokenizer implementations must implement this trait to provide
/// consistent tokenization across different LLM providers.
pub trait Tokenizer: Send + Sync {
    /// Encode text into token IDs.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to encode. Must be valid UTF-8.
    ///
    /// # Returns
    ///
    /// A vector of token IDs, or an error if encoding fails.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError` if the text cannot be encoded.
    fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError>;

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The token IDs to decode.
    ///
    /// # Returns
    ///
    /// The decoded text, or an error if decoding fails.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError` if the tokens cannot be decoded.
    #[allow(dead_code)]
    fn decode(&self, tokens: &[usize]) -> Result<String, TokenizerError>;

    /// Count tokens in text (optimized path).
    ///
    /// This method is optimized for cases where you only need the count,
    /// not the actual token IDs. Implementations may override this for
    /// better performance.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to count tokens in.
    ///
    /// # Returns
    ///
    /// The number of tokens in the text, or an error if tokenization fails.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError` if the text cannot be tokenized.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use tokuin::tokenizers::Tokenizer;
    /// # let tokenizer = tokuin::tokenizers::OpenAITokenizer::new("gpt-4").unwrap();
    /// let count = tokenizer.count_tokens("Hello, world!")?;
    /// # Ok::<(), tokuin::error::TokenizerError>(())
    /// ```
    fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        self.encode(text).map(|tokens| tokens.len())
    }

    /// Get the tokenizer's name/identifier.
    ///
    /// # Returns
    ///
    /// A string slice containing the tokenizer name.
    fn name(&self) -> &str;

    /// Get the estimated price per 1K tokens for input.
    ///
    /// # Returns
    ///
    /// The price in USD per 1K input tokens, or `None` if pricing is unknown.
    fn input_price_per_1k(&self) -> Option<f64>;

    /// Get the estimated price per 1K tokens for output.
    ///
    /// # Returns
    ///
    /// The price in USD per 1K output tokens, or `None` if pricing is unknown.
    fn output_price_per_1k(&self) -> Option<f64>;
}
