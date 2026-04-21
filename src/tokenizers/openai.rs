/// OpenAI tokenizer implementation using tiktoken-rs.
#[cfg(feature = "openai")]
use crate::error::TokenizerError;
use crate::tokenizers::Tokenizer;
use tiktoken_rs::{bpe_for_model, CoreBPE};

/// OpenAI tokenizer implementation.
///
/// This tokenizer uses the `tiktoken-rs` crate to provide accurate
/// tokenization for OpenAI models.
///
/// # Example
///
/// ```rust
/// use tokuin::tokenizers::{OpenAITokenizer, Tokenizer};
///
/// let tokenizer = OpenAITokenizer::new("gpt-4")?;
/// let count = tokenizer.count_tokens("Hello, world!")?;
/// # Ok::<(), tokuin::error::TokenizerError>(())
/// ```
pub struct OpenAITokenizer {
    bpe: CoreBPE,
    model_name: String,
    input_price: Option<f64>,
    output_price: Option<f64>,
}

impl OpenAITokenizer {
    /// Create a new OpenAI tokenizer for the specified model.
    ///
    /// # Arguments
    ///
    /// * `model` - The OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
    ///
    /// # Returns
    ///
    /// A new `OpenAITokenizer` instance, or an error if the model is not supported.
    ///
    /// # Errors
    ///
    /// Returns `TokenizerError::InitializationFailed` if the model is not recognized
    /// or cannot be initialized.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tokuin::tokenizers::OpenAITokenizer;
    ///
    /// let tokenizer = OpenAITokenizer::new("gpt-4")?;
    /// # Ok::<(), tokuin::error::TokenizerError>(())
    /// ```
    pub fn new(model: &str) -> Result<Self, TokenizerError> {
        let bpe = bpe_for_model(model).map_err(|e| {
            TokenizerError::InitializationFailed(format!(
                "Failed to initialize tokenizer for model '{}': {}",
                model, e
            ))
        })?;

        // Set pricing based on model (default pricing as of 2024)
        let (input_price, output_price) = match model {
            "gpt-4" | "gpt-4-0314" | "gpt-4-32k" | "gpt-4-32k-0314" => (Some(0.03), Some(0.06)),
            "gpt-4-turbo-preview" | "gpt-4-turbo" | "gpt-4-0125-preview" => {
                (Some(0.01), Some(0.03))
            }
            "gpt-3.5-turbo" | "gpt-3.5-turbo-0301" => (Some(0.0015), Some(0.002)),
            "gpt-3.5-turbo-16k" => (Some(0.003), Some(0.004)),
            _ => (None, None),
        };

        Ok(Self {
            bpe: bpe.clone(),
            model_name: model.to_string(),
            input_price,
            output_price,
        })
    }
}

impl Tokenizer for OpenAITokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        Ok(self
            .bpe
            .encode_with_special_tokens(text)
            .into_iter()
            .map(|t| t as usize)
            .collect())
    }

    fn decode(&self, tokens: &[usize]) -> Result<String, TokenizerError> {
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        self.bpe
            .decode(&tokens_u32)
            .map_err(|e| TokenizerError::DecodingFailed(e.to_string()))
    }

    fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        // Use optimized counting method
        Ok(self.bpe.encode_with_special_tokens(text).len())
    }

    fn name(&self) -> &str {
        &self.model_name
    }

    fn input_price_per_1k(&self) -> Option<f64> {
        self.input_price
    }

    fn output_price_per_1k(&self) -> Option<f64> {
        self.output_price
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_tokenizer_creation() {
        let tokenizer = OpenAITokenizer::new("gpt-4");
        assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_openai_tokenizer_invalid_model() {
        let tokenizer = OpenAITokenizer::new("invalid-model");
        assert!(tokenizer.is_err());
    }

    #[test]
    fn test_encode_decode() {
        let tokenizer = OpenAITokenizer::new("gpt-4").unwrap();
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_count_tokens() {
        let tokenizer = OpenAITokenizer::new("gpt-4").unwrap();
        let count = tokenizer.count_tokens("Hello, world!").unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_count_tokens_empty() {
        let tokenizer = OpenAITokenizer::new("gpt-4").unwrap();
        let count = tokenizer.count_tokens("").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_unicode_handling() {
        let tokenizer = OpenAITokenizer::new("gpt-4").unwrap();
        let text = "Hello 世界 🌍";
        let count = tokenizer.count_tokens(text).unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_pricing() {
        let tokenizer = OpenAITokenizer::new("gpt-4").unwrap();
        assert_eq!(tokenizer.input_price_per_1k(), Some(0.03));
        assert_eq!(tokenizer.output_price_per_1k(), Some(0.06));
    }
}
