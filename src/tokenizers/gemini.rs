/// Google Gemini tokenizer implementation.
#[cfg(feature = "gemini")]
use crate::error::TokenizerError;
use crate::tokenizers::Tokenizer;

#[cfg(all(feature = "gemini", feature = "sentencepiece"))]
use sentencepiece::SentencePieceProcessor;

/// Google Gemini tokenizer implementation.
///
/// Note: This is an approximation using SentencePiece. For exact token counts,
/// you may need to use Google's official tokenizer or API.
///
/// # Example
///
/// ```rust
/// use tokuin::tokenizers::{GeminiTokenizer, Tokenizer};
///
/// let tokenizer = GeminiTokenizer::new("gemini-pro")?;
/// let count = tokenizer.count_tokens("Hello, world!")?;
/// # Ok::<(), tokuin::error::TokenizerError>(())
/// ```
#[cfg(feature = "gemini")]
pub struct GeminiTokenizer {
    #[cfg(feature = "sentencepiece")]
    processor: Option<SentencePieceProcessor>,
    model_name: String,
    input_price: Option<f64>,
    output_price: Option<f64>,
}

#[cfg(feature = "gemini")]
impl GeminiTokenizer {
    /// Create a new Gemini tokenizer for the specified model.
    ///
    /// # Arguments
    ///
    /// * `model` - The Gemini model name (e.g., "gemini-pro", "gemini-2.5-pro")
    ///
    /// # Returns
    ///
    /// A new `GeminiTokenizer` instance, or an error if initialization fails.
    ///
    /// # Note
    ///
    /// This implementation uses a generic SentencePiece model as an approximation.
    /// For production use, consider using Google's official tokenizer or API.
    pub fn new(model: &str) -> Result<Self, TokenizerError> {
        // Set pricing based on model (as of 2024)
        let (input_price, output_price) = match model {
            "gemini-2.5-pro" | "gemini-pro" => {
                // $1.25 per 1M input, $10 per 1M output
                (Some(0.00125), Some(0.01))
            }
            "gemini-2.5-flash" | "gemini-flash" => {
                // $0.075 per 1M input, $0.30 per 1M output
                (Some(0.000075), Some(0.0003))
            }
            _ => crate::tokenizers::table_price_per_1k(model),
        };

        Ok(Self {
            #[cfg(feature = "sentencepiece")]
            processor: None,
            model_name: model.to_string(),
            input_price,
            output_price,
        })
    }

    /// Create a Gemini tokenizer with a custom model file.
    ///
    /// # Arguments
    ///
    /// * `model` - The model name
    /// * `model_path` - Path to the SentencePiece model file
    ///
    /// # Returns
    ///
    /// A new `GeminiTokenizer` instance.
    #[cfg(feature = "sentencepiece")]
    #[allow(dead_code)]
    pub fn with_model_file(model: &str, model_path: &str) -> Result<Self, TokenizerError> {
        let processor = SentencePieceProcessor::open(model_path).map_err(|e| {
            TokenizerError::InitializationFailed(format!(
                "Failed to load Gemini tokenizer model from '{}': {}",
                model_path, e
            ))
        })?;

        let (input_price, output_price) = match model {
            "gemini-2.5-pro" | "gemini-pro" => (Some(0.00125), Some(0.01)),
            "gemini-2.5-flash" | "gemini-flash" => (Some(0.000075), Some(0.0003)),
            _ => crate::tokenizers::table_price_per_1k(model),
        };

        Ok(Self {
            processor: Some(processor),
            model_name: model.to_string(),
            input_price,
            output_price,
        })
    }
}

#[cfg(feature = "gemini")]
impl Tokenizer for GeminiTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        #[cfg(feature = "sentencepiece")]
        if let Some(ref processor) = self.processor {
            return processor
                .encode(text)
                .map(|pieces| pieces.into_iter().map(|piece| piece.id as usize).collect())
                .map_err(|e| TokenizerError::EncodingFailed(e.to_string()));
        }

        // Fallback: Use character-based approximation
        // Gemini typically uses ~4 characters per token (similar to GPT)
        // This is an approximation and may not be exact
        Ok(text
            .chars()
            .collect::<Vec<_>>()
            .chunks(4)
            .map(|_| 1)
            .collect())
    }

    fn decode(&self, _tokens: &[usize]) -> Result<String, TokenizerError> {
        #[cfg(feature = "sentencepiece")]
        if let Some(ref processor) = self.processor {
            let tokens_u32: Vec<u32> = _tokens.iter().map(|&t| t as u32).collect();
            return processor
                .decode_piece_ids(&tokens_u32)
                .map_err(|e| TokenizerError::DecodingFailed(e.to_string()));
        }

        // Fallback: Can't decode without processor
        Err(TokenizerError::DecodingFailed(
            "Decoding requires SentencePiece model file".to_string(),
        ))
    }

    fn count_tokens(&self, text: &str) -> Result<usize, TokenizerError> {
        #[cfg(feature = "sentencepiece")]
        if let Some(ref processor) = self.processor {
            return processor
                .encode(text)
                .map(|tokens| tokens.len())
                .map_err(|e| TokenizerError::EncodingFailed(e.to_string()));
        }

        // Fallback: Character-based approximation
        // Gemini typically uses ~4 characters per token
        // This is an approximation - for exact counts, use Google's API or provide model file
        Ok((text.chars().count() as f64 / 4.0).ceil() as usize)
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
