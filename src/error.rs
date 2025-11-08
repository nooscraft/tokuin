/// Error types for the tokuin crate.
use thiserror::Error;

/// Errors that can occur during tokenization.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum TokenizerError {
    #[error("Failed to initialize tokenizer: {0}")]
    InitializationFailed(String),

    #[error("Unsupported model: {model}")]
    UnsupportedModel { model: String },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Encoding failed: {0}")]
    EncodingFailed(String),

    #[error("Decoding failed: {0}")]
    DecodingFailed(String),

    #[cfg(feature = "openai")]
    #[error("OpenAI tokenizer error: {0}")]
    OpenAI(String),
}

/// Errors related to model registry and configuration.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum ModelError {
    #[error("Model not found: {model}")]
    ModelNotFound { model: String },

    #[error("Failed to load model configuration: {0}")]
    ConfigLoadFailed(String),

    #[error("Invalid pricing configuration: {0}")]
    InvalidPricing(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
}

/// Errors that can occur during parsing.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum ParseError {
    #[error("Invalid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    #[error("Missing required field: {field}")]
    MissingField { field: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Application-level errors.
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "load-test")]
    #[error("HTTP error: {0}")]
    Http(String),

    #[cfg(feature = "load-test")]
    #[error("API error: {0}")]
    Api(String),

    #[cfg(feature = "load-test")]
    #[error("Configuration error: {0}")]
    Config(String),

    #[cfg(feature = "load-test")]
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn tokenizer_error_variants_construct() {
        let _ = TokenizerError::InitializationFailed("init failed".into());
        let _ = TokenizerError::UnsupportedModel {
            model: "unknown-model".into(),
        };
        let _ = TokenizerError::InvalidInput("bad input".into());
        let _ = TokenizerError::EncodingFailed("encode failure".into());
        let _ = TokenizerError::DecodingFailed("decode failure".into());
        #[cfg(feature = "openai")]
        {
            let _ = TokenizerError::OpenAI("openai failure".into());
        }
    }

    #[test]
    fn model_error_variants_construct() {
        let _ = ModelError::ModelNotFound {
            model: "missing".into(),
        };
        let _ = ModelError::ConfigLoadFailed("load failure".into());
        let _ = ModelError::InvalidPricing("pricing issue".into());
    }

    #[test]
    fn parse_error_variants_construct() {
        let json_err = serde_json::from_str::<Value>("invalid").unwrap_err();
        let _ = ParseError::InvalidJson(json_err);
        let _ = ParseError::InvalidFormat("bad format".into());
        let _ = ParseError::MissingField {
            field: "field".into(),
        };
    }
}
