/// Model registry for managing available models and their tokenizers.
use crate::error::ModelError;
use crate::tokenizers::Tokenizer;

#[cfg(feature = "openai")]
use crate::tokenizers::OpenAITokenizer;

#[cfg(feature = "gemini")]
use crate::tokenizers::GeminiTokenizer;

use std::collections::HashMap;

/// Information about a model.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelInfo {
    /// The provider name (e.g., "openai", "anthropic").
    pub provider: String,
    /// The model name/identifier.
    pub model: String,
    /// Input price per 1K tokens in USD.
    pub input_price: Option<f64>,
    /// Output price per 1K tokens in USD.
    pub output_price: Option<f64>,
}

/// Registry for managing models and their tokenizers.
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Create a new model registry with default models.
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
        };
        registry.register_default_models();
        registry
    }

    /// Register a model in the registry.
    pub fn register(&mut self, name: String, info: ModelInfo) {
        self.models.insert(name, info);
    }

    /// Get information about a model.
    #[allow(dead_code)]
    pub fn get_model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        // Try direct lookup first
        if let Some(info) = self.models.get(model_name) {
            return Some(info);
        }

        // Try aliases
        let alias = self.resolve_alias(model_name);
        self.models.get(&alias)
    }

    /// Create a tokenizer for the specified model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model (e.g., "gpt-4").
    ///
    /// # Returns
    ///
    /// A boxed tokenizer, or an error if the model is not supported.
    ///
    /// # Errors
    ///
    /// Returns `ModelError::ModelNotFound` if the model is not registered,
    /// or `ModelError::Tokenizer` if the tokenizer cannot be created.
    pub fn get_tokenizer(&self, model_name: &str) -> Result<Box<dyn Tokenizer>, ModelError> {
        let model = self.resolve_alias(model_name);

        #[cfg(feature = "openai")]
        if model.starts_with("gpt-") || model.starts_with("text-") {
            return OpenAITokenizer::new(&model)
                .map(|t| Box::new(t) as Box<dyn Tokenizer>)
                .map_err(ModelError::from);
        }

        #[cfg(feature = "gemini")]
        if model.starts_with("gemini-") {
            // Note: This will fail without a model file, but provides the structure
            // In production, you'd handle model file loading or use an approximation
            return GeminiTokenizer::new(&model)
                .map(|t| Box::new(t) as Box<dyn Tokenizer>)
                .map_err(ModelError::from);
        }

        Err(ModelError::ModelNotFound {
            model: model_name.to_string(),
        })
    }

    /// Resolve model aliases to canonical names.
    fn resolve_alias(&self, model_name: &str) -> String {
        match model_name {
            "gpt-4-turbo" | "gpt-4-turbo-preview" => "gpt-4-turbo-preview".to_string(),
            _ => model_name.to_string(),
        }
    }

    /// Register default models.
    fn register_default_models(&mut self) {
        // OpenAI models
        #[cfg(feature = "openai")]
        {
            self.register(
                "gpt-4".to_string(),
                ModelInfo {
                    provider: "openai".to_string(),
                    model: "gpt-4".to_string(),
                    input_price: Some(0.03),
                    output_price: Some(0.06),
                },
            );

            self.register(
                "gpt-4-turbo".to_string(),
                ModelInfo {
                    provider: "openai".to_string(),
                    model: "gpt-4-turbo".to_string(),
                    input_price: Some(0.01),
                    output_price: Some(0.03),
                },
            );

            self.register(
                "gpt-3.5-turbo".to_string(),
                ModelInfo {
                    provider: "openai".to_string(),
                    model: "gpt-3.5-turbo".to_string(),
                    input_price: Some(0.0015),
                    output_price: Some(0.002),
                },
            );
        }

        // Gemini models
        #[cfg(feature = "gemini")]
        {
            self.register(
                "gemini-pro".to_string(),
                ModelInfo {
                    provider: "google".to_string(),
                    model: "gemini-pro".to_string(),
                    input_price: Some(0.00125),
                    output_price: Some(0.01),
                },
            );

            self.register(
                "gemini-2.5-pro".to_string(),
                ModelInfo {
                    provider: "google".to_string(),
                    model: "gemini-2.5-pro".to_string(),
                    input_price: Some(0.00125),
                    output_price: Some(0.01),
                },
            );

            self.register(
                "gemini-2.5-flash".to_string(),
                ModelInfo {
                    provider: "google".to_string(),
                    model: "gemini-2.5-flash".to_string(),
                    input_price: Some(0.000075),
                    output_price: Some(0.0003),
                },
            );
        }
    }

    /// List all registered models.
    #[allow(dead_code)]
    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "openai")]
    fn test_get_tokenizer() {
        let registry = ModelRegistry::new();
        let tokenizer = registry.get_tokenizer("gpt-4");
        assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_get_model_info() {
        let registry = ModelRegistry::new();
        let info = registry.get_model_info("gpt-4");
        assert!(info.is_some());
        if let Some(info) = info {
            assert_eq!(info.provider, "openai");
            assert_eq!(info.model, "gpt-4");
            assert_eq!(info.input_price, Some(0.03));
            assert_eq!(info.output_price, Some(0.06));
        }
    }

    #[test]
    fn test_list_models() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();
        assert!(!models.is_empty());
    }
}
