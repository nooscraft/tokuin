/// Pricing configuration management.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pricing configuration for models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingConfig {
    /// Pricing by provider and model.
    #[serde(flatten)]
    pub providers: HashMap<String, ProviderPricing>,
}

/// Pricing for a provider's models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPricing {
    #[serde(flatten)]
    pub models: HashMap<String, ModelPricing>,
}

/// Pricing for a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Input price per 1K tokens in USD.
    pub input: f64,
    /// Output price per 1K tokens in USD.
    pub output: f64,
}

impl PricingConfig {
    /// Create a new pricing configuration with default values.
    pub fn new() -> Self {
        let mut providers = HashMap::new();

        // OpenAI pricing
        let mut openai_models = HashMap::new();
        openai_models.insert(
            "gpt-4".to_string(),
            ModelPricing {
                input: 0.03,
                output: 0.06,
            },
        );
        openai_models.insert(
            "gpt-4-turbo".to_string(),
            ModelPricing {
                input: 0.01,
                output: 0.03,
            },
        );
        openai_models.insert(
            "gpt-3.5-turbo".to_string(),
            ModelPricing {
                input: 0.0015,
                output: 0.002,
            },
        );

        providers.insert(
            "openai".to_string(),
            ProviderPricing {
                models: openai_models,
            },
        );

        Self { providers }
    }

    /// Get pricing for a model.
    ///
    /// # Arguments
    ///
    /// * `provider` - The provider name (e.g., "openai").
    /// * `model` - The model name (e.g., "gpt-4").
    ///
    /// # Returns
    ///
    /// The pricing for the model, or `None` if not found.
    #[allow(dead_code)]
    pub fn get_pricing(&self, provider: &str, model: &str) -> Option<&ModelPricing> {
        self.providers.get(provider)?.models.get(model)
    }

    /// Load pricing from a TOML configuration file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the TOML configuration file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    #[allow(dead_code)]
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: PricingConfig = toml::from_str(&content)?;
        Ok(config)
    }
}

impl Default for PricingConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_pricing() {
        let config = PricingConfig::new();
        let pricing = config.get_pricing("openai", "gpt-4");
        assert!(pricing.is_some());
        if let Some(p) = pricing {
            assert_eq!(p.input, 0.03);
            assert_eq!(p.output, 0.06);
        }
    }

    #[test]
    fn test_get_pricing_nonexistent() {
        let config = PricingConfig::new();
        let pricing = config.get_pricing("nonexistent", "model");
        assert!(pricing.is_none());
    }

    #[test]
    fn test_from_file() {
        let mut temp = tempfile::NamedTempFile::new().expect("create temp file");
        let content = r#"
            [openai]
            [openai.gpt-4]
            input = 0.02
            output = 0.04
        "#;
        use std::io::Write;
        temp.write_all(content.as_bytes())
            .expect("write pricing file");

        let config =
            PricingConfig::from_file(temp.path()).expect("load pricing config from temp file");
        let pricing = config.get_pricing("openai", "gpt-4");
        assert!(pricing.is_some());
        if let Some(p) = pricing {
            assert_eq!(p.input, 0.02);
            assert_eq!(p.output, 0.04);
        }
    }
}
