/// HTTP client abstraction for LLM API providers.
#[cfg(feature = "load-test")]
use crate::error::AppError;
#[cfg(feature = "load-test")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "load-test")]
use std::time::Duration;

/// Response from an LLM API call.
#[cfg(feature = "load-test")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// The generated text content
    pub content: String,
    /// Input tokens used
    pub input_tokens: Option<usize>,
    /// Output tokens used
    pub output_tokens: Option<usize>,
    /// Total tokens used
    pub total_tokens: Option<usize>,
    /// Model used for the response
    pub model: String,
}

/// Trait for LLM API clients.
#[cfg(feature = "load-test")]
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a prompt to the LLM API and get a response.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The prompt text or messages to send
    /// * `model` - The model to use
    ///
    /// # Returns
    ///
    /// The LLM response with content and token usage
    async fn send_request(&self, prompt: &str, model: &str) -> Result<LlmResponse, AppError>;

    /// Get the provider name.
    #[allow(dead_code)]
    fn provider_name(&self) -> &str;
}

/// Enum wrapper for different LLM clients.
#[cfg(feature = "load-test")]
pub enum LlmClientEnum {
    #[cfg(feature = "load-test")]
    OpenAI(crate::http::providers::openai::OpenAIClient),
    #[cfg(feature = "load-test")]
    OpenRouter(crate::http::providers::openrouter::OpenRouterClient),
}

#[cfg(feature = "load-test")]
#[async_trait::async_trait]
impl LlmClient for LlmClientEnum {
    async fn send_request(&self, prompt: &str, model: &str) -> Result<LlmResponse, AppError> {
        match self {
            LlmClientEnum::OpenAI(client) => client.send_request(prompt, model).await,
            LlmClientEnum::OpenRouter(client) => client.send_request(prompt, model).await,
        }
    }

    fn provider_name(&self) -> &str {
        match self {
            LlmClientEnum::OpenAI(client) => client.provider_name(),
            LlmClientEnum::OpenRouter(client) => client.provider_name(),
        }
    }
}

/// HTTP client configuration.
#[cfg(feature = "load-test")]
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// API endpoint URL
    pub endpoint: String,
    /// API key
    pub api_key: String,
    /// Request timeout
    pub timeout: Duration,
    /// Additional headers
    pub headers: Vec<(String, String)>,
}

#[cfg(feature = "load-test")]
impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            api_key: String::new(),
            timeout: Duration::from_secs(60),
            headers: Vec::new(),
        }
    }
}
