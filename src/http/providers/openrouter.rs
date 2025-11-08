/// OpenRouter API client implementation.
#[cfg(feature = "load-test")]
use crate::error::AppError;
#[cfg(feature = "load-test")]
use crate::http::client::{ClientConfig, LlmClient, LlmResponse};
#[cfg(feature = "load-test")]
use reqwest::Client;
#[cfg(feature = "load-test")]
use serde::{Deserialize, Serialize};

/// OpenRouter API client.
///
/// OpenRouter provides a unified API for accessing multiple LLM providers.
/// It uses a compatible format with OpenAI's API.
#[cfg(feature = "load-test")]
pub struct OpenRouterClient {
    client: Client,
    config: ClientConfig,
}

/// OpenRouter API request payload (compatible with OpenAI format).
#[cfg(feature = "load-test")]
#[derive(Debug, Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<Message>,
    temperature: Option<f64>,
}

/// Message in OpenRouter format (same as OpenAI).
#[cfg(feature = "load-test")]
#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

/// OpenRouter API response (compatible with OpenAI format).
#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
    model: Option<String>,
}

/// Choice in OpenRouter response.
#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct Choice {
    message: MessageResponse,
}

/// Message in OpenRouter response.
#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct MessageResponse {
    content: String,
}

/// Token usage information.
#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: Option<usize>,
    completion_tokens: Option<usize>,
    total_tokens: Option<usize>,
}

#[cfg(feature = "load-test")]
impl OpenRouterClient {
    /// Create a new OpenRouter client.
    pub fn new(config: ClientConfig) -> Result<Self, AppError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| AppError::Http(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { client, config })
    }

    /// Get the default OpenRouter API endpoint.
    fn default_endpoint() -> String {
        "https://openrouter.ai/api/v1/chat/completions".to_string()
    }
}

#[cfg(feature = "load-test")]
#[async_trait::async_trait]
impl LlmClient for OpenRouterClient {
    async fn send_request(&self, prompt: &str, model: &str) -> Result<LlmResponse, AppError> {
        let endpoint = if self.config.endpoint.is_empty() {
            Self::default_endpoint()
        } else {
            self.config.endpoint.clone()
        };

        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        let request = OpenRouterRequest {
            model: model.to_string(),
            messages,
            temperature: Some(0.7),
        };

        let mut req = self
            .client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            // OpenRouter requires HTTP-Referer header (can be any URL)
            .header("HTTP-Referer", "https://github.com/nooscraft/tokuin")
            // Optional: X-Title header for identifying the application
            .header("X-Title", "Tokuin");

        // Add custom headers (these will override defaults if same key)
        for (key, value) in &self.config.headers {
            req = req.header(key, value);
        }

        let response = req
            .json(&request)
            .send()
            .await
            .map_err(|e| AppError::Http(format!("Request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AppError::Api(format!(
                "OpenRouter API error ({}): {}",
                status, error_text
            )));
        }

        let api_response: OpenRouterResponse = response
            .json()
            .await
            .map_err(|e| AppError::Http(format!("Failed to parse JSON response: {}", e)))?;

        let content = api_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| AppError::Api("No response content".to_string()))?;

        let usage = api_response.usage;
        let model_name = api_response.model.unwrap_or_else(|| model.to_string());

        Ok(LlmResponse {
            content,
            input_tokens: usage.as_ref().and_then(|u| u.prompt_tokens),
            output_tokens: usage.as_ref().and_then(|u| u.completion_tokens),
            total_tokens: usage.as_ref().and_then(|u| u.total_tokens),
            model: model_name,
        })
    }

    fn provider_name(&self) -> &str {
        "openrouter"
    }
}

#[cfg(test)]
#[cfg(feature = "load-test")]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn test_openrouter_client_creation() {
        let config = ClientConfig {
            endpoint: String::new(),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(60),
            headers: Vec::new(),
        };
        let client = OpenRouterClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_default_endpoint() {
        let endpoint = OpenRouterClient::default_endpoint();
        assert_eq!(endpoint, "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn test_provider_name() {
        let config = ClientConfig {
            endpoint: String::new(),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(60),
            headers: Vec::new(),
        };
        let client = OpenRouterClient::new(config).unwrap();
        assert_eq!(client.provider_name(), "openrouter");
    }

    #[tokio::test]
    async fn send_request_includes_required_headers_and_payload() {
        let server = MockServer::start_async().await;
        let path = "/api/v1/chat/completions";

        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(path)
                    .header("Authorization", "Bearer or-key")
                    .header("Content-Type", "application/json")
                    .header("HTTP-Referer", "https://example.com")
                    .header("X-Title", "Custom Title")
                    .json_body(json!({
                        "model": "openrouter/llama",
                        "messages": [
                            { "role": "user", "content": "load test" }
                        ],
                        "temperature": 0.7
                    }));

                then.status(200).json_body(json!({
                    "choices": [
                        { "message": { "content": "ack" } }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 3,
                        "total_tokens": 14
                    },
                    "model": "openrouter/llama"
                }));
            })
            .await;

        let config = ClientConfig {
            endpoint: format!("{}{}", server.base_url(), path),
            api_key: "or-key".to_string(),
            timeout: Duration::from_secs(30),
            headers: vec![
                ("HTTP-Referer".into(), "https://example.com".into()),
                ("X-Title".into(), "Custom Title".into()),
            ],
        };

        let client = OpenRouterClient::new(config).expect("client initialization");
        let response = client
            .send_request("load test", "openrouter/llama")
            .await
            .expect("request should succeed");

        assert_eq!(response.content, "ack");
        assert_eq!(response.input_tokens, Some(11));
        assert_eq!(response.output_tokens, Some(3));
        assert_eq!(response.total_tokens, Some(14));
        assert_eq!(response.model, "openrouter/llama");

        mock.assert_async().await;
    }
}
