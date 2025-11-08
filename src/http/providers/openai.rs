/// OpenAI API client implementation.
#[cfg(feature = "load-test")]
use crate::error::AppError;
#[cfg(feature = "load-test")]
use crate::http::client::{ClientConfig, LlmClient, LlmResponse};
#[cfg(feature = "load-test")]
use reqwest::Client;
#[cfg(feature = "load-test")]
use serde::{Deserialize, Serialize};

/// OpenAI API client.
#[cfg(feature = "load-test")]
pub struct OpenAIClient {
    client: Client,
    config: ClientConfig,
}

/// OpenAI API request payload.
#[cfg(feature = "load-test")]
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
    temperature: Option<f64>,
}

/// Message in OpenAI format.
#[cfg(feature = "load-test")]
#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

/// OpenAI API response.
#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
    model: Option<String>,
}

/// Choice in OpenAI response.
#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct Choice {
    message: MessageResponse,
}

/// Message in OpenAI response.
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
impl OpenAIClient {
    /// Create a new OpenAI client.
    pub fn new(config: ClientConfig) -> Result<Self, AppError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| AppError::Http(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { client, config })
    }

    /// Get the default OpenAI API endpoint.
    fn default_endpoint() -> String {
        "https://api.openai.com/v1/chat/completions".to_string()
    }
}

#[cfg(feature = "load-test")]
#[async_trait::async_trait]
impl LlmClient for OpenAIClient {
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

        let request = OpenAIRequest {
            model: model.to_string(),
            messages,
            temperature: Some(0.7),
        };

        let mut req = self
            .client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        // Add custom headers
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
                "API error ({}): {}",
                status, error_text
            )));
        }

        let api_response: OpenAIResponse = response
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
        "openai"
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
    fn test_openai_client_creation() {
        let config = ClientConfig {
            endpoint: String::new(),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(60),
            headers: Vec::new(),
        };
        let client = OpenAIClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_default_endpoint() {
        let endpoint = OpenAIClient::default_endpoint();
        assert_eq!(endpoint, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_provider_name() {
        let config = ClientConfig {
            endpoint: String::new(),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(60),
            headers: Vec::new(),
        };
        let client = OpenAIClient::new(config).unwrap();
        assert_eq!(client.provider_name(), "openai");
    }

    #[tokio::test]
    async fn send_request_posts_expected_payload_and_headers() {
        let server = MockServer::start_async().await;
        let path = "/v1/chat/completions";

        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(path)
                    .header("Authorization", "Bearer test-key")
                    .header("Content-Type", "application/json")
                    .header("X-Custom-Header", "custom-value")
                    .json_body(json!({
                        "model": "gpt-4o-mini",
                        "messages": [
                            { "role": "user", "content": "ping" }
                        ],
                        "temperature": 0.7
                    }));

                then.status(200).json_body(json!({
                    "choices": [
                        { "message": { "content": "pong" } }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 7,
                        "total_tokens": 12
                    },
                    "model": "gpt-4o-mini"
                }));
            })
            .await;

        let config = ClientConfig {
            endpoint: format!("{}{}", server.base_url(), path),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(30),
            headers: vec![("X-Custom-Header".into(), "custom-value".into())],
        };

        let client = OpenAIClient::new(config).expect("client initialization");
        let response = client
            .send_request("ping", "gpt-4o-mini")
            .await
            .expect("request should succeed");

        assert_eq!(response.content, "pong");
        assert_eq!(response.input_tokens, Some(5));
        assert_eq!(response.output_tokens, Some(7));
        assert_eq!(response.total_tokens, Some(12));
        assert_eq!(response.model, "gpt-4o-mini");

        mock.assert_async().await;
    }
}
