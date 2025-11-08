/// Anthropic API client implementation.
#[cfg(feature = "load-test")]
use crate::error::AppError;
#[cfg(feature = "load-test")]
use crate::http::client::{ClientConfig, LlmClient, LlmResponse};
#[cfg(feature = "load-test")]
use reqwest::Client;
#[cfg(feature = "load-test")]
use serde::{Deserialize, Serialize};

/// Anthropic API client.
#[cfg(feature = "load-test")]
pub struct AnthropicClient {
    client: Client,
    config: ClientConfig,
}

#[cfg(feature = "load-test")]
impl AnthropicClient {
    /// Create a new Anthropic client.
    pub fn new(config: ClientConfig) -> Result<Self, AppError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| AppError::Http(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { client, config })
    }

    fn default_endpoint() -> String {
        "https://api.anthropic.com/v1/messages".to_string()
    }
}

#[cfg(feature = "load-test")]
#[async_trait::async_trait]
impl LlmClient for AnthropicClient {
    async fn send_request(&self, prompt: &str, model: &str) -> Result<LlmResponse, AppError> {
        let endpoint = if self.config.endpoint.is_empty() {
            Self::default_endpoint()
        } else {
            self.config.endpoint.clone()
        };

        let request = AnthropicRequest {
            model: model.to_string(),
            max_tokens: 1024,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: Some(0.7),
        };

        let mut req = self
            .client
            .post(&endpoint)
            .header("x-api-key", &self.config.api_key)
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01");

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
                "Anthropic API error ({}): {}",
                status, error_text
            )));
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| AppError::Http(format!("Failed to parse JSON response: {}", e)))?;

        let content = api_response
            .content
            .into_iter()
            .find_map(|block| block.text)
            .ok_or_else(|| {
                AppError::Api("No response content returned from Anthropic".to_string())
            })?;

        let usage = api_response.usage;
        let model_name = api_response.model.unwrap_or_else(|| model.to_string());

        Ok(LlmResponse {
            content,
            input_tokens: usage.as_ref().and_then(|u| u.input_tokens),
            output_tokens: usage.as_ref().and_then(|u| u.output_tokens),
            total_tokens: usage.as_ref().and_then(|u| u.total_tokens),
            model: model_name,
        })
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}

#[cfg(feature = "load-test")]
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[cfg(feature = "load-test")]
#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    usage: Option<AnthropicUsage>,
    model: Option<String>,
}

#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(default)]
    text: Option<String>,
}

#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(rename = "input_tokens")]
    input_tokens: Option<usize>,
    #[serde(rename = "output_tokens")]
    output_tokens: Option<usize>,
    #[serde(rename = "total_tokens")]
    total_tokens: Option<usize>,
}

#[cfg(all(test, feature = "load-test"))]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;
    use std::time::Duration;

    #[tokio::test]
    async fn anthropic_client_sends_expected_payload() {
        let server = MockServer::start_async().await;
        let path = "/v1/messages";

        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(path)
                    .header("x-api-key", "sk-test")
                    .header("content-type", "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .json_body(json!({
                        "model": "claude-3-sonnet",
                        "max_tokens": 1024,
                        "messages": [
                            { "role": "user", "content": "Hello there" }
                        ],
                        "temperature": 0.7
                    }));

                then.status(200).json_body(json!({
                    "content": [
                        { "text": "Hello from Anthropic!" }
                    ],
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 18,
                        "total_tokens": 30
                    },
                    "model": "claude-3-sonnet"
                }));
            })
            .await;

        let config = ClientConfig {
            endpoint: format!("{}{}", server.base_url(), path),
            api_key: "sk-test".into(),
            timeout: Duration::from_secs(30),
            headers: Vec::new(),
        };

        let client = AnthropicClient::new(config).expect("client initialization");
        let response = client
            .send_request("Hello there", "claude-3-sonnet")
            .await
            .expect("request should succeed");

        assert_eq!(response.content, "Hello from Anthropic!");
        assert_eq!(response.input_tokens, Some(12));
        assert_eq!(response.output_tokens, Some(18));
        assert_eq!(response.total_tokens, Some(30));
        assert_eq!(response.model, "claude-3-sonnet");

        mock.assert_async().await;
    }
}
