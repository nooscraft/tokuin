/// Generic REST API client implementation.
#[cfg(feature = "load-test")]
use crate::error::AppError;
#[cfg(feature = "load-test")]
use crate::http::client::{ClientConfig, LlmClient, LlmResponse};
#[cfg(feature = "load-test")]
use reqwest::Client;
#[cfg(feature = "load-test")]
use serde::{Deserialize, Serialize};

/// Generic REST API client.
#[cfg(feature = "load-test")]
pub struct GenericClient {
    client: Client,
    config: ClientConfig,
}

#[cfg(feature = "load-test")]
impl GenericClient {
    /// Create a new generic client.
    pub fn new(config: ClientConfig) -> Result<Self, AppError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| AppError::Http(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { client, config })
    }
}

#[cfg(feature = "load-test")]
#[async_trait::async_trait]
impl LlmClient for GenericClient {
    async fn send_request(&self, prompt: &str, model: &str) -> Result<LlmResponse, AppError> {
        if self.config.endpoint.is_empty() {
            return Err(AppError::Config(
                "Generic provider requires an endpoint. Specify one with --endpoint.".to_string(),
            ));
        }

        let request = GenericRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
        };

        let mut req = self.client.post(&self.config.endpoint);

        if !self.config.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

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
                "Generic API error ({}): {}",
                status, error_text
            )));
        }

        let api_response: GenericResponse = response
            .json()
            .await
            .map_err(|e| AppError::Http(format!("Failed to parse JSON response: {}", e)))?;

        let content = api_response.primary_content().ok_or_else(|| {
            AppError::Api("Generic response did not contain content field".into())
        })?;

        let usage = api_response.usage.unwrap_or_default();

        Ok(LlmResponse {
            content,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            model: model.to_string(),
        })
    }

    fn provider_name(&self) -> &str {
        "generic"
    }
}

#[cfg(feature = "load-test")]
#[derive(Debug, Serialize)]
struct GenericRequest {
    model: String,
    prompt: String,
}

#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize, Default)]
struct GenericResponse {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    response: Option<String>,
    #[serde(default)]
    result: Option<String>,
    #[serde(default)]
    output: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    choices: Option<Vec<GenericChoice>>,
    #[serde(default)]
    usage: Option<GenericUsage>,
}

#[cfg(feature = "load-test")]
impl GenericResponse {
    fn primary_content(&self) -> Option<String> {
        self.content
            .clone()
            .or(self.response.clone())
            .or(self.result.clone())
            .or(self.output.clone())
            .or(self.message.clone())
            .or_else(|| {
                self.choices.as_ref().and_then(|choices| {
                    choices.iter().find_map(|choice| {
                        choice
                            .text
                            .clone()
                            .or_else(|| choice.message.as_ref().and_then(|m| m.content.clone()))
                    })
                })
            })
    }
}

#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct GenericChoice {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    message: Option<GenericMessage>,
}

#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize)]
struct GenericMessage {
    #[serde(default)]
    content: Option<String>,
}

#[cfg(feature = "load-test")]
#[derive(Debug, Deserialize, Default)]
struct GenericUsage {
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
    async fn generic_client_handles_response_variants() {
        let server = MockServer::start_async().await;
        let path = "/mock-endpoint";

        let mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path(path)
                    .header("Authorization", "Bearer generic-key")
                    .json_body(json!({
                        "model": "custom-model",
                        "prompt": "Ping"
                    }));

                then.status(200).json_body(json!({
                    "choices": [
                        { "message": { "content": "Pong!" } }
                    ],
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 7,
                        "total_tokens": 12
                    }
                }));
            })
            .await;

        let config = ClientConfig {
            endpoint: format!("{}{}", server.base_url(), path),
            api_key: "generic-key".into(),
            timeout: Duration::from_secs(15),
            headers: vec![("X-Custom".into(), "Value".into())],
        };

        let client = GenericClient::new(config).expect("generic client init");
        let response = client
            .send_request("Ping", "custom-model")
            .await
            .expect("request should succeed");

        assert_eq!(response.content, "Pong!");
        assert_eq!(response.input_tokens, Some(5));
        assert_eq!(response.output_tokens, Some(7));
        assert_eq!(response.total_tokens, Some(12));

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn generic_client_requires_endpoint() {
        let config = ClientConfig {
            endpoint: String::new(),
            api_key: String::new(),
            timeout: Duration::from_secs(15),
            headers: Vec::new(),
        };

        let client = GenericClient::new(config).expect("generic client init");
        let result = client.send_request("prompt", "model").await;
        assert!(matches!(result, Err(AppError::Config(_))));
    }
}
