/// Generic REST API client implementation.
#[cfg(feature = "load-test")]
// TODO: Implement in Phase 4
use crate::error::AppError;
#[cfg(feature = "load-test")]
use crate::http::client::{ClientConfig, LlmClient, LlmResponse};

/// Generic REST API client.
#[cfg(feature = "load-test")]
#[allow(dead_code)]
pub struct GenericClient {
    _config: ClientConfig,
}

#[cfg(feature = "load-test")]
impl GenericClient {
    /// Create a new generic client.
    #[allow(dead_code)]
    pub fn new(_config: ClientConfig) -> Result<Self, AppError> {
        Err(AppError::Config(
            "Generic client not yet implemented".to_string(),
        ))
    }
}

#[cfg(feature = "load-test")]
#[async_trait::async_trait]
impl LlmClient for GenericClient {
    async fn send_request(&self, _prompt: &str, _model: &str) -> Result<LlmResponse, AppError> {
        Err(AppError::Config(
            "Generic client not yet implemented".to_string(),
        ))
    }

    fn provider_name(&self) -> &str {
        "generic"
    }
}

#[cfg(all(test, feature = "load-test"))]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn generic_client_stub_returns_error() {
        let config = ClientConfig {
            endpoint: String::new(),
            api_key: "test".into(),
            timeout: Duration::from_secs(30),
            headers: Vec::new(),
        };

        let client = GenericClient::new(config.clone());
        assert!(client.is_err());

        let stub = GenericClient { _config: config };
        let result = stub.send_request("prompt", "model").await;
        assert!(matches!(result, Err(AppError::Config(_))));
    }
}
