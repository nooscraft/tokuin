/// Load testing simulator implementation.
#[cfg(feature = "load-test")]
use crate::error::AppError;
#[cfg(feature = "load-test")]
use crate::http::client::LlmClient;
#[cfg(feature = "load-test")]
use crate::simulator::config::SimulatorConfig;
#[cfg(feature = "load-test")]
use std::sync::Arc;
#[cfg(feature = "load-test")]
use std::time::{Duration, Instant};
#[cfg(feature = "load-test")]
use tokio::time::sleep;

/// Result of a single request.
#[cfg(feature = "load-test")]
#[derive(Debug, Clone)]
pub struct RequestResult {
    /// Success status
    pub success: bool,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Response content (if successful)
    pub content: Option<String>,
    /// Input tokens
    pub input_tokens: Option<usize>,
    /// Output tokens
    pub output_tokens: Option<usize>,
    /// Total tokens
    pub total_tokens: Option<usize>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Load testing simulator.
#[cfg(feature = "load-test")]
pub struct Simulator {
    config: SimulatorConfig,
}

#[cfg(feature = "load-test")]
impl Simulator {
    /// Create a new simulator.
    pub fn new(config: SimulatorConfig) -> Self {
        Self { config }
    }

    /// Run the load test.
    pub async fn run<C: LlmClient + 'static>(
        &self,
        client: Arc<C>,
        prompt: &str,
        model: &str,
    ) -> Result<Vec<RequestResult>, AppError> {
        self.run_with_progress(client, prompt, model, None).await
    }

    /// Run the load test with optional progress bar.
    pub async fn run_with_progress<C: LlmClient + 'static>(
        &self,
        client: Arc<C>,
        prompt: &str,
        model: &str,
        progress_bar: Option<Arc<indicatif::ProgressBar>>,
    ) -> Result<Vec<RequestResult>, AppError> {
        let mut results = Vec::with_capacity(self.config.runs);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.concurrency));
        let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let successful = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let total_latency = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let start_time = Arc::new(std::sync::Mutex::new(Instant::now()));

        for _i in 0..self.config.runs {
            let permit = semaphore.clone().acquire_owned().await.map_err(|e| {
                AppError::Config(format!("Failed to acquire semaphore permit: {}", e))
            })?;

            let client = client.clone();
            let prompt = prompt.to_string();
            let model = model.to_string();
            let config = self.config.clone();
            let progress = progress_bar.clone();
            let completed_clone = completed.clone();
            let successful_clone = successful.clone();
            let failed_clone = failed.clone();
            let total_latency_clone = total_latency.clone();
            let start_time_clone = start_time.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit;
                let result = Self::execute_request(client, &prompt, &model, &config).await;

                // Update progress
                let completed_count =
                    completed_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if result.success {
                    successful_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    total_latency_clone
                        .fetch_add(result.latency_ms, std::sync::atomic::Ordering::Relaxed);
                } else {
                    failed_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }

                if let Some(ref pb) = progress {
                    let success_count = successful_clone.load(std::sync::atomic::Ordering::Relaxed);
                    let fail_count = failed_clone.load(std::sync::atomic::Ordering::Relaxed);
                    let total_lat = total_latency_clone.load(std::sync::atomic::Ordering::Relaxed);
                    let avg_latency = if success_count > 0 {
                        total_lat / success_count as u64
                    } else {
                        0
                    };

                    // Calculate throughput (requests per second)
                    let elapsed = if let Ok(start) = start_time_clone.lock() {
                        start.elapsed().as_secs_f64()
                    } else {
                        0.0
                    };
                    let throughput = if elapsed > 0.0 {
                        completed_count as f64 / elapsed
                    } else {
                        0.0
                    };

                    pb.set_message(format!(
                        "Success: {} | Failed: {} | Avg Latency: {}ms | Throughput: {:.1} req/s",
                        success_count, fail_count, avg_latency, throughput
                    ));
                    pb.set_position(completed_count as u64);
                }

                result
            });

            results.push(handle);

            // Apply think time if configured
            if let Some(ref think_time) = self.config.think_time {
                let delay_ms = if think_time.min_ms == think_time.max_ms {
                    think_time.min_ms
                } else {
                    fastrand::u64(think_time.min_ms..=think_time.max_ms)
                };
                sleep(Duration::from_millis(delay_ms)).await;
            }
        }

        // Collect all results
        let mut collected_results = Vec::new();
        for handle in results {
            match handle.await {
                Ok(result) => collected_results.push(result),
                Err(e) => collected_results.push(RequestResult {
                    success: false,
                    latency_ms: 0,
                    content: None,
                    input_tokens: None,
                    output_tokens: None,
                    total_tokens: None,
                    error: Some(format!("Task join error: {}", e)),
                }),
            }
        }

        // Finish progress bar
        if let Some(ref pb) = progress_bar {
            pb.finish_with_message("Load test completed");
        }

        Ok(collected_results)
    }

    /// Execute a single request with retries.
    async fn execute_request<C: LlmClient>(
        client: Arc<C>,
        prompt: &str,
        model: &str,
        config: &SimulatorConfig,
    ) -> RequestResult {
        if config.dry_run {
            return RequestResult {
                success: true,
                latency_ms: 0,
                content: Some("(dry run)".to_string()),
                input_tokens: None,
                output_tokens: None,
                total_tokens: None,
                error: None,
            };
        }

        let mut last_error = None;

        for attempt in 0..=config.retry {
            let start = Instant::now();

            match client.send_request(prompt, model).await {
                Ok(response) => {
                    let latency_ms = start.elapsed().as_millis() as u64;
                    return RequestResult {
                        success: true,
                        latency_ms,
                        content: Some(response.content),
                        input_tokens: response.input_tokens,
                        output_tokens: response.output_tokens,
                        total_tokens: response.total_tokens,
                        error: None,
                    };
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                    if attempt < config.retry {
                        // Exponential backoff
                        let delay_ms = (2_u64.pow(attempt)) * 100;
                        sleep(Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        RequestResult {
            success: false,
            latency_ms: 0,
            content: None,
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
            error: last_error,
        }
    }
}

#[cfg(all(test, feature = "load-test"))]
mod tests {
    use super::*;
    use crate::error::AppError;
    use crate::http::client::LlmResponse;
    use async_trait::async_trait;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    struct MockClient {
        responses: Mutex<VecDeque<Result<LlmResponse, AppError>>>,
        call_count: AtomicUsize,
    }

    impl MockClient {
        fn new(responses: Vec<Result<LlmResponse, AppError>>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
                call_count: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LlmClient for MockClient {
        async fn send_request(&self, _prompt: &str, _model: &str) -> Result<LlmResponse, AppError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut guard = self.responses.lock().expect("responses mutex poisoned");
            guard
                .pop_front()
                .unwrap_or_else(|| Err(AppError::Api("no response available".into())))
        }

        fn provider_name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn run_respects_dry_run_mode() {
        let mut config = SimulatorConfig::new(2, 3);
        config.dry_run = true;
        config.retry = 0;
        let simulator = Simulator::new(config);

        let client = Arc::new(MockClient::new(Vec::new()));

        let results = simulator
            .run(client.clone(), "test prompt", "mock-model")
            .await
            .expect("dry run should not fail");

        assert_eq!(results.len(), 3);
        assert_eq!(client.calls(), 0, "dry run must avoid network calls");

        for result in results {
            assert!(result.success);
            assert_eq!(result.content.as_deref(), Some("(dry run)"));
            assert!(result.error.is_none());
        }
    }

    #[tokio::test]
    async fn run_handles_retries_and_failures() {
        let responses = vec![
            Ok(LlmResponse {
                content: "ok-1".into(),
                input_tokens: Some(30),
                output_tokens: Some(12),
                total_tokens: Some(42),
                model: "mock".into(),
            }),
            Err(AppError::Api("rate limited".into())),
            Ok(LlmResponse {
                content: "ok-2".into(),
                input_tokens: Some(25),
                output_tokens: Some(6),
                total_tokens: Some(31),
                model: "mock".into(),
            }),
            Err(AppError::Api("transient error".into())),
            Err(AppError::Api("persistent failure".into())),
        ];

        let client = Arc::new(MockClient::new(responses));

        let mut config = SimulatorConfig::new(1, 3);
        config.retry = 1;
        config.dry_run = false;
        config.think_time = None;

        let simulator = Simulator::new(config);

        let results = simulator
            .run(client.clone(), "prompt", "mock-model")
            .await
            .expect("simulation should complete");

        assert_eq!(results.len(), 3);
        assert_eq!(client.calls(), 5, "should perform retries when configured");

        assert!(results[0].success);
        assert_eq!(results[0].content.as_deref(), Some("ok-1"));
        assert!(results[0].error.is_none());

        assert!(results[1].success, "second request should succeed after retry");
        assert_eq!(results[1].content.as_deref(), Some("ok-2"));

        assert!(
            !results[2].success,
            "third request should fail after exhausting retries"
        );
        assert_eq!(
            results[2].error.as_deref(),
            Some("API error: persistent failure")
        );
    }
}
