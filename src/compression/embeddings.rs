/// Embedding provider for semantic scoring
///
/// This module provides embedding functionality for semantic understanding
/// of text, enabling better compression quality through semantic similarity
/// rather than just keyword matching.
use crate::error::AppError;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

/// Trait for embedding providers
///
/// Embedding providers convert text into dense vector representations
/// that capture semantic meaning.
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding for a single text
    fn embed(&self, text: &str) -> Result<Vec<f32>, AppError>;

    /// Generate embeddings for multiple texts (batch processing)
    ///
    /// Default implementation processes sequentially, but implementations
    /// can override for better performance.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, AppError> {
        texts.iter().map(|text| self.embed(text)).collect()
    }

    /// Get the dimension of embeddings produced by this provider
    fn dimension(&self) -> usize;

    /// Check if the provider is available and ready to use
    fn is_available(&self) -> bool;
}

/// Simple embedding cache to avoid recomputing embeddings
#[derive(Clone)]
pub struct EmbeddingCache {
    cache: HashMap<String, Vec<f32>>,
    max_size: usize,
}

impl EmbeddingCache {
    /// Create a new cache with a maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
        }
    }

    /// Get an embedding from cache or compute it
    pub fn get_or_compute<F>(
        &mut self,
        text: &str,
        _provider: &dyn EmbeddingProvider,
        compute: F,
    ) -> Result<Vec<f32>, AppError>
    where
        F: FnOnce(&str) -> Result<Vec<f32>, AppError>,
    {
        let hash = Self::hash_text(text);
        let key = hash.to_string();

        if let Some(emb) = self.cache.get(&key) {
            return Ok(emb.clone());
        }

        let emb = compute(text)?;

        // Evict oldest entries if cache is full
        if self.cache.len() >= self.max_size {
            // Simple eviction: remove first entry (FIFO)
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }

        self.cache.insert(key, emb.clone());
        Ok(emb)
    }

    /// Hash text for cache key
    fn hash_text(text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Cosine similarity between two embedding vectors
pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f64 {
    if v1.len() != v2.len() {
        return 0.0;
    }

    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    (dot_product / (norm1 * norm2)) as f64
}

/// Normalize an embedding vector to unit length
pub fn normalize_embedding(embedding: &mut [f32]) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in embedding.iter_mut() {
            *val /= norm;
        }
    }
}

/// Stub implementation for when embeddings are not available
///
/// This allows the code to compile and run without embedding dependencies,
/// falling back to heuristic scoring.
pub struct StubEmbeddingProvider;

impl EmbeddingProvider for StubEmbeddingProvider {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, AppError> {
        Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
            "Embeddings not available. Install with --features compression-embeddings".to_string(),
        )))
    }

    fn dimension(&self) -> usize {
        0
    }

    fn is_available(&self) -> bool {
        false
    }
}

#[cfg(feature = "compression-embeddings")]
mod onnx_impl {
    use super::*;
    use crate::error::AppError;
    use std::path::{Path, PathBuf};

    /// ONNX Runtime-based embedding provider using MiniLM model
    ///
    /// Uses ONNX Runtime for fast inference with pre-trained sentence transformer models.
    /// 
    /// **Current Status**: Uses placeholder tokenizer-based embeddings.
    /// Full ONNX model inference is pending API integration with the `ort` crate.
    /// 
    /// **To enable full ONNX inference:**
    /// 1. Download or convert the model to ONNX format (see `download_from_huggingface`)
    /// 2. Fix the `ort` crate API usage (the API structure differs from expected)
    /// 3. Update `embed_text()` to use actual ONNX inference instead of placeholder
    /// 
    /// The placeholder provides basic semantic signal but is not as accurate as the trained model.
    pub struct OnnxEmbeddingProvider {
        tokenizer: tokenizers::Tokenizer,
        dimension: usize,
        #[cfg(feature = "ort")]
        #[allow(dead_code)]
        model_loaded: bool,
        // TODO: Add ONNX session when ort crate API is properly integrated
        // session: Option<ort::Session>,
    }

    impl OnnxEmbeddingProvider {
        /// Create a new ONNX embedding provider
        ///
        /// This will download the model from HuggingFace if not present,
        /// then load it into memory.
        pub fn new() -> Result<Self, AppError> {
            let model_path = Self::ensure_model()?;
            let model_file = model_path.join("model.onnx");

            // Load tokenizer first (required for both model and fallback)
            let tokenizer_file = model_path.join("tokenizer.json");
            let tokenizer = if tokenizer_file.exists() {
                tokenizers::Tokenizer::from_file(&tokenizer_file).map_err(|e| {
                    AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                        "Failed to load tokenizer: {}",
                        e
                    )))
                })?
            } else {
                return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                    "Tokenizer not found. Please ensure tokenizer.json is in the model directory."
                        .to_string(),
                )));
            };

            // Check if model file exists (for future ONNX integration)
            // For now, we use placeholder embeddings even if the model file exists
            // because the ort crate API integration is pending
            let model_loaded = if model_file.exists() {
                eprintln!("⚠️  Note: ONNX model file found but full inference is not yet implemented.");
                eprintln!("   Using placeholder tokenizer-based embeddings for now.");
                eprintln!("   See documentation in OnnxEmbeddingProvider for integration status.");
                false
            } else {
                eprintln!("⚠️  Note: ONNX model file not found. Using placeholder embeddings.");
                eprintln!("   To use full model inference (when implemented), download model.onnx from HuggingFace.");
                false
            };

            // Get model output dimension from metadata or use default
            let dimension = 384; // all-MiniLM-L6-v2 default dimension

            Ok(Self {
                tokenizer,
                dimension,
                #[cfg(feature = "ort")]
                model_loaded,
            })
        }

        /// Get the model cache directory
        fn model_cache_dir() -> Result<PathBuf, AppError> {
            let cache_dir = dirs::cache_dir()
                .ok_or_else(|| {
                    AppError::Io(std::io::Error::other(
                        "Failed to get cache directory".to_string(),
                    ))
                })?
                .join("tokuin")
                .join("models")
                .join("all-MiniLM-L6-v2");

            std::fs::create_dir_all(&cache_dir).map_err(|e| {
                AppError::Io(std::io::Error::other(format!(
                    "Failed to create cache directory: {}",
                    e
                )))
            })?;

            Ok(cache_dir)
        }

        /// Download model from HuggingFace if not present
        /// Returns the model path even if download fails (for placeholder mode)
        fn ensure_model() -> Result<PathBuf, AppError> {
            let model_path = Self::model_cache_dir()?;
            let model_file = model_path.join("model.onnx");
            let tokenizer_file = model_path.join("tokenizer.json");

            // If both files exist, we're good
            if model_file.exists() && tokenizer_file.exists() {
                return Ok(model_path);
            }

            // Try to download if hf-hub is available
            #[cfg(feature = "hf-hub")]
            {
                let mut needs_download = false;
                if !tokenizer_file.exists() {
                    needs_download = true;
                }
                if !model_file.exists() {
                    needs_download = true;
                }

                if needs_download {
                    eprintln!("Downloading embedding model from HuggingFace...");
                    eprintln!("This may take a few minutes on first use.");
                    
                    // Try download, but don't fail if it doesn't work
                    if let Err(e) = Self::download_from_huggingface(&model_path) {
                        eprintln!("⚠️  Warning: Could not download model from HuggingFace: {}", e);
                        eprintln!("   Using placeholder tokenizer-based embeddings for now.");
                        // Continue anyway - we'll use placeholder embeddings
                    }
                }
            }

            // Return path even if download failed - we can still use placeholder mode
            Ok(model_path)
        }

        /// Load ONNX session from model file
        /// 
        /// **TODO**: This function is not yet implemented due to ort crate API differences.
        /// The ort crate (v2.0.0-rc.10) API structure differs from what was expected.
        /// 
        /// **What's needed:**
        /// 1. Research the correct ort crate API for Session creation
        /// 2. Handle error conversion from ort::Error to AppError
        /// 3. Properly configure optimization level and threading
        /// 
        /// **Resources:**
        /// - ort crate documentation: https://docs.rs/ort
        /// - ONNX Runtime C API: https://onnxruntime.ai/docs/api/c/
        #[cfg(feature = "ort")]
        #[allow(dead_code)]
        fn load_onnx_session(_model_path: &Path) -> Result<(), AppError> {
            // TODO: Implement ONNX session loading
            // The ort crate API needs to be properly integrated
            Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "ONNX session loading not yet implemented. See function documentation.".to_string(),
            )))
        }

        /// Download model files from HuggingFace
        /// Uses hf-hub if available, falls back to direct HTTP download
        #[cfg(feature = "hf-hub")]
        fn download_from_huggingface(model_path: &Path) -> Result<(), AppError> {
            use hf_hub::api::sync::Api;
            let model_id = "sentence-transformers/all-MiniLM-L6-v2";
            let dest_tokenizer = model_path.join("tokenizer.json");

            // Try hf-hub first
            eprintln!("Attempting download via hf-hub...");
            match Self::download_via_hf_hub(model_path, model_id) {
                Ok(()) => {
                    eprintln!("✓ Tokenizer downloaded successfully via hf-hub");
                    return Ok(());
                }
                Err(e) => {
                    eprintln!("⚠️  hf-hub download failed: {}", e);
                    eprintln!("   Falling back to direct HTTP download...");
                }
            }

            // Fallback: Direct HTTP download
            Self::download_via_http(model_path, model_id)
        }

        /// Download using hf-hub crate
        /// 
        /// Downloads the tokenizer (required) and attempts to download the ONNX model
        /// (optional, for future full inference support).
        #[cfg(feature = "hf-hub")]
        fn download_via_hf_hub(model_path: &Path, model_id: &str) -> Result<(), AppError> {
            use hf_hub::api::sync::Api;

            let api = Api::new().map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Failed to initialize HuggingFace API: {}",
                    e
                )))
            })?;

            let repo = api.model(model_id.to_string());
            
            // Download tokenizer (required for placeholder embeddings)
            let tokenizer_file = model_path.join("tokenizer.json");
            if !tokenizer_file.exists() {
                let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
                    AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                        "hf-hub tokenizer download failed: {}",
                        e
                    )))
                })?;
                std::fs::copy(&tokenizer_path, &tokenizer_file).map_err(|e| {
                    AppError::Io(std::io::Error::other(format!(
                        "Failed to copy tokenizer: {}",
                        e
                    )))
                })?;
                eprintln!("✓ Tokenizer downloaded");
            }

            // Try to download ONNX model (for future full inference support)
            // Note: Most sentence-transformers models don't have pre-converted ONNX files
            // Users will need to convert them manually using optimum-cli
            let model_file = model_path.join("model.onnx");
            if !model_file.exists() {
                // Check if ONNX model exists in the repo (unlikely for most models)
                if let Ok(onnx_path) = repo.get("model.onnx") {
                    std::fs::copy(&onnx_path, &model_file).map_err(|e| {
                        AppError::Io(std::io::Error::other(format!(
                            "Failed to copy ONNX model: {}",
                            e
                        )))
                    })?;
                    eprintln!("✓ ONNX model downloaded");
                } else {
                    eprintln!("⚠️  Note: ONNX model not found in HuggingFace repo.");
                    eprintln!("   For full ONNX inference (when implemented), convert the model:");
                    eprintln!("   pip install optimum[exporters]");
                    eprintln!("   optimum-cli export onnx --model {} ./onnx_model/", model_id);
                    eprintln!("   Then copy model.onnx to: {:?}", model_file);
                }
            }

            Ok(())
        }

        /// Download using direct HTTP (fallback when hf-hub fails)
        #[cfg(feature = "hf-hub")]
        fn download_via_http(model_path: &Path, model_id: &str) -> Result<(), AppError> {
            use std::io::Write;

            let tokenizer_url = format!(
                "https://huggingface.co/{}/resolve/main/tokenizer.json",
                model_id
            );

            eprintln!("Downloading from: {}", tokenizer_url);

            // Try using reqwest blocking client if available
            #[cfg(feature = "load-test")]
            {
                return Self::download_via_reqwest(model_path, &tokenizer_url);
            }

            // Fallback: provide manual download instructions
            let dest_path = model_path.join("tokenizer.json");
            eprintln!("⚠️  HTTP client not available. Please download manually:");
            eprintln!("   curl -L {} -o {:?}", tokenizer_url, dest_path);
            eprintln!("\n   Or use: wget {} -O {:?}", tokenizer_url, dest_path);

            Err(AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                "Please download tokenizer.json manually from {} to {:?}",
                tokenizer_url, dest_path
            ))))
        }

        /// Download using reqwest (requires blocking feature) or curl fallback
        #[cfg(all(feature = "hf-hub", feature = "load-test"))]
        fn download_via_reqwest(model_path: &Path, url: &str) -> Result<(), AppError> {
            use std::io::Write;
            use std::process::Command;

            // Try using curl first (simpler and more reliable)
            let dest_path = model_path.join("tokenizer.json");
            eprintln!("Attempting download via curl...");
            
            let curl_result = Command::new("curl")
                .arg("-L")  // Follow redirects
                .arg("-f")  // Fail on HTTP errors
                .arg("-s")  // Silent mode
                .arg("-S")  // Show errors
                .arg(url)
                .arg("-o")
                .arg(&dest_path)
                .output();

            match curl_result {
                Ok(output) if output.status.success() => {
                    eprintln!("✓ Tokenizer downloaded successfully via curl");
                    return Ok(());
                }
                Ok(output) => {
                    let error_msg = String::from_utf8_lossy(&output.stderr);
                    eprintln!("⚠️  curl failed: {}", error_msg);
                }
                Err(_) => {
                    eprintln!("⚠️  curl not available, trying reqwest...");
                }
            }

            // Fallback: try reqwest with tokio runtime
            eprintln!("Attempting download via reqwest (async)...");
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Failed to create tokio runtime: {}",
                    e
                )))
            })?;

            rt.block_on(async {
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(60))
                    .build()
                    .map_err(|e| {
                        AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                            "Failed to create HTTP client: {}",
                            e
                        )))
                    })?;

                let response = client.get(url).send().await.map_err(|e| {
                    AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                        "HTTP request failed: {}",
                        e
                    )))
                })?;

                if !response.status().is_success() {
                    return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                        "HTTP request failed with status: {}",
                        response.status()
                    ))));
                }

                let content = response.bytes().await.map_err(|e| {
                    AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                        "Failed to read response: {}",
                        e
                    )))
                })?;

                let mut file = std::fs::File::create(&dest_path).map_err(|e| {
                    AppError::Io(std::io::Error::other(format!(
                        "Failed to create file {:?}: {}",
                        dest_path, e
                    )))
                })?;

                file.write_all(&content).map_err(|e| {
                    AppError::Io(std::io::Error::other(format!(
                        "Failed to write file: {}",
                        e
                    )))
                })?;

                Ok::<(), AppError>(())
            })
        }

        /// Generate embedding for a single text
        /// 
        /// **Current Implementation**: Uses placeholder tokenizer-based embeddings.
        /// 
        /// **Future**: Will use ONNX model inference when fully integrated.
        /// The placeholder provides basic semantic signal but is not as accurate
        /// as the trained model embeddings.
        fn embed_text(&self, text: &str) -> Result<Vec<f32>, AppError> {
            // TODO: When ONNX integration is complete, check if model is loaded
            // and use embed_with_onnx() instead of embed_placeholder()
            
            // Current: Always use placeholder
            Self::embed_placeholder(&self.tokenizer, text, self.dimension)
        }

        /// Generate placeholder embedding using tokenizer-based features
        /// 
        /// This creates embeddings based on token frequencies, positions, and word hashes.
        /// While not as accurate as the trained model, it provides some semantic signal
        /// for compression scoring.
        fn embed_placeholder(
            tokenizer: &tokenizers::Tokenizer,
            text: &str,
            dimension: usize,
        ) -> Result<Vec<f32>, AppError> {
            // Tokenize text to get meaningful features
            let encoding = tokenizer.encode(text, true).map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Tokenization failed: {}",
                    e
                )))
            })?;

            // Create a simple embedding based on token frequencies and positions
            let mut embedding = vec![0.0f32; dimension];
            
            let ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            
            // Simple TF-IDF-like embedding from tokens
            for (idx, &token_id) in ids.iter().enumerate() {
                if idx < dimension && attention_mask.get(idx).copied().unwrap_or(0) > 0 {
                    // Use token ID to seed embedding values
                    let seed = (token_id as u64).wrapping_mul(17).wrapping_add(idx as u64);
                    let value = ((seed % 10000) as f32 / 10000.0) * 2.0 - 1.0;
                    embedding[idx] += value * 0.1; // Small contribution per token
                }
            }
            
            // Add position-based features
            let text_lower = text.to_lowercase();
            let words: Vec<&str> = text_lower.split_whitespace().collect();
            for (i, word) in words.iter().take(dimension).enumerate() {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                word.hash(&mut hasher);
                let hash = hasher.finish();
                let value = ((hash % 10000) as f32 / 10000.0) * 2.0 - 1.0;
                embedding[i] += value * 0.05;
            }
            
            // Normalize to unit vector
            normalize_embedding(&mut embedding);

            Ok(embedding)
        }
    }

    impl EmbeddingProvider for OnnxEmbeddingProvider {
        fn embed(&self, text: &str) -> Result<Vec<f32>, AppError> {
            self.embed_text(text)
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, AppError> {
            // Process in batch for better performance
            texts.iter().map(|text| self.embed_text(text)).collect()
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn is_available(&self) -> bool {
            // We're available if we have a tokenizer (even without ONNX model)
            // The placeholder embeddings will work with just the tokenizer
            true
        }
    }
}

#[cfg(feature = "compression-embeddings")]
pub use onnx_impl::OnnxEmbeddingProvider;

// Keep CandleEmbeddingProvider as alias for backward compatibility
#[cfg(feature = "compression-embeddings")]
pub use onnx_impl::OnnxEmbeddingProvider as CandleEmbeddingProvider;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 0.001);

        let v3 = vec![1.0, 0.0, 0.0];
        let v4 = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&v3, &v4) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_embedding() {
        let mut v = vec![3.0, 4.0];
        normalize_embedding(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_cache() {
        let mut cache = EmbeddingCache::new(10);
        let provider = StubEmbeddingProvider;

        // Cache should be empty initially
        assert_eq!(cache.len(), 0);

        // Attempting to get from cache will fail (stub provider)
        let result = cache.get_or_compute("test", &provider, |_| {
            Ok(vec![0.1, 0.2, 0.3])
        });
        assert!(result.is_ok());
        assert_eq!(cache.len(), 1);

        // Second call should hit cache
        let result2 = cache.get_or_compute("test", &provider, |_| {
            Ok(vec![0.4, 0.5, 0.6]) // Different value, but should use cache
        });
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), vec![0.1, 0.2, 0.3]); // Should return cached value
    }
}

