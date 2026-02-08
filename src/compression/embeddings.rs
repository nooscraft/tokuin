/// Embedding provider for semantic scoring
///
/// This module provides embedding functionality for semantic understanding
/// of text, enabling better compression quality through semantic similarity
/// rather than just keyword matching.
use crate::error::AppError;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Type alias for progress callback function
pub type ProgressCallback = Box<dyn Fn(&str) + Send + Sync>;

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
    #[allow(dead_code)] // Public API trait method
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, AppError> {
        texts.iter().map(|text| self.embed(text)).collect()
    }

    /// Get the dimension of embeddings produced by this provider
    #[allow(dead_code)] // Public API trait method
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
    #[allow(dead_code)] // Public API method
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    #[allow(dead_code)] // Public API method
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
#[allow(dead_code)] // Public API struct
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
    pub struct OnnxEmbeddingProvider {
        tokenizer: tokenizers::Tokenizer,
        dimension: usize,
        #[cfg(feature = "ort")]
        session: Option<std::sync::Mutex<ort::session::Session>>,
        #[cfg(feature = "ort")]
        #[allow(dead_code)] // Internal field for tracking model state
        model_loaded: bool,
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
                return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Tokenizer not found at {:?}. Run 'tokuin setup models' to download embedding models.",
                    tokenizer_file
                ))));
            };

            // Try to load ONNX model if available
            #[cfg(feature = "ort")]
            let (session, model_loaded) = if model_file.exists() {
                match Self::load_onnx_session(&model_file) {
                    Ok(sess) => {
                        eprintln!("✓ ONNX model loaded successfully");
                        (Some(std::sync::Mutex::new(sess)), true)
                    }
                    Err(e) => {
                        eprintln!("⚠️  Warning: Failed to load ONNX model: {}", e);
                        eprintln!("   Falling back to placeholder embeddings.");
                        (None, false)
                    }
                }
            } else {
                eprintln!("⚠️  Note: ONNX model file not found. Using placeholder embeddings.");
                eprintln!("   To use full model inference, run: tokuin setup models --onnx");
                eprintln!("   Or convert manually: pip install optimum[exporters] && optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 ./onnx_model/");
                (None, false)
            };

            #[cfg(not(feature = "ort"))]
            let (session, model_loaded) = (None, false);

            // Get model output dimension from metadata or use default
            let dimension = 384; // all-MiniLM-L6-v2 default dimension

            Ok(Self {
                tokenizer,
                dimension,
                #[cfg(feature = "ort")]
                session,
                #[cfg(feature = "ort")]
                model_loaded,
            })
        }

        /// Get the model directory, checking bundled location first, then cache
        pub fn model_cache_dir() -> Result<PathBuf, AppError> {
            // Check for bundled models (relative to binary location)
            if let Ok(exe_path) = std::env::current_exe() {
                if let Some(exe_dir) = exe_path.parent() {
                    let bundled_path = exe_dir.join("models").join("all-MiniLM-L6-v2");
                    let tokenizer_file = bundled_path.join("tokenizer.json");
                    if tokenizer_file.exists() {
                        return Ok(bundled_path);
                    }
                }
            }

            // Fallback to cache directory
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
                    eprintln!("⚠️  Models not found. Run 'tokuin setup models' to download them.");
                    eprintln!("   Attempting automatic download...");
                    eprintln!("   This may take a few minutes on first use.");

                    // Try download, but don't fail if it doesn't work
                    if let Err(e) = Self::download_from_huggingface(&model_path) {
                        eprintln!(
                            "⚠️  Warning: Could not download model from HuggingFace: {}",
                            e
                        );
                        eprintln!("   Run 'tokuin setup models' to download manually.");
                        eprintln!("   Using placeholder tokenizer-based embeddings for now.");
                        // Continue anyway - we'll use placeholder embeddings
                    } else {
                        eprintln!("✓ Models downloaded successfully");
                    }
                }
            }

            // Return path even if download failed - we can still use placeholder mode
            Ok(model_path)
        }

        /// Load ONNX session from model file
        #[cfg(feature = "ort")]
        fn load_onnx_session(model_path: &Path) -> Result<ort::session::Session, AppError> {
            use ort::session::Session;

            // Create session builder with optimization
            let builder = Session::builder().map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Failed to create ONNX session builder: {}",
                    e
                )))
            })?;

            let session = builder.commit_from_file(model_path).map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Failed to load ONNX model from {:?}: {}",
                    model_path, e
                )))
            })?;

            Ok(session)
        }

        /// Download model files from HuggingFace
        /// Uses hf-hub if available, falls back to direct HTTP download
        #[cfg(feature = "hf-hub")]
        fn download_from_huggingface(model_path: &Path) -> Result<(), AppError> {
            let model_id = "sentence-transformers/all-MiniLM-L6-v2";
            let _tokenizer_file = model_path.join("tokenizer.json");

            // Try hf-hub first
            eprintln!("Attempting download via hf-hub...");
            match Self::download_via_hf_hub(model_path, model_id, true, false) {
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

        /// Download tokenizer from HuggingFace
        #[cfg(feature = "hf-hub")]
        pub fn download_tokenizer(model_path: &Path) -> Result<(), AppError> {
            let model_id = "sentence-transformers/all-MiniLM-L6-v2";
            Self::download_via_hf_hub(model_path, model_id, true, false)
        }

        /// Download ONNX model from HuggingFace (if available)
        #[cfg(feature = "hf-hub")]
        pub fn download_onnx_model(model_path: &Path) -> Result<(), AppError> {
            let model_id = "sentence-transformers/all-MiniLM-L6-v2";
            Self::download_via_hf_hub(model_path, model_id, false, true)
        }

        /// Download using hf-hub crate
        ///
        /// Downloads the tokenizer (required) and attempts to download the ONNX model
        /// (optional, for future full inference support).
        #[cfg(feature = "hf-hub")]
        fn download_via_hf_hub(
            model_path: &Path,
            model_id: &str,
            download_tokenizer: bool,
            download_onnx: bool,
        ) -> Result<(), AppError> {
            use hf_hub::api::sync::Api;

            let api = Api::new().map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Failed to initialize HuggingFace API: {}",
                    e
                )))
            })?;

            let repo = api.model(model_id.to_string());

            // Download tokenizer (required for placeholder embeddings)
            if download_tokenizer {
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
                }
            }

            // Try to download ONNX model (for future full inference support)
            // Note: Most sentence-transformers models don't have pre-converted ONNX files
            // Users will need to convert them manually using optimum-cli
            if download_onnx {
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
                    } else {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            "ONNX model not found in HuggingFace repo. Convert manually using optimum-cli.".to_string()
                        )));
                    }
                }
            }

            Ok(())
        }

        /// Download using direct HTTP (fallback when hf-hub fails)
        #[cfg(feature = "hf-hub")]
        fn download_via_http(model_path: &Path, model_id: &str) -> Result<(), AppError> {
            let tokenizer_url = format!(
                "https://huggingface.co/{}/resolve/main/tokenizer.json",
                model_id
            );

            eprintln!("Downloading from: {}", tokenizer_url);

            // Try using reqwest blocking client if available
            #[cfg(feature = "load-test")]
            {
                Self::download_via_reqwest(model_path, &tokenizer_url)
            }

            #[cfg(not(feature = "load-test"))]
            {
                // Fallback: provide manual download instructions
                let dest_path = model_path.join("tokenizer.json");
                eprintln!("⚠️  HTTP client not available. Please download manually:");
                eprintln!("   curl -L {} -o {:?}", tokenizer_url, dest_path);
                eprintln!("\n   Or use: wget {} -O {:?}", tokenizer_url, dest_path);

                Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                    format!(
                        "Please download tokenizer.json manually from {} to {:?}",
                        tokenizer_url, dest_path
                    ),
                )))
            }
        }

        /// Download using reqwest (requires blocking feature) or curl fallback
        #[cfg(all(feature = "hf-hub", feature = "load-test"))]
        fn download_via_reqwest(model_path: &Path, url: &str) -> Result<(), AppError> {
            use std::process::Command;

            // Try using curl first (simpler and more reliable)
            let dest_path = model_path.join("tokenizer.json");
            eprintln!("Attempting download via curl...");

            let curl_result = Command::new("curl")
                .arg("-L") // Follow redirects
                .arg("-f") // Fail on HTTP errors
                .arg("-s") // Silent mode
                .arg("-S") // Show errors
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
                use std::io::Write;

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
                    return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                        format!("HTTP request failed with status: {}", response.status()),
                    )));
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
        /// Uses ONNX model inference if available, otherwise falls back to placeholder.
        fn embed_text(&self, text: &str) -> Result<Vec<f32>, AppError> {
            #[cfg(feature = "ort")]
            {
                if let Some(session_mutex) = &self.session {
                    let mut session = session_mutex.lock().map_err(|e| {
                        AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                            "Failed to lock ONNX session: {}",
                            e
                        )))
                    })?;
                    return Self::embed_with_onnx(
                        &mut session,
                        &self.tokenizer,
                        text,
                        self.dimension,
                    );
                }
            }

            // Fallback to placeholder implementation
            Self::embed_placeholder(&self.tokenizer, text, self.dimension)
        }

        /// Generate embedding using ONNX model inference
        #[cfg(feature = "ort")]
        fn embed_with_onnx(
            session: &mut ort::session::Session,
            tokenizer: &tokenizers::Tokenizer,
            text: &str,
            dimension: usize,
        ) -> Result<Vec<f32>, AppError> {
            use ort::inputs;

            // Tokenize the input
            let encoding = tokenizer.encode(text, true).map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Tokenization failed: {}",
                    e
                )))
            })?;

            let ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();

            // Convert to arrays for ONNX Runtime
            // MiniLM models typically use max_length of 512
            let max_length = 512;
            let mut input_ids = vec![0i64; max_length];
            let mut attention_mask_arr = vec![0i64; max_length];

            for (i, &id) in ids.iter().take(max_length).enumerate() {
                input_ids[i] = id as i64;
                attention_mask_arr[i] = attention_mask.get(i).copied().unwrap_or(0) as i64;
            }

            // Create ONNX input tensors from vectors
            // ort::value::Value::from_array accepts (shape, Vec<T>) format
            use ort::value::Value;

            let input_ids_value = Value::from_array(([1, max_length], input_ids)).map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Failed to create input_ids tensor: {}",
                    e
                )))
            })?;

            let attention_mask_value = Value::from_array(([1, max_length], attention_mask_arr))
                .map_err(|e| {
                    AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                        "Failed to create attention_mask tensor: {}",
                        e
                    )))
                })?;

            // Create inputs using ort::inputs macro
            let inputs = inputs! {
                "input_ids" => input_ids_value,
                "attention_mask" => attention_mask_value,
            };

            // Run inference
            let outputs = session.run(inputs).map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "ONNX inference failed: {}",
                    e
                )))
            })?;

            // Extract embeddings from output
            // For sentence-transformers, the output is typically the last hidden state
            // We need to get the pooled output (usually CLS token or mean pooling)
            // The output key might be "last_hidden_state" or the model might output pooled directly
            let output_key = outputs
                .keys()
                .find(|k| k.contains("hidden") || k.contains("pooler") || k.contains("output"))
                .or_else(|| outputs.keys().next())
                .ok_or_else(|| {
                    AppError::Parse(crate::error::ParseError::InvalidFormat(
                        "No output found in ONNX inference results".to_string(),
                    ))
                })?;

            let output_value = outputs.get(output_key).ok_or_else(|| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                    "Output key '{}' not found",
                    output_key
                )))
            })?;

            // Extract tensor data from Value using try_extract_tensor
            // This is the correct method according to ort crate v2.0.0-rc.10 documentation
            // try_extract_tensor returns (&Shape, &[T]) tuple
            let (output_shape_ref, output_slice) =
                output_value.try_extract_tensor::<f32>().map_err(|e| {
                    AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                        "Failed to extract tensor from output: {}",
                        e
                    )))
                })?;

            // Convert shape to Vec<usize>
            let output_shape_actual: Vec<usize> =
                output_shape_ref.iter().map(|&d| d as usize).collect();

            let mut embedding = Vec::with_capacity(dimension);

            // Handle different output shapes
            // Access data from the slice using manual indexing based on shape
            if output_shape_actual.len() == 3 {
                // [batch, seq_len, hidden] - take mean pooling over sequence dimension
                let batch_size = output_shape_actual[0];
                let seq_len = output_shape_actual[1];
                let hidden_size = output_shape_actual[2];

                // Mean pooling: average over sequence length
                for d in 0..dimension.min(hidden_size) {
                    let mut sum = 0.0f32;
                    let mut count = 0;
                    for b in 0..batch_size {
                        for s in 0..seq_len {
                            let idx = (b * seq_len * hidden_size) + (s * hidden_size) + d;
                            if idx < output_slice.len() {
                                sum += output_slice[idx];
                                count += 1;
                            }
                        }
                    }
                    embedding.push(if count > 0 { sum / count as f32 } else { 0.0 });
                }
            } else if output_shape_actual.len() == 2 {
                // [batch, hidden] - already pooled
                let hidden_size = output_shape_actual[1];
                for d in 0..dimension.min(hidden_size) {
                    let idx = d; // First batch element, dimension d
                    if idx < output_slice.len() {
                        embedding.push(output_slice[idx]);
                    } else {
                        embedding.push(0.0);
                    }
                }
            } else {
                return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                    format!("Unexpected output shape: {:?}", output_shape_actual),
                )));
            }

            // Pad or truncate to expected dimension
            while embedding.len() < dimension {
                embedding.push(0.0);
            }
            embedding.truncate(dimension);

            // Normalize to unit vector
            normalize_embedding(&mut embedding);

            Ok(embedding)
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
// Note: This is kept for API compatibility but may not be used in all code paths
#[cfg(feature = "compression-embeddings")]
#[allow(unused_imports)]
pub use onnx_impl::OnnxEmbeddingProvider as CandleEmbeddingProvider;

/// Download embedding models with optional progress reporting
///
/// Downloads the tokenizer (required) and optionally the ONNX model.
/// Returns the path to the model directory.
#[cfg(feature = "compression-embeddings")]
pub fn download_models(
    force: bool,
    download_onnx: bool,
    progress_callback: Option<ProgressCallback>,
) -> Result<std::path::PathBuf, AppError> {
    use onnx_impl::OnnxEmbeddingProvider;

    let model_path = OnnxEmbeddingProvider::model_cache_dir()?;
    let tokenizer_file = model_path.join("tokenizer.json");
    let model_file = model_path.join("model.onnx");

    let report = |msg: &str| {
        if let Some(ref cb) = progress_callback {
            cb(msg);
        } else {
            eprintln!("{}", msg);
        }
    };

    // Download tokenizer if needed
    if force || !tokenizer_file.exists() {
        if tokenizer_file.exists() && force {
            report("Re-downloading tokenizer...");
        } else {
            report("Downloading tokenizer...");
        }

        #[cfg(feature = "hf-hub")]
        {
            OnnxEmbeddingProvider::download_tokenizer(&model_path)?;
            report("✓ Tokenizer downloaded successfully");
        }

        #[cfg(not(feature = "hf-hub"))]
        {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "hf-hub feature not enabled. Cannot download models.".to_string(),
            )));
        }
    } else {
        report("✓ Tokenizer already exists");
    }

    // Download/convert ONNX model if requested
    if download_onnx && (force || !model_file.exists()) {
        if model_file.exists() && force {
            report("Re-downloading ONNX model...");
        } else {
            report("Downloading/converting ONNX model...");
        }

        #[cfg(feature = "hf-hub")]
        {
            // Try to download from HuggingFace first
            if OnnxEmbeddingProvider::download_onnx_model(&model_path).is_err() {
                report("⚠️  ONNX model not available in HuggingFace repo.");
                report("   To convert manually:");
                report("   pip install optimum[exporters]");
                report("   optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 ./onnx_model/");
                report(&format!("   Then copy model.onnx to: {:?}", model_file));
            } else {
                report("✓ ONNX model downloaded successfully");
            }
        }
    } else if model_file.exists() {
        report("✓ ONNX model already exists");
    }

    Ok(model_path)
}

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
        let result = cache.get_or_compute("test", &provider, |_| Ok(vec![0.1, 0.2, 0.3]));
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
