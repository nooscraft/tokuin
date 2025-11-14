/// Data structures for prompt library analysis.
use crate::parsers::Message;
use std::path::PathBuf;

/// Analysis result for a single prompt file.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields are used in formatters and future features
pub struct PromptAnalysis {
    /// Path to the prompt file
    pub file_path: PathBuf,
    /// Identifier for the prompt (filename or extracted ID)
    pub prompt_id: String,
    /// Raw file content
    pub raw_content: String,
    /// Parsed messages
    pub messages: Vec<Message>,
    /// Total token count
    pub token_count: usize,
    /// Input cost per invocation
    pub input_cost: f64,
    /// Output cost per invocation (estimated)
    pub output_cost: f64,
    /// Total cost per invocation
    pub total_cost: f64,
    /// Whether this prompt exceeds model context limits
    pub exceeds_limit: bool,
    /// Normalized content for duplicate detection
    pub normalized_content: String,
}

/// Aggregated insights for a prompt library.
#[derive(Debug, Clone)]
pub struct LibraryInsights {
    /// Total number of prompts analyzed
    pub total_prompts: usize,
    /// Total tokens across all prompts
    pub total_tokens: usize,
    /// Total cost per invocation across all prompts
    pub total_cost: f64,
    /// Estimated monthly cost at specified invocation rate
    pub monthly_cost: f64,
    /// Token distribution histogram: (bucket_max, count)
    pub token_distribution: Vec<(usize, usize)>,
    /// Top N most expensive prompts
    pub top_expensive: Vec<PromptAnalysis>,
    /// Prompts that exceed model context limits
    pub exceeded_limits: Vec<PromptAnalysis>,
    /// Groups of duplicate prompts (each group is a list of file paths)
    pub duplicates: Vec<Vec<PathBuf>>,
}

impl LibraryInsights {
    /// Create empty insights.
    pub fn new() -> Self {
        Self {
            total_prompts: 0,
            total_tokens: 0,
            total_cost: 0.0,
            monthly_cost: 0.0,
            token_distribution: Vec::new(),
            top_expensive: Vec::new(),
            exceeded_limits: Vec::new(),
            duplicates: Vec::new(),
        }
    }
}

impl Default for LibraryInsights {
    fn default() -> Self {
        Self::new()
    }
}
