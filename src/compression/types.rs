/// Core data structures for Hieratic prompt compression
use serde::{Deserialize, Serialize};

#[cfg(feature = "compression")]
use chrono::{DateTime, Utc};
#[cfg(feature = "compression")]
use uuid::Uuid;

/// Compression level determines aggressiveness of token reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionLevel {
    /// Light compression (30-50% reduction)
    Light,
    /// Medium compression (50-70% reduction)
    Medium,
    /// Aggressive compression (70-90% reduction)
    Aggressive,
}

impl Default for CompressionLevel {
    fn default() -> Self {
        CompressionLevel::Medium
    }
}

impl std::str::FromStr for CompressionLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "light" => Ok(CompressionLevel::Light),
            "medium" => Ok(CompressionLevel::Medium),
            "aggressive" => Ok(CompressionLevel::Aggressive),
            _ => Err(format!("Invalid compression level: {}", s)),
        }
    }
}

/// Section types in a Hieratic document
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HieraticSection {
    /// @ROLE directive
    Role { id: String, content: Option<String> },
    /// @EXAMPLES directive
    Examples { id: String, content: Option<String> },
    /// @CONSTRAINTS directive
    Constraints { id: String, content: Option<String> },
    /// @TASK directive (main instruction)
    Task { content: String },
    /// @FOCUS directive
    Focus { areas: Vec<String> },
    /// @STYLE directive
    Style { preferences: Vec<String> },
    /// @FORMAT directive
    Format { structure: String },
    /// @CONTEXT directive (reference to context library)
    Context { path: String },
}

/// Complete Hieratic document representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HieraticDocument {
    /// Format version (e.g., "1.0")
    pub version: String,
    /// Optional context library path
    pub context_path: Option<String>,
    /// Document sections in order
    pub sections: Vec<HieraticSection>,
}

impl HieraticDocument {
    /// Create a new Hieratic document with default version
    pub fn new() -> Self {
        Self {
            version: "1.0".to_string(),
            context_path: None,
            sections: Vec::new(),
        }
    }

    /// Add a section to the document
    pub fn add_section(&mut self, section: HieraticSection) {
        self.sections.push(section);
    }

    /// Get the task section content if it exists
    pub fn get_task(&self) -> Option<&str> {
        self.sections.iter().find_map(|section| match section {
            HieraticSection::Task { content } => Some(content.as_str()),
            _ => None,
        })
    }

    /// Get role content if it exists
    pub fn get_role(&self) -> Option<(&str, Option<&str>)> {
        self.sections.iter().find_map(|section| match section {
            HieraticSection::Role { id, content } => {
                Some((id.as_str(), content.as_ref().map(|s| s.as_str())))
            }
            _ => None,
        })
    }
}

impl Default for HieraticDocument {
    fn default() -> Self {
        Self::new()
    }
}

/// A reusable context pattern extracted from prompt library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPattern {
    /// Unique identifier for this pattern
    pub id: String,
    /// The full content of the pattern
    pub content: String,
    /// Number of prompts this pattern appears in
    pub frequency: usize,
    /// Average token count of this pattern
    pub avg_tokens: usize,
    /// Category (role, examples, constraints, etc.)
    pub category: String,
    /// Optional tags for organization
    pub tags: Vec<String>,
}

impl ContextPattern {
    /// Create a new context pattern
    pub fn new(id: String, content: String, category: String) -> Self {
        Self {
            id,
            content,
            frequency: 1,
            avg_tokens: 0,
            category,
            tags: Vec::new(),
        }
    }
}

/// Context library containing reusable patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextLibrary {
    /// Library metadata
    pub metadata: LibraryMetadata,
    /// Collection of patterns
    pub patterns: Vec<ContextPattern>,
}

/// Metadata for a context library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryMetadata {
    /// Format version
    pub version: String,
    /// Format type (always "hieratic")
    pub format: String,
    /// Creation timestamp
    pub created_at: String,
    /// Source directory that was scanned
    pub source_directory: Option<String>,
    /// Total number of patterns
    pub total_patterns: usize,
}

impl ContextLibrary {
    /// Create a new empty context library
    pub fn new() -> Self {
        Self {
            metadata: LibraryMetadata {
                version: "1.0".to_string(),
                format: "hieratic".to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                source_directory: None,
                total_patterns: 0,
            },
            patterns: Vec::new(),
        }
    }

    /// Add a pattern to the library
    pub fn add_pattern(&mut self, pattern: ContextPattern) {
        self.patterns.push(pattern);
        self.metadata.total_patterns = self.patterns.len();
    }

    /// Find a pattern by ID
    pub fn get_pattern(&self, id: &str) -> Option<&ContextPattern> {
        self.patterns.iter().find(|p| p.id == id)
    }

    /// Get all patterns of a specific category
    pub fn patterns_by_category(&self, category: &str) -> Vec<&ContextPattern> {
        self.patterns
            .iter()
            .filter(|p| p.category == category)
            .collect()
    }
}

impl Default for ContextLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Compression anchor for incremental compression
///
/// Anchors mark points in the conversation/document where compression has been applied.
/// This allows subsequent compressions to only process new content (delta compression).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionAnchor {
    /// Unique identifier for this anchor
    pub id: String,
    /// Token position in the original stream where this anchor ends
    pub token_position: usize,
    /// Compressed summary of the content up to this point
    pub summary: String,
    /// Number of tokens produced by the summary
    pub summary_tokens: usize,
    /// Timestamp when anchor was created
    #[cfg(feature = "compression")]
    pub timestamp: DateTime<Utc>,
    /// Number of original tokens prior to this anchor
    pub original_tokens_before: usize,
    /// Compression ratio at this anchor
    pub compression_ratio: f64,
}

impl CompressionAnchor {
    /// Create a new anchor from a summary
    #[cfg(feature = "compression")]
    pub fn new(
        token_position: usize,
        summary: String,
        summary_tokens: usize,
        original_tokens_before: usize,
        compression_ratio: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            token_position,
            summary,
            summary_tokens,
            timestamp: Utc::now(),
            original_tokens_before,
            compression_ratio,
        }
    }

    /// Create an anchor representing the entire base document (first run)
    #[cfg(feature = "compression")]
    pub fn from_base_document(summary: String, summary_tokens: usize, total_tokens: usize) -> Self {
        let compression_ratio = if total_tokens > 0 {
            summary_tokens as f64 / total_tokens as f64
        } else {
            0.0
        };

        Self::new(total_tokens, summary, summary_tokens, 0, compression_ratio)
    }
}

/// Result of compressing a prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionResult {
    /// Original token count
    pub original_tokens: usize,
    /// Compressed token count
    pub compressed_tokens: usize,
    /// Compression ratio (0.0 to 1.0)
    pub compression_ratio: f64,
    /// Number of tokens saved
    pub tokens_saved: usize,
    /// Context references used
    pub context_refs: Vec<ContextReference>,
    /// Extractive compression stats
    pub extractive_stats: ExtractionStats,
    /// The compressed Hieratic document
    pub document: HieraticDocument,
    /// Compression anchors for incremental compression
    pub anchors: Vec<CompressionAnchor>,
    /// Quality metrics (if calculated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_metrics: Option<crate::compression::quality::QualityMetrics>,
}

impl CompressionResult {
    /// Calculate compression percentage
    pub fn compression_percentage(&self) -> f64 {
        self.compression_ratio * 100.0
    }

    /// Estimate cost savings for a given model and invocation count
    pub fn estimate_savings(
        &self,
        input_price_per_1k: f64,
        monthly_invocations: u64,
    ) -> CostSavings {
        let original_cost = (self.original_tokens as f64 / 1000.0)
            * input_price_per_1k
            * monthly_invocations as f64;
        let compressed_cost = (self.compressed_tokens as f64 / 1000.0)
            * input_price_per_1k
            * monthly_invocations as f64;
        let savings = original_cost - compressed_cost;

        CostSavings {
            original_cost,
            compressed_cost,
            savings,
            savings_percentage: (savings / original_cost) * 100.0,
        }
    }
}

/// Context reference used in compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextReference {
    /// Pattern ID
    pub id: String,
    /// Category (role, examples, etc.)
    pub category: String,
    /// Original token count
    pub original_tokens: usize,
    /// Compressed token count (reference length)
    pub compressed_tokens: usize,
    /// Tokens saved
    pub tokens_saved: usize,
}

/// Statistics about extractive compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionStats {
    /// Tokens removed from low-relevance sentences
    pub low_relevance_removed: usize,
    /// Tokens removed from redundant phrases
    pub redundant_removed: usize,
    /// Number of sections compressed
    pub sections_compressed: usize,
}

impl Default for ExtractionStats {
    fn default() -> Self {
        Self {
            low_relevance_removed: 0,
            redundant_removed: 0,
            sections_compressed: 0,
        }
    }
}

/// Cost savings calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSavings {
    /// Original monthly cost
    pub original_cost: f64,
    /// Compressed monthly cost
    pub compressed_cost: f64,
    /// Total savings
    pub savings: f64,
    /// Savings percentage
    pub savings_percentage: f64,
}

/// Pattern match found during extraction
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// The pattern that was matched
    pub pattern: ContextPattern,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
}

/// Configuration for compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression level
    pub level: CompressionLevel,
    /// Path to context library (if any)
    pub context_library_path: Option<String>,
    /// Force inline mode (no references)
    pub force_inline: bool,
    /// Minimum similarity for pattern matching (0.0 to 1.0)
    pub min_similarity: f64,
    /// Target compression ratio
    pub target_ratio: Option<f64>,
    /// Enable structured document mode (better handling of JSON, code, tables)
    pub structured_mode: bool,
    /// Enable incremental compression mode
    pub incremental_mode: bool,
    /// Token threshold for creating anchors (default: 1000 tokens)
    pub anchor_threshold: usize,
    /// Token retention threshold (how many recent tokens to keep uncompressed)
    pub retention_threshold: usize,
    /// Previous compression result for incremental mode
    pub previous_result: Option<Box<CompressionResult>>,
    /// Scoring mode (heuristic, semantic, or hybrid)
    pub scoring_mode: ScoringMode,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            level: CompressionLevel::Medium,
            context_library_path: None,
            force_inline: false,
            min_similarity: 0.85,
            target_ratio: None,
            structured_mode: false,
            incremental_mode: false,
            anchor_threshold: 1000,
            retention_threshold: 500,
            previous_result: None,
            scoring_mode: ScoringMode::Heuristic,
        }
    }
}

/// Scoring mode for compression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoringMode {
    /// Heuristic-based scoring (keyword matching, position-based)
    Heuristic,
    /// Embedding-based semantic scoring (requires compression-embeddings feature)
    Semantic,
    /// Hybrid: combine embeddings and heuristics (best of both)
    Hybrid,
}

impl Default for ScoringMode {
    fn default() -> Self {
        ScoringMode::Heuristic
    }
}

impl std::str::FromStr for ScoringMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "heuristic" => Ok(ScoringMode::Heuristic),
            "semantic" => Ok(ScoringMode::Semantic),
            "hybrid" => Ok(ScoringMode::Hybrid),
            _ => Err(format!("Invalid scoring mode: {}", s)),
        }
    }
}

/// Output format for compression results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Hieratic format (.hieratic)
    Hieratic,
    /// Expanded full text
    Expanded,
    /// JSON format
    Json,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hieratic" => Ok(OutputFormat::Hieratic),
            "expanded" => Ok(OutputFormat::Expanded),
            "json" => Ok(OutputFormat::Json),
            _ => Err(format!("Invalid output format: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_level_from_str() {
        assert_eq!(
            "light".parse::<CompressionLevel>().unwrap(),
            CompressionLevel::Light
        );
        assert_eq!(
            "medium".parse::<CompressionLevel>().unwrap(),
            CompressionLevel::Medium
        );
        assert_eq!(
            "aggressive".parse::<CompressionLevel>().unwrap(),
            CompressionLevel::Aggressive
        );
        assert!("invalid".parse::<CompressionLevel>().is_err());
    }

    #[test]
    fn test_hieratic_document_creation() {
        let mut doc = HieraticDocument::new();
        assert_eq!(doc.version, "1.0");
        assert_eq!(doc.sections.len(), 0);

        doc.add_section(HieraticSection::Task {
            content: "Test task".to_string(),
        });
        assert_eq!(doc.sections.len(), 1);
        assert_eq!(doc.get_task(), Some("Test task"));
    }

    #[test]
    fn test_context_library_creation() {
        let mut lib = ContextLibrary::new();
        assert_eq!(lib.metadata.total_patterns, 0);

        let pattern = ContextPattern::new(
            "test_001".to_string(),
            "Test content".to_string(),
            "role".to_string(),
        );
        lib.add_pattern(pattern);

        assert_eq!(lib.metadata.total_patterns, 1);
        assert!(lib.get_pattern("test_001").is_some());
        assert!(lib.get_pattern("nonexistent").is_none());
    }

    #[test]
    fn test_compression_result_calculations() {
        let result = CompressionResult {
            original_tokens: 1000,
            compressed_tokens: 200,
            compression_ratio: 0.80,
            tokens_saved: 800,
            context_refs: Vec::new(),
            extractive_stats: ExtractionStats::default(),
            document: HieraticDocument::new(),
            anchors: Vec::new(),
            quality_metrics: None,
        };

        assert_eq!(result.compression_percentage(), 80.0);

        let savings = result.estimate_savings(0.01, 1000); // $0.01 per 1K tokens, 1000 invocations
        assert_eq!(savings.original_cost, 10.0); // 1000 tokens * $0.01 * 1000 invocations
        assert_eq!(savings.compressed_cost, 2.0); // 200 tokens * $0.01 * 1000 invocations
        assert_eq!(savings.savings, 8.0);
    }

    #[test]
    fn test_output_format_from_str() {
        assert_eq!(
            "hieratic".parse::<OutputFormat>().unwrap(),
            OutputFormat::Hieratic
        );
        assert_eq!(
            "expanded".parse::<OutputFormat>().unwrap(),
            OutputFormat::Expanded
        );
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert!("invalid".parse::<OutputFormat>().is_err());
    }
}
