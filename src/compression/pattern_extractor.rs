/// Pattern extraction from prompt libraries
///
/// Identifies reusable context patterns across multiple prompts
use crate::compression::similarity::{normalize_text, normalized_similarity};
use crate::compression::types::{ContextLibrary, ContextPattern};
use crate::error::AppError;
use crate::tokenizers::Tokenizer;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Configuration for pattern extraction
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Minimum number of prompts a pattern must appear in
    pub min_frequency: usize,
    /// Minimum similarity threshold (0.0 to 1.0)
    pub min_similarity: f64,
    /// Categories to extract
    #[allow(dead_code)] // Public API field
    pub categories: Vec<String>,
    /// Minimum tokens for a pattern to be considered
    pub min_tokens: usize,
    /// Maximum tokens for a pattern
    pub max_tokens: usize,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_frequency: 2,
            min_similarity: 0.85,
            categories: vec![
                "role".to_string(),
                "examples".to_string(),
                "constraints".to_string(),
            ],
            min_tokens: 20,
            max_tokens: 500,
        }
    }
}

/// Pattern extractor
pub struct PatternExtractor {
    config: ExtractionConfig,
    tokenizer: Box<dyn Tokenizer>,
}

impl PatternExtractor {
    /// Create a new pattern extractor
    pub fn new(tokenizer: Box<dyn Tokenizer>, config: ExtractionConfig) -> Self {
        Self { config, tokenizer }
    }

    /// Extract patterns from a directory of prompts
    pub fn extract_from_directory<P: AsRef<Path>>(
        &self,
        directory: P,
    ) -> Result<ContextLibrary, AppError> {
        let mut prompts = Vec::new();

        // Read all prompt files
        for entry in WalkDir::new(&directory)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() && is_prompt_file(path) {
                match fs::read_to_string(path) {
                    Ok(content) => prompts.push((path.display().to_string(), content)),
                    Err(e) => eprintln!("Warning: Could not read {}: {}", path.display(), e),
                }
            }
        }

        if prompts.is_empty() {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "No prompt files found in directory".to_string(),
            )));
        }

        self.extract_from_prompts(&prompts, directory.as_ref())
    }

    /// Extract patterns from a list of prompts
    fn extract_from_prompts(
        &self,
        prompts: &[(String, String)],
        source_dir: &Path,
    ) -> Result<ContextLibrary, AppError> {
        let mut library = ContextLibrary::new();
        library.metadata.source_directory = Some(source_dir.display().to_string());

        // Extract candidates from each prompt
        let mut candidates: Vec<PatternCandidate> = Vec::new();

        for (_path, content) in prompts {
            let extracted = self.extract_candidates_from_text(content);
            candidates.extend(extracted);
        }

        // Group similar candidates
        let grouped = self.group_similar_candidates(&candidates);

        // Filter by frequency and convert to patterns
        let mut pattern_id = 1;
        for group in grouped {
            if group.len() >= self.config.min_frequency {
                // Use the longest version as the canonical content
                let canonical = group.iter().max_by_key(|c| c.content.len()).unwrap();

                let tokens = self.tokenizer.count_tokens(&canonical.content)?;
                if tokens < self.config.min_tokens || tokens > self.config.max_tokens {
                    continue;
                }

                let category = self.categorize_pattern(&canonical.content);
                let id = format!("{}_{:03}", category, pattern_id);
                pattern_id += 1;

                let mut pattern = ContextPattern::new(id, canonical.content.clone(), category);
                pattern.frequency = group.len();
                pattern.avg_tokens = tokens;

                library.add_pattern(pattern);
            }
        }

        Ok(library)
    }

    /// Extract candidate patterns from text
    fn extract_candidates_from_text(&self, text: &str) -> Vec<PatternCandidate> {
        let mut candidates = Vec::new();

        // Extract paragraphs as candidates
        let paragraphs: Vec<&str> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        for paragraph in paragraphs {
            let normalized = normalize_text(paragraph);
            if normalized.split_whitespace().count() < 10 {
                continue; // Too short
            }

            candidates.push(PatternCandidate {
                content: paragraph.trim().to_string(),
                normalized: normalized.clone(),
                hash: calculate_hash(&normalized),
            });
        }

        // Extract sentences as candidates (for examples)
        for sentence in text.split('.') {
            let trimmed = sentence.trim();
            if trimmed.is_empty() {
                continue;
            }

            let normalized = normalize_text(trimmed);
            let word_count = normalized.split_whitespace().count();

            if (5..=50).contains(&word_count) {
                candidates.push(PatternCandidate {
                    content: trimmed.to_string(),
                    normalized: normalized.clone(),
                    hash: calculate_hash(&normalized),
                });
            }
        }

        candidates
    }

    /// Group similar candidates together
    fn group_similar_candidates<'a>(
        &self,
        candidates: &'a [PatternCandidate],
    ) -> Vec<Vec<&'a PatternCandidate>> {
        let mut groups: Vec<Vec<&PatternCandidate>> = Vec::new();
        let mut used: HashSet<usize> = HashSet::new();

        for (i, candidate) in candidates.iter().enumerate() {
            if used.contains(&i) {
                continue;
            }

            let mut group = vec![candidate];
            used.insert(i);

            // Find similar candidates
            for (j, other) in candidates.iter().enumerate().skip(i + 1) {
                if used.contains(&j) {
                    continue;
                }

                // Quick hash check
                if candidate.hash == other.hash {
                    group.push(other);
                    used.insert(j);
                } else {
                    // Fuzzy match
                    let similarity =
                        normalized_similarity(&candidate.normalized, &other.normalized);
                    if similarity >= self.config.min_similarity {
                        group.push(other);
                        used.insert(j);
                    }
                }
            }

            groups.push(group);
        }

        groups
    }

    /// Categorize a pattern based on its content
    fn categorize_pattern(&self, content: &str) -> String {
        let lower = content.to_lowercase();

        // Role indicators
        if lower.contains("you are") || lower.contains("your role") || lower.contains("expert") {
            return "role".to_string();
        }

        // Example indicators
        if lower.contains("example") || lower.contains("instance") || lower.contains("e.g.") {
            return "examples".to_string();
        }

        // Constraint indicators
        if lower.contains("must") || lower.contains("should not") || lower.contains("constraint") {
            return "constraints".to_string();
        }

        // Default
        "general".to_string()
    }
}

/// A candidate pattern during extraction
#[derive(Debug, Clone)]
struct PatternCandidate {
    content: String,
    normalized: String,
    hash: u64,
}

/// Calculate a simple hash of text
fn calculate_hash(text: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Check if a file is a prompt file based on extension
fn is_prompt_file(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();
        matches!(
            ext_str.as_str(),
            "txt" | "md" | "markdown" | "json" | "yaml" | "yml" | "prompt" | "hieratic" | "hrt"
        )
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizers::openai::OpenAITokenizer;

    fn create_test_tokenizer() -> Box<dyn Tokenizer> {
        Box::new(OpenAITokenizer::new("gpt-4").unwrap())
    }

    #[test]
    fn test_extract_candidates() {
        let tokenizer = create_test_tokenizer();
        let config = ExtractionConfig::default();
        let extractor = PatternExtractor::new(tokenizer, config);

        let text = "You are an expert programmer.\n\nExample 1: Fixed a bug.\n\nExample 2: Optimized code.";
        let candidates = extractor.extract_candidates_from_text(text);

        assert!(candidates.len() >= 2);
    }

    #[test]
    fn test_categorize_pattern() {
        let tokenizer = create_test_tokenizer();
        let config = ExtractionConfig::default();
        let extractor = PatternExtractor::new(tokenizer, config);

        assert_eq!(
            extractor.categorize_pattern("You are an expert developer"),
            "role"
        );
        assert_eq!(
            extractor.categorize_pattern("Example: Fixed a bug"),
            "examples"
        );
        assert_eq!(
            extractor.categorize_pattern("You must not use profanity"),
            "constraints"
        );
    }

    #[test]
    fn test_group_similar_candidates() {
        let tokenizer = create_test_tokenizer();
        let config = ExtractionConfig {
            min_similarity: 0.8,
            ..Default::default()
        };
        let extractor = PatternExtractor::new(tokenizer, config);

        let candidates = vec![
            PatternCandidate {
                content: "You are an expert".to_string(),
                normalized: "you are an expert".to_string(),
                hash: calculate_hash("you are an expert"),
            },
            PatternCandidate {
                content: "You are an expert".to_string(),
                normalized: "you are an expert".to_string(),
                hash: calculate_hash("you are an expert"),
            },
            PatternCandidate {
                content: "Different content".to_string(),
                normalized: "different content".to_string(),
                hash: calculate_hash("different content"),
            },
        ];

        let groups = extractor.group_similar_candidates(&candidates);

        // Should have 2 groups: one with 2 similar items, one with 1 different item
        assert!(groups.len() >= 2);
    }

    #[test]
    fn test_is_prompt_file() {
        assert!(is_prompt_file(Path::new("test.txt")));
        assert!(is_prompt_file(Path::new("test.md")));
        assert!(is_prompt_file(Path::new("test.json")));
        assert!(is_prompt_file(Path::new("test.yaml")));
        assert!(is_prompt_file(Path::new("test.hieratic")));
        assert!(!is_prompt_file(Path::new("test.rs")));
        assert!(!is_prompt_file(Path::new("test.exe")));
    }
}
