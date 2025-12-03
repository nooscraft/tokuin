/// Quality metrics for compression fidelity
///
/// This module provides metrics to measure how well a compressed prompt
/// preserves the semantic meaning and critical information of the original.
use crate::compression::hieratic_decoder::HieraticDecoder;
use crate::error::AppError;
use serde::{Deserialize, Serialize};

/// Quality metrics for a compression result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
    /// Semantic similarity between original and expanded compressed prompt
    pub semantic_similarity: f64,
    /// Percentage of critical instructions preserved
    pub critical_instruction_preservation: f64,
    /// Information density retention (keyword/entity coverage)
    pub information_retention: f64,
    /// Structural integrity (for structured documents)
    pub structural_integrity: f64,
    /// Number of critical patterns found in original
    pub critical_patterns_found: usize,
    /// Number of critical patterns preserved
    pub critical_patterns_preserved: usize,
}

impl QualityMetrics {
    /// Create default (no quality data available)
    pub fn unknown() -> Self {
        Self {
            overall_score: 0.0,
            semantic_similarity: 0.0,
            critical_instruction_preservation: 0.0,
            information_retention: 0.0,
            structural_integrity: 0.0,
            critical_patterns_found: 0,
            critical_patterns_preserved: 0,
        }
    }

    /// Get quality rating as string
    pub fn rating(&self) -> &'static str {
        match self.overall_score {
            x if x >= 0.9 => "Excellent",
            x if x >= 0.75 => "Good",
            x if x >= 0.6 => "Fair",
            x if x >= 0.4 => "Poor",
            _ => "Very Poor",
        }
    }

    /// Check if quality is acceptable (>= 0.7)
    pub fn is_acceptable(&self) -> bool {
        self.overall_score >= 0.7
    }
}

/// Critical instruction patterns that must be preserved
const CRITICAL_PATTERNS: &[&str] = &[
    "response format",
    "separator is always",
    "do not output",
    "extracted_value",
    "doc_page_number",
    "must include",
    "required field",
    "output format",
    "json schema",
    "xml structure",
];

/// Calculate quality metrics for a compression result
pub fn calculate_quality_metrics(
    original: &str,
    compressed_hieratic: &str,
    context_lib_path: Option<&str>,
) -> Result<QualityMetrics, AppError> {
    // Expand the compressed prompt back to full text
    let mut decoder = HieraticDecoder::new()?;

    if let Some(path) = context_lib_path {
        decoder = decoder.load_context_library(path)?;
    }

    let expanded = decoder.decode(compressed_hieratic)?;

    // 1. Semantic similarity (placeholder - requires embeddings)
    let semantic_similarity = calculate_semantic_similarity(original, &expanded)?;

    // 2. Critical instruction preservation
    let (found, preserved) = check_critical_instructions(original, &expanded);
    let critical_instruction_preservation = if found > 0 {
        preserved as f64 / found as f64
    } else {
        1.0 // No critical patterns means nothing to preserve
    };

    // 3. Information retention (keyword-based)
    let information_retention = calculate_information_retention(original, &expanded)?;

    // 4. Structural integrity
    let structural_integrity = check_structural_integrity(original, &expanded);

    // Calculate overall score (weighted average)
    let overall_score = (semantic_similarity * 0.4)
        + (critical_instruction_preservation * 0.3)
        + (information_retention * 0.2)
        + (structural_integrity * 0.1);

    Ok(QualityMetrics {
        overall_score,
        semantic_similarity,
        critical_instruction_preservation,
        information_retention,
        structural_integrity,
        critical_patterns_found: found,
        critical_patterns_preserved: preserved,
    })
}

/// Calculate semantic similarity using embeddings (if available)
fn calculate_semantic_similarity(original: &str, expanded: &str) -> Result<f64, AppError> {
    // Try to use embeddings if available
    #[cfg(feature = "compression-embeddings")]
    {
        use crate::compression::embeddings::{
            cosine_similarity, EmbeddingProvider, OnnxEmbeddingProvider,
        };

        if let Ok(provider) = OnnxEmbeddingProvider::new() {
            if provider.is_available() {
                let orig_emb = EmbeddingProvider::embed(&provider, original)?;
                let exp_emb = EmbeddingProvider::embed(&provider, expanded)?;
                return Ok(cosine_similarity(&orig_emb, &exp_emb).max(0.0));
            }
        }
    }

    // Fallback to text similarity
    use crate::compression::similarity::normalized_similarity;

    // Compare normalized versions
    let similarity = normalized_similarity(original, expanded);

    // Boost similarity score slightly (expanded text should be similar to original)
    Ok(similarity.min(1.0))
}

/// Check if critical instructions are preserved
fn check_critical_instructions(original: &str, expanded: &str) -> (usize, usize) {
    let original_lower = original.to_lowercase();
    let expanded_lower = expanded.to_lowercase();

    let mut found = 0;
    let mut preserved = 0;

    for pattern in CRITICAL_PATTERNS {
        if original_lower.contains(pattern) {
            found += 1;
            if expanded_lower.contains(pattern) {
                preserved += 1;
            }
        }
    }

    (found, preserved)
}

/// Calculate information retention based on keyword/entity coverage
fn calculate_information_retention(original: &str, expanded: &str) -> Result<f64, AppError> {
    // Extract important keywords (non-stop words, capitalized terms, etc.)
    let original_keywords = extract_keywords(original);
    let expanded_keywords = extract_keywords(expanded);

    if original_keywords.is_empty() {
        return Ok(1.0);
    }

    // Calculate Jaccard similarity of keywords
    let intersection: usize = original_keywords
        .iter()
        .filter(|k| expanded_keywords.contains(k.as_str()))
        .count();

    let union = original_keywords.len() + expanded_keywords.len() - intersection;

    let jaccard = if union > 0 {
        intersection as f64 / union as f64
    } else {
        0.0
    };

    Ok(jaccard)
}

/// Extract important keywords from text
fn extract_keywords(text: &str) -> std::collections::HashSet<String> {
    use std::collections::HashSet;

    // Simple stop words list
    let stop_words: HashSet<&str> = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "should", "could", "may", "might", "must", "can",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
    ]
    .iter()
    .cloned()
    .collect();

    text.split_whitespace()
        .map(|w| w.to_lowercase())
        .map(|w| {
            // Remove punctuation
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| w.len() > 2 && !stop_words.contains(w.as_str()))
        .collect()
}

/// Check structural integrity (JSON, HTML, tables preserved)
fn check_structural_integrity(original: &str, expanded: &str) -> f64 {
    // Check for JSON structures
    let original_has_json = original.contains('{') && original.contains('}');
    let expanded_has_json = expanded.contains('{') && expanded.contains('}');

    // Check for HTML/XML tags
    let original_has_html = original.contains('<') && original.contains('>');
    let expanded_has_html = expanded.contains('<') && expanded.contains('>');

    // Check for table markers
    let original_has_table = original.contains("|") || original.contains("\t");
    let expanded_has_table = expanded.contains("|") || expanded.contains("\t");

    let mut score = 1.0;

    if original_has_json && !expanded_has_json {
        score *= 0.5; // JSON structure lost
    }
    if original_has_html && !expanded_has_html {
        score *= 0.5; // HTML structure lost
    }
    if original_has_table && !expanded_has_table {
        score *= 0.5; // Table structure lost
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_instructions() {
        let original = "Please provide the response format: JSON. Do not output XML.";
        let expanded = "Provide response format: JSON. Do not output XML.";

        let (found, preserved) = check_critical_instructions(original, expanded);
        assert!(found >= 2); // Should find "response format" and "do not output"
        assert!(preserved >= 2);
    }

    #[test]
    fn test_information_retention() {
        let original = "Analyze the security vulnerabilities in the authentication system";
        let expanded = "Analyze security vulnerabilities authentication system";

        let retention = calculate_information_retention(original, expanded).unwrap();
        assert!(retention > 0.7); // Most keywords should be preserved
    }

    #[test]
    fn test_structural_integrity() {
        let original = "Here is JSON: {\"key\": \"value\"}";
        let expanded = "Here is JSON: {\"key\": \"value\"}";

        let integrity = check_structural_integrity(original, expanded);
        assert_eq!(integrity, 1.0);

        let original_with_json = "JSON: {\"key\": \"value\"}";
        let expanded_no_json = "JSON data";

        let integrity_lost = check_structural_integrity(original_with_json, expanded_no_json);
        assert!(integrity_lost < 1.0);
    }
}
