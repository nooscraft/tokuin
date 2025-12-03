/// Text similarity utilities for pattern matching
use strsim::levenshtein;

/// Calculate similarity between two strings using Levenshtein distance
/// Returns a value between 0.0 (completely different) and 1.0 (identical)
pub fn calculate_similarity(text1: &str, text2: &str) -> f64 {
    let distance = levenshtein(text1, text2);
    let max_len = text1.len().max(text2.len());

    if max_len == 0 {
        return 1.0; // Both empty strings are identical
    }

    1.0 - (distance as f64 / max_len as f64)
}

/// Normalize text for comparison
/// - Converts to lowercase
/// - Removes extra whitespace
/// - Trims leading/trailing whitespace
pub fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Calculate normalized similarity (normalizes both texts before comparing)
pub fn normalized_similarity(text1: &str, text2: &str) -> f64 {
    let norm1 = normalize_text(text1);
    let norm2 = normalize_text(text2);
    calculate_similarity(&norm1, &norm2)
}

/// Check if two texts are similar above a threshold
#[allow(dead_code)] // Public API function
pub fn is_similar(text1: &str, text2: &str, threshold: f64) -> bool {
    normalized_similarity(text1, text2) >= threshold
}

/// Find the best matching substring in a larger text
/// Returns (best_match_start, best_match_end, similarity_score)
pub fn find_best_match(
    needle: &str,
    haystack: &str,
    min_similarity: f64,
) -> Option<(usize, usize, f64)> {
    let normalized_needle = normalize_text(needle);
    let needle_words: Vec<&str> = normalized_needle.split_whitespace().collect();
    let needle_len = needle_words.len();

    if needle_len == 0 {
        return None;
    }

    let haystack_normalized = normalize_text(haystack);
    let haystack_words: Vec<&str> = haystack_normalized.split_whitespace().collect();

    if haystack_words.len() < needle_len {
        return None;
    }

    let mut best_match: Option<(usize, usize, f64)> = None;
    let mut best_score = min_similarity;

    // Sliding window approach
    for start in 0..=(haystack_words.len() - needle_len) {
        let window = &haystack_words[start..start + needle_len];
        let window_text = window.join(" ");
        let score = calculate_similarity(&normalized_needle, &window_text);

        if score > best_score {
            // Find actual character positions in original haystack
            let (char_start, char_end) = find_char_positions(haystack, start, start + needle_len);
            best_match = Some((char_start, char_end, score));
            best_score = score;
        }
    }

    best_match
}

/// Find character positions for word indices in text
fn find_char_positions(text: &str, word_start: usize, word_end: usize) -> (usize, usize) {
    let normalized = normalize_text(text);
    let words: Vec<&str> = normalized.split_whitespace().collect();

    if words.is_empty() || word_start >= words.len() {
        return (0, text.len());
    }

    // This is approximate - we're finding positions in normalized text
    // For production, you'd want more precise mapping
    let prefix_len: usize = words[..word_start].iter().map(|w| w.len() + 1).sum();
    let selected_len: usize = words[word_start..word_end.min(words.len())]
        .iter()
        .map(|w| w.len() + 1)
        .sum();

    (prefix_len, prefix_len + selected_len)
}

/// Calculate Jaccard similarity (useful for comparing sets of words)
#[allow(dead_code)] // Public API function
pub fn jaccard_similarity(text1: &str, text2: &str) -> f64 {
    let norm1 = normalize_text(text1);
    let norm2 = normalize_text(text2);
    let words1: std::collections::HashSet<_> = norm1.split_whitespace().collect();
    let words2: std::collections::HashSet<_> = norm2.split_whitespace().collect();

    if words1.is_empty() && words2.is_empty() {
        return 1.0;
    }

    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

/// Calculate cosine similarity based on word frequency
#[allow(dead_code)] // Public API function
pub fn cosine_similarity(text1: &str, text2: &str) -> f64 {
    use std::collections::HashMap;

    let norm1 = normalize_text(text1);
    let norm2 = normalize_text(text2);

    let mut freq1: HashMap<&str, i32> = HashMap::new();
    let mut freq2: HashMap<&str, i32> = HashMap::new();

    for word in norm1.split_whitespace() {
        *freq1.entry(word).or_insert(0) += 1;
    }

    for word in norm2.split_whitespace() {
        *freq2.entry(word).or_insert(0) += 1;
    }

    if freq1.is_empty() || freq2.is_empty() {
        return 0.0;
    }

    // Calculate dot product
    let mut dot_product = 0.0;
    for (word, count1) in &freq1 {
        if let Some(count2) = freq2.get(word) {
            dot_product += (*count1 as f64) * (*count2 as f64);
        }
    }

    // Calculate magnitudes
    let mag1: f64 = freq1.values().map(|c| (c * c) as f64).sum::<f64>().sqrt();
    let mag2: f64 = freq2.values().map(|c| (c * c) as f64).sum::<f64>().sqrt();

    if mag1 == 0.0 || mag2 == 0.0 {
        return 0.0;
    }

    dot_product / (mag1 * mag2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_similarity() {
        assert_eq!(calculate_similarity("hello", "hello"), 1.0);
        assert_eq!(calculate_similarity("", ""), 1.0);
        assert!(calculate_similarity("hello", "hallo") > 0.7);
        assert!(calculate_similarity("hello", "world") < 0.5);
    }

    #[test]
    fn test_normalize_text() {
        assert_eq!(normalize_text("  Hello   World  "), "hello world");
        assert_eq!(
            normalize_text("Test\n\nMultiple\tWhitespace"),
            "test multiple whitespace"
        );
    }

    #[test]
    fn test_normalized_similarity() {
        assert_eq!(normalized_similarity("Hello World", "hello world"), 1.0);
        assert_eq!(normalized_similarity("  Test  ", "test"), 1.0);
    }

    #[test]
    fn test_is_similar() {
        assert!(is_similar("hello world", "Hello World", 0.9));
        assert!(!is_similar("hello world", "goodbye world", 0.9));
    }

    #[test]
    fn test_jaccard_similarity() {
        assert_eq!(jaccard_similarity("hello world", "hello world"), 1.0);
        assert_eq!(jaccard_similarity("hello world", "world hello"), 1.0);
        assert!(jaccard_similarity("hello world", "hello universe") > 0.3);
        assert!(jaccard_similarity("hello world", "goodbye universe") == 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        assert!(cosine_similarity("hello world", "hello world") > 0.99);
        assert!(cosine_similarity("hello world hello", "hello") > 0.8);
        assert!(cosine_similarity("hello", "world") == 0.0);
    }
}
