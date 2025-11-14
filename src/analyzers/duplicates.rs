/// Duplicate prompt detection.
use std::collections::HashMap;
use std::path::PathBuf;

/// Normalize content for duplicate detection.
pub fn normalize_content(content: &str) -> String {
    content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase()
}

/// Detect duplicate prompts based on normalized content.
pub fn detect_duplicates(
    analyses: &[crate::analyzers::types::PromptAnalysis],
) -> Vec<Vec<PathBuf>> {
    let mut content_map: HashMap<String, Vec<PathBuf>> = HashMap::new();

    // Group files by normalized content
    for analysis in analyses {
        let normalized = normalize_content(&analysis.normalized_content);
        content_map
            .entry(normalized)
            .or_insert_with(Vec::new)
            .push(analysis.file_path.clone());
    }

    // Return groups with more than one file
    content_map
        .into_values()
        .filter(|group| group.len() > 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzers::types::PromptAnalysis;
    use crate::parsers::Message;

    fn create_analysis(path: &str, content: &str) -> PromptAnalysis {
        PromptAnalysis {
            file_path: PathBuf::from(path),
            prompt_id: path.to_string(),
            raw_content: content.to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: content.to_string(),
            }],
            token_count: 10,
            input_cost: 0.0,
            output_cost: 0.0,
            total_cost: 0.0,
            exceeds_limit: false,
            normalized_content: content.to_string(),
        }
    }

    #[test]
    fn test_normalize_content() {
        let input = "  Hello  \n\n  World  \n";
        let normalized = normalize_content(input);
        assert_eq!(normalized, "hello\nworld");
    }

    #[test]
    fn test_detect_duplicates_exact() {
        let analyses = vec![
            create_analysis("file1.txt", "Hello world"),
            create_analysis("file2.txt", "Hello world"),
            create_analysis("file3.txt", "Different content"),
        ];

        let duplicates = detect_duplicates(&analyses);
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].len(), 2);
    }

    #[test]
    fn test_detect_duplicates_with_whitespace() {
        let analyses = vec![
            create_analysis("file1.txt", "Hello\nworld"),
            create_analysis("file2.txt", "  Hello  \n  world  "),
        ];

        let duplicates = detect_duplicates(&analyses);
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].len(), 2);
    }

    #[test]
    fn test_detect_duplicates_no_duplicates() {
        let analyses = vec![
            create_analysis("file1.txt", "Hello world"),
            create_analysis("file2.txt", "Different content"),
            create_analysis("file3.txt", "Another different"),
        ];

        let duplicates = detect_duplicates(&analyses);
        assert!(duplicates.is_empty());
    }
}
