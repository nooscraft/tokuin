use std::path::PathBuf;
/// Unit tests for analyzers module.
use tokuin::analyzers::duplicates::{detect_duplicates, normalize_content};
use tokuin::analyzers::scanner::PromptScanner;
use tokuin::analyzers::types::PromptAnalysis;
use tokuin::parsers::Message;

#[test]
fn test_normalize_content() {
    let input = "  Hello  \n\n  World  \n";
    let normalized = normalize_content(input);
    assert_eq!(normalized, "hello\nworld");
}

#[test]
fn test_normalize_content_empty() {
    let input = "   \n\n  \n";
    let normalized = normalize_content(input);
    assert_eq!(normalized, "");
}

#[test]
fn test_normalize_content_single_line() {
    let input = "Single line prompt";
    let normalized = normalize_content(input);
    assert_eq!(normalized, "single line prompt");
}

fn create_test_analysis(path: &str, content: &str, tokens: usize, cost: f64) -> PromptAnalysis {
    PromptAnalysis {
        file_path: PathBuf::from(path),
        prompt_id: path.to_string(),
        raw_content: content.to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: content.to_string(),
        }],
        token_count: tokens,
        input_cost: cost,
        output_cost: 0.0,
        total_cost: cost,
        exceeds_limit: false,
        normalized_content: normalize_content(content),
    }
}

#[test]
fn test_detect_duplicates_exact() {
    let analyses = vec![
        create_test_analysis("file1.txt", "Hello world", 10, 0.01),
        create_test_analysis("file2.txt", "Hello world", 10, 0.01),
        create_test_analysis("file3.txt", "Different content", 15, 0.02),
    ];

    let duplicates = detect_duplicates(&analyses);
    assert_eq!(duplicates.len(), 1, "Should find one duplicate group");
    assert_eq!(duplicates[0].len(), 2, "Group should have 2 files");
}

#[test]
fn test_detect_duplicates_with_whitespace() {
    let analyses = vec![
        create_test_analysis("file1.txt", "Hello\nworld", 10, 0.01),
        create_test_analysis("file2.txt", "  Hello  \n  world  ", 10, 0.01),
    ];

    let duplicates = detect_duplicates(&analyses);
    assert_eq!(
        duplicates.len(),
        1,
        "Should detect duplicates despite whitespace"
    );
}

#[test]
fn test_detect_duplicates_no_duplicates() {
    let analyses = vec![
        create_test_analysis("file1.txt", "Hello world", 10, 0.01),
        create_test_analysis("file2.txt", "Different content", 15, 0.02),
        create_test_analysis("file3.txt", "Another different", 20, 0.03),
    ];

    let duplicates = detect_duplicates(&analyses);
    assert!(duplicates.is_empty(), "Should find no duplicates");
}

#[test]
fn test_calculate_distribution() {
    let analyses = vec![
        create_test_analysis("file1.txt", "test", 50, 0.01),
        create_test_analysis("file2.txt", "test", 150, 0.01),
        create_test_analysis("file3.txt", "test", 600, 0.01),
        create_test_analysis("file4.txt", "test", 2000, 0.01),
        create_test_analysis("file5.txt", "test", 60000, 0.01),
    ];

    let dist = PromptScanner::calculate_distribution(&analyses);
    assert!(!dist.is_empty(), "Should have distribution");
    assert!(
        dist.iter().any(|(_, count)| *count > 0),
        "Should have counts"
    );
}

#[test]
fn test_generate_insights_empty() {
    let insights = PromptScanner::generate_insights(&[], 10, 1000);
    assert_eq!(insights.total_prompts, 0);
    assert_eq!(insights.total_tokens, 0);
    assert_eq!(insights.total_cost, 0.0);
}

#[test]
fn test_generate_insights_basic() {
    let analyses = vec![
        create_test_analysis("file1.txt", "test", 100, 0.01),
        create_test_analysis("file2.txt", "test", 200, 0.02),
        create_test_analysis("file3.txt", "test", 300, 0.03),
    ];

    let insights = PromptScanner::generate_insights(&analyses, 2, 1000);
    assert_eq!(insights.total_prompts, 3);
    assert_eq!(insights.total_tokens, 600);
    assert_eq!(insights.total_cost, 0.06);
    assert_eq!(insights.monthly_cost, 60.0);
    assert_eq!(insights.top_expensive.len(), 2);
}

#[test]
fn test_generate_insights_exceeded_limits() {
    let mut analysis = create_test_analysis("file1.txt", "test", 10000, 0.01);
    analysis.exceeds_limit = true;
    let analyses = vec![analysis];

    let insights = PromptScanner::generate_insights(&analyses, 10, 1000);
    assert_eq!(insights.exceeded_limits.len(), 1);
}

#[test]
fn test_generate_insights_duplicates() {
    let analyses = vec![
        create_test_analysis("file1.txt", "Duplicate content", 100, 0.01),
        create_test_analysis("file2.txt", "Duplicate content", 100, 0.01),
        create_test_analysis("file3.txt", "Different", 200, 0.02),
    ];

    let insights = PromptScanner::generate_insights(&analyses, 10, 1000);
    assert_eq!(insights.duplicates.len(), 1);
    assert_eq!(insights.duplicates[0].len(), 2);
}
