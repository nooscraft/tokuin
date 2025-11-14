/// Formatter for prompt library insights.
use crate::analyzers::types::LibraryInsights;
use serde::Serialize;

/// Formatter for library insights.
pub struct InsightsFormatter;

impl InsightsFormatter {
    /// Format insights as text.
    pub fn format_text(
        insights: &LibraryInsights,
        model: &str,
        context_limit: Option<usize>,
    ) -> String {
        let mut output = Vec::new();

        output.push("Prompt Library Insights".to_string());
        output.push("=".repeat(50).to_string());
        output.push(String::new());

        // Summary
        output.push("Summary".to_string());
        output.push("-".repeat(50).to_string());
        output.push(format!("Total Prompts: {}", insights.total_prompts));
        output.push(format!("Total Tokens: {}", insights.total_tokens));
        output.push(format!(
            "Estimated Cost (per invocation): ${:.4}",
            insights.total_cost
        ));
        output.push(format!("Monthly Cost: ${:.2}", insights.monthly_cost));
        output.push(String::new());

        // Token Distribution
        if !insights.token_distribution.is_empty() {
            output.push("Token Distribution".to_string());
            output.push("-".repeat(50).to_string());
            for (bucket_max, count) in &insights.token_distribution {
                let bucket_label = if *bucket_max == usize::MAX {
                    "50K+".to_string()
                } else if *bucket_max == 100 {
                    "0-100".to_string()
                } else {
                    // Find previous bucket
                    let buckets = [100, 500, 1000, 5000, 10000, 50000];
                    let prev = buckets
                        .iter()
                        .position(|&b| b == *bucket_max)
                        .map_or(0, |i| if i > 0 { buckets[i - 1] + 1 } else { 0 });
                    format!("{}-{}", prev, bucket_max)
                };
                let bar_length = ((*count as f64 / 10.0).ceil() as usize).max(1);
                let bar = "█".repeat(bar_length);
                output.push(format!("[{}] {:>6} {}", bucket_label, count, bar));
            }
            output.push(String::new());
        }

        // Top N Most Expensive
        if !insights.top_expensive.is_empty() {
            output.push(format!(
                "Top {} Most Expensive Prompts",
                insights.top_expensive.len()
            ));
            output.push("-".repeat(50).to_string());
            for (i, analysis) in insights.top_expensive.iter().enumerate() {
                let file_display = analysis.file_path.to_string_lossy();
                output.push(format!(
                    "{}. {} - {} tokens - ${:.4}/invocation",
                    i + 1,
                    file_display,
                    analysis.token_count,
                    analysis.total_cost
                ));
            }
            output.push(String::new());
        }

        // Exceeded Limits
        if !insights.exceeded_limits.is_empty() {
            output.push(format!(
                "Prompts Exceeding Context Limits ({}{})",
                model,
                context_limit
                    .map(|l| format!(": {} tokens", l))
                    .unwrap_or_default()
            ));
            output.push("-".repeat(50).to_string());
            for analysis in &insights.exceeded_limits {
                let file_display = analysis.file_path.to_string_lossy();
                if let Some(limit) = context_limit {
                    let excess = analysis.token_count - limit;
                    output.push(format!(
                        "⚠️  {} - {} tokens (exceeds by {})",
                        file_display, analysis.token_count, excess
                    ));
                } else {
                    output.push(format!(
                        "⚠️  {} - {} tokens",
                        file_display, analysis.token_count
                    ));
                }
            }
            output.push(String::new());
        }

        // Duplicates
        if !insights.duplicates.is_empty() {
            output.push("Duplicate Prompts Detected".to_string());
            output.push("-".repeat(50).to_string());
            for (group_idx, group) in insights.duplicates.iter().enumerate() {
                output.push(format!("Group {} ({} files):", group_idx + 1, group.len()));
                for path in group {
                    output.push(format!("  - {}", path.to_string_lossy()));
                }
            }
        }

        output.join("\n")
    }

    /// Format insights as JSON.
    pub fn format_json(insights: &LibraryInsights) -> Result<String, serde_json::Error> {
        #[derive(Serialize)]
        struct InsightsJson {
            summary: SummaryJson,
            token_distribution: Vec<DistributionJson>,
            top_expensive: Vec<AnalysisJson>,
            exceeded_limits: Vec<AnalysisJson>,
            duplicates: Vec<Vec<String>>,
        }

        #[derive(Serialize)]
        struct SummaryJson {
            total_prompts: usize,
            total_tokens: usize,
            total_cost: f64,
            monthly_cost: f64,
        }

        #[derive(Serialize)]
        struct DistributionJson {
            bucket: String,
            count: usize,
        }

        #[derive(Serialize)]
        struct AnalysisJson {
            file_path: String,
            prompt_id: String,
            token_count: usize,
            total_cost: f64,
        }

        let buckets = [100, 500, 1000, 5000, 10000, 50000];
        let mut distribution = Vec::new();
        for (bucket_max, count) in &insights.token_distribution {
            let bucket_label = if *bucket_max == usize::MAX {
                "50K+".to_string()
            } else if *bucket_max == 100 {
                "0-100".to_string()
            } else {
                let prev = buckets
                    .iter()
                    .position(|&b| b == *bucket_max)
                    .map_or(0, |i| if i > 0 { buckets[i - 1] + 1 } else { 0 });
                format!("{}-{}", prev, bucket_max)
            };
            distribution.push(DistributionJson {
                bucket: bucket_label,
                count: *count,
            });
        }

        let json = InsightsJson {
            summary: SummaryJson {
                total_prompts: insights.total_prompts,
                total_tokens: insights.total_tokens,
                total_cost: insights.total_cost,
                monthly_cost: insights.monthly_cost,
            },
            token_distribution: distribution,
            top_expensive: insights
                .top_expensive
                .iter()
                .map(|a| AnalysisJson {
                    file_path: a.file_path.to_string_lossy().to_string(),
                    prompt_id: a.prompt_id.clone(),
                    token_count: a.token_count,
                    total_cost: a.total_cost,
                })
                .collect(),
            exceeded_limits: insights
                .exceeded_limits
                .iter()
                .map(|a| AnalysisJson {
                    file_path: a.file_path.to_string_lossy().to_string(),
                    prompt_id: a.prompt_id.clone(),
                    token_count: a.token_count,
                    total_cost: a.total_cost,
                })
                .collect(),
            duplicates: insights
                .duplicates
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect()
                })
                .collect(),
        };

        serde_json::to_string_pretty(&json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzers::types::{LibraryInsights, PromptAnalysis};
    use crate::parsers::Message;
    use std::path::PathBuf;

    #[test]
    fn test_format_text_empty() {
        let insights = LibraryInsights::new();
        let output = InsightsFormatter::format_text(&insights, "gpt-4", None);
        assert!(output.contains("Total Prompts: 0"));
    }

    #[test]
    fn test_format_json_empty() {
        let insights = LibraryInsights::new();
        let output = InsightsFormatter::format_json(&insights).unwrap();
        assert!(output.contains("\"total_prompts\": 0"));
    }

    fn create_test_analysis(path: &str, tokens: usize, cost: f64) -> PromptAnalysis {
        PromptAnalysis {
            file_path: PathBuf::from(path),
            prompt_id: path.to_string(),
            raw_content: "test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "test".to_string(),
            }],
            token_count: tokens,
            input_cost: cost,
            output_cost: 0.0,
            total_cost: cost,
            exceeds_limit: false,
            normalized_content: "test".to_string(),
        }
    }

    #[test]
    fn test_format_text_with_data() {
        let mut insights = LibraryInsights::new();
        insights.total_prompts = 2;
        insights.total_tokens = 1000;
        insights.total_cost = 0.05;
        insights.monthly_cost = 50.0;
        insights.top_expensive = vec![create_test_analysis("test1.txt", 500, 0.03)];
        insights.exceeded_limits = vec![create_test_analysis("test2.txt", 10000, 0.02)];

        let output = InsightsFormatter::format_text(&insights, "gpt-4", Some(8192));
        assert!(output.contains("Total Prompts: 2"));
        assert!(output.contains("test1.txt"));
        assert!(output.contains("test2.txt"));
    }
}
