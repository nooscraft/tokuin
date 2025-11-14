/// Directory scanner for prompt files.
use crate::analyzers::duplicates::normalize_content;
use crate::analyzers::types::{LibraryInsights, PromptAnalysis};
use crate::error::AppError;
use crate::models::ModelRegistry;
use crate::parsers::{JsonParser, Parser, TextParser};
use std::path::Path;
use walkdir::WalkDir;

/// File format detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileFormat {
    Text,
    Json,
    Markdown,
    Yaml,
    Unknown,
}

/// Scanner for analyzing prompt libraries.
pub struct PromptScanner {
    registry: ModelRegistry,
    model: String,
    context_limit: Option<usize>,
}

impl PromptScanner {
    /// Create a new prompt scanner.
    pub fn new(registry: ModelRegistry, model: String, context_limit: Option<usize>) -> Self {
        Self {
            registry,
            model,
            context_limit,
        }
    }

    /// Scan a directory and analyze all prompt files.
    pub fn scan_directory(&self, dir: &Path) -> Result<Vec<PromptAnalysis>, AppError> {
        let mut analyses = Vec::new();

        for entry in WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if matches!(
                    ext_lower.as_str(),
                    "txt" | "md" | "json" | "yaml" | "yml" | "prompt"
                ) {
                    match self.analyze_file(path) {
                        Ok(Some(analysis)) => analyses.push(analysis),
                        Ok(None) => {
                            // File format not supported, skip silently
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to analyze {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }

        Ok(analyses)
    }

    /// Analyze a single file and return its analysis.
    fn analyze_file(&self, path: &Path) -> Result<Option<PromptAnalysis>, AppError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to read {}: {}",
                path.display(),
                e
            )))
        })?;

        let format = Self::detect_format(path, &content);
        let messages = match self.parse_file(&content, format)? {
            Some(msgs) => msgs,
            None => return Ok(None),
        };

        // Get tokenizer
        let tokenizer = self.registry.get_tokenizer(&self.model)?;

        // Count tokens
        let mut total_tokens = 0;
        for message in &messages {
            total_tokens += tokenizer.count_tokens(&message.content)?;
        }

        // Calculate costs
        let pricing = self.registry.pricing_for(&self.model);
        let (input_cost, output_cost) = if let Some((input_rate, output_rate)) = pricing {
            (
                (total_tokens as f64 / 1000.0) * input_rate,
                // Estimate 100 output tokens
                (100.0 / 1000.0) * output_rate,
            )
        } else {
            (0.0, 0.0)
        };
        let total_cost = input_cost + output_cost;

        // Check context limits
        let exceeds_limit = if let Some(limit) = self.context_limit {
            total_tokens > limit
        } else {
            false
        };

        // Normalize for duplicate detection
        let normalized = normalize_content(&content);

        let prompt_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Some(PromptAnalysis {
            file_path: path.to_path_buf(),
            prompt_id,
            raw_content: content,
            messages,
            token_count: total_tokens,
            input_cost,
            output_cost,
            total_cost,
            exceeds_limit,
            normalized_content: normalized,
        }))
    }

    /// Detect file format from path and content.
    fn detect_format(path: &Path, content: &str) -> FileFormat {
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            match ext_lower.as_str() {
                "json" => {
                    if content.trim_start().starts_with('{')
                        || content.trim_start().starts_with('[')
                    {
                        return FileFormat::Json;
                    }
                }
                "md" | "markdown" => return FileFormat::Markdown,
                "yaml" | "yml" => return FileFormat::Yaml,
                _ => {}
            }
        }

        // Try to detect from content
        if content.trim_start().starts_with('{') || content.trim_start().starts_with('[') {
            return FileFormat::Json;
        }

        FileFormat::Text
    }

    /// Parse file content based on format.
    fn parse_file(
        &self,
        content: &str,
        format: FileFormat,
    ) -> Result<Option<Vec<crate::parsers::Message>>, AppError> {
        match format {
            FileFormat::Json => {
                let parser = JsonParser::new();
                match parser.parse(content) {
                    Ok(messages) => Ok(Some(messages)),
                    Err(_) => {
                        // Try as text if JSON parsing fails
                        let text_parser = TextParser::new();
                        Ok(text_parser.parse(content).ok())
                    }
                }
            }
            FileFormat::Text => {
                let parser = TextParser::new();
                Ok(parser.parse(content).ok())
            }
            FileFormat::Markdown => {
                // For now, treat markdown as text (extract text content)
                // TODO: Implement proper markdown prompt extraction
                let text_parser = TextParser::new();
                Ok(text_parser.parse(content).ok())
            }
            FileFormat::Yaml => {
                // For now, skip YAML (CrewAI parser not implemented yet)
                // TODO: Implement CrewAI YAML parser
                Ok(None)
            }
            #[allow(dead_code)]
            FileFormat::Unknown => Ok(None),
        }
    }

    /// Generate insights from a list of analyses.
    pub fn generate_insights(
        analyses: &[PromptAnalysis],
        top_n: usize,
        monthly_invocations: u64,
    ) -> LibraryInsights {
        let mut insights = LibraryInsights::new();

        if analyses.is_empty() {
            return insights;
        }

        insights.total_prompts = analyses.len();
        insights.total_tokens = analyses.iter().map(|a| a.token_count).sum();
        insights.total_cost = analyses.iter().map(|a| a.total_cost).sum();
        insights.monthly_cost = insights.total_cost * monthly_invocations as f64;

        // Token distribution histogram
        insights.token_distribution = Self::calculate_distribution(analyses);

        // Top N most expensive
        let mut sorted_by_cost: Vec<&PromptAnalysis> = analyses.iter().collect();
        sorted_by_cost.sort_by(|a, b| b.total_cost.partial_cmp(&a.total_cost).unwrap());
        insights.top_expensive = sorted_by_cost.into_iter().take(top_n).cloned().collect();

        // Exceeded limits
        insights.exceeded_limits = analyses
            .iter()
            .filter(|a| a.exceeds_limit)
            .cloned()
            .collect();

        // Duplicates
        insights.duplicates = crate::analyzers::duplicates::detect_duplicates(analyses);

        insights
    }

    /// Calculate token distribution histogram.
    pub fn calculate_distribution(analyses: &[PromptAnalysis]) -> Vec<(usize, usize)> {
        let buckets = [100, 500, 1000, 5000, 10000, 50000];
        let mut distribution = vec![0; buckets.len() + 1]; // +1 for overflow bucket

        for analysis in analyses {
            let tokens = analysis.token_count;
            let mut placed = false;
            for (i, &bucket_max) in buckets.iter().enumerate() {
                if tokens <= bucket_max {
                    distribution[i] += 1;
                    placed = true;
                    break;
                }
            }
            if !placed {
                distribution[buckets.len()] += 1; // Overflow bucket
            }
        }

        // Convert to (bucket_max, count) pairs
        let mut result = Vec::new();
        for (i, &bucket_max) in buckets.iter().enumerate() {
            if distribution[i] > 0 {
                result.push((bucket_max, distribution[i]));
            }
        }
        if distribution[buckets.len()] > 0 {
            result.push((usize::MAX, distribution[buckets.len()]));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzers::types::PromptAnalysis;
    use crate::parsers::Message;
    use std::path::PathBuf;

    #[test]
    fn test_calculate_distribution() {
        let analyses = vec![
            create_analysis(50),
            create_analysis(150),
            create_analysis(600),
            create_analysis(2000),
            create_analysis(60000),
        ];

        let dist = PromptScanner::calculate_distribution(&analyses);
        // Should have buckets: 100 (50), 500 (150), 1000 (600), 5000 (2000), overflow (60000)
        assert!(dist.len() >= 4, "Should have at least 4 buckets");
        assert!(
            dist.iter().any(|(max, _)| *max == 100),
            "Should have 100 bucket"
        );
        assert!(
            dist.iter().any(|(max, _)| *max == 500),
            "Should have 500 bucket"
        );
        assert_eq!(
            dist.iter().map(|(_, count)| count).sum::<usize>(),
            5,
            "Total count should be 5"
        );
    }

    fn create_analysis(tokens: usize) -> PromptAnalysis {
        PromptAnalysis {
            file_path: PathBuf::from("test.txt"),
            prompt_id: "test".to_string(),
            raw_content: "test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "test".to_string(),
            }],
            token_count: tokens,
            input_cost: 0.0,
            output_cost: 0.0,
            total_cost: 0.0,
            exceeds_limit: false,
            normalized_content: "test".to_string(),
        }
    }
}
