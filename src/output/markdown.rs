/// Markdown formatter for reports.
#[cfg(feature = "markdown")]
use crate::output::{Formatter, TokenResult};

/// Markdown formatter for generating markdown reports.
#[cfg(feature = "markdown")]
pub struct MarkdownFormatter {
    show_breakdown: bool,
}

#[cfg(feature = "markdown")]
impl MarkdownFormatter {
    /// Create a new markdown formatter.
    ///
    /// # Arguments
    ///
    /// * `show_breakdown` - Whether to show role-based breakdown.
    pub fn new(show_breakdown: bool) -> Self {
        Self { show_breakdown }
    }
}

#[cfg(feature = "markdown")]
impl Formatter for MarkdownFormatter {
    fn format_result(&self, result: &TokenResult) -> String {
        let mut output = Vec::new();

        output.push(format!("## Token Analysis: {}", result.model));
        output.push(String::new());
        output.push(format!("**Total Tokens:** {}", result.tokens));
        output.push(String::new());

        if let Some(breakdown) = &result.breakdown {
            if self.show_breakdown {
                output.push("### Breakdown by Role".to_string());
                output.push(String::new());
                output.push("| Role | Tokens |".to_string());
                output.push("|------|--------|".to_string());
                output.push(format!("| System | {} |", breakdown.system));
                output.push(format!("| User | {} |", breakdown.user));
                output.push(format!("| Assistant | {} |", breakdown.assistant));
                output.push(String::new());
            }
        }

        if result.input_cost.is_some() || result.output_cost.is_some() {
            output.push("### Cost Estimation".to_string());
            output.push(String::new());
            if let Some(cost) = result.input_cost {
                output.push(format!("- **Input Cost:** ${:.4}", cost));
            }
            if let Some(cost) = result.output_cost {
                output.push(format!("- **Output Cost:** ${:.4}", cost));
            }
            // Show total cost if both are available
            if let (Some(input_cost), Some(output_cost)) = (result.input_cost, result.output_cost) {
                let total_cost = input_cost + output_cost;
                output.push(format!("- **Total Cost:** ${:.4}", total_cost));
            }
        }

        output.join("\n")
    }

    fn format_comparison(&self, results: &[TokenResult]) -> String {
        let mut output = vec![
            "## Model Comparison".to_string(),
            String::new(),
            "| Model | Tokens | Input Cost | Output Cost |".to_string(),
            "|-------|--------|------------|-------------|".to_string(),
        ];

        for result in results {
            let input_cost = result
                .input_cost
                .map(|c| format!("${:.4}", c))
                .unwrap_or_else(|| "n/a".to_string());
            let output_cost = result
                .output_cost
                .map(|c| format!("${:.4}", c))
                .unwrap_or_else(|| "n/a".to_string());
            output.push(format!(
                "| {} | {} | {} | {} |",
                result.model, result.tokens, input_cost, output_cost
            ));
        }

        output.join("\n")
    }
}

#[cfg(test)]
#[cfg(feature = "markdown")]
mod tests {
    use super::*;
    use crate::output::TokenResult;

    #[test]
    fn test_format_result() {
        let formatter = MarkdownFormatter::new(false);
        let result = TokenResult {
            model: "gpt-4".to_string(),
            tokens: 100,
            input_cost: Some(0.003),
            output_cost: None,
            breakdown: None,
        };
        let output = formatter.format_result(&result);
        assert!(output.contains("gpt-4"));
        assert!(output.contains("100"));
    }
}
