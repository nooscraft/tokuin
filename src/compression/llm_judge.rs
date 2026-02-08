/// LLM-as-a-judge evaluation for compressed prompts
///
/// This module provides output-level evaluation by comparing outputs from
/// original and compressed prompts using an LLM judge.
#[cfg(all(feature = "compression", feature = "load-test"))]
use crate::error::AppError;
#[cfg(all(feature = "compression", feature = "load-test"))]
use crate::http::client::LlmClient;
#[cfg(all(feature = "compression", feature = "load-test"))]
use serde::{Deserialize, Serialize};

/// LLM judge evaluation metrics
#[cfg(all(feature = "compression", feature = "load-test"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmJudgeMetrics {
    /// Output equivalence score (0-100) - Do outputs convey same meaning?
    pub output_equivalence: f64,
    /// Instruction compliance score (0-100) - Do both follow instructions correctly?
    pub instruction_compliance: f64,
    /// Information completeness score (0-100) - Is all critical info preserved?
    pub information_completeness: f64,
    /// Quality preservation score (0-100) - Is output quality equivalent?
    pub quality_preservation: f64,
    /// Overall fidelity score (0-100) - Would compressed prompt be acceptable replacement?
    pub overall_fidelity: f64,
    /// Justification for the scores
    pub justification: String,
    /// Key differences between outputs
    pub key_differences: Vec<String>,
    /// Model used to generate outputs
    pub evaluation_model: String,
    /// Model used to judge outputs
    pub judge_model: String,
    /// Total cost of evaluation (2 outputs + 1 judge)
    pub evaluation_cost: Option<f64>,
    /// Output from original prompt (for debugging)
    pub original_output: Option<String>,
    /// Output from compressed prompt (for debugging)
    pub compressed_output: Option<String>,
}

impl LlmJudgeMetrics {
    /// Get overall rating as string
    pub fn rating(&self) -> &'static str {
        match self.overall_fidelity {
            x if x >= 90.0 => "Excellent",
            x if x >= 75.0 => "Good",
            x if x >= 60.0 => "Fair",
            x if x >= 40.0 => "Poor",
            _ => "Very Poor",
        }
    }

    /// Check if quality is acceptable (>= 70)
    pub fn is_acceptable(&self) -> bool {
        self.overall_fidelity >= 70.0
    }
}

/// Normalize model name to OpenRouter format
///
/// Converts simple model names (e.g., "gpt-4") to OpenRouter format (e.g., "openai/gpt-4")
/// If already in provider/model format, returns as-is.
#[cfg(all(feature = "compression", feature = "load-test"))]
pub fn normalize_model_name(model: &str) -> String {
    // If already in provider/model format, return as-is
    if model.contains('/') {
        return model.to_string();
    }

    // Map common model names to OpenRouter format
    let normalized: String = match model.to_lowercase().as_str() {
        // OpenAI models
        "gpt-4" | "gpt4" => "openai/gpt-4".to_string(),
        "gpt-4-turbo" | "gpt4-turbo" => "openai/gpt-4-turbo".to_string(),
        "gpt-3.5-turbo" | "gpt-3.5" | "gpt35" => "openai/gpt-3.5-turbo".to_string(),
        "gpt-3.5-turbo-16k" => "openai/gpt-3.5-turbo-16k".to_string(),
        // Anthropic models
        "claude-3-opus" | "claude-opus" => "anthropic/claude-3-opus".to_string(),
        "claude-3-sonnet" | "claude-sonnet" => "anthropic/claude-3-sonnet".to_string(),
        "claude-3-haiku" | "claude-haiku" => "anthropic/claude-3-haiku".to_string(),
        "claude-2" => "anthropic/claude-2".to_string(),
        // Default: assume OpenAI
        _ => {
            // Try to detect provider from model name
            if model.to_lowercase().contains("claude") {
                format!("anthropic/{}", model)
            } else {
                format!("openai/{}", model)
            }
        }
    };

    normalized
}

/// Build evaluation prompt comparing outputs
#[cfg(all(feature = "compression", feature = "load-test"))]
fn build_evaluation_prompt(
    original_prompt: &str,
    output_original: &str,
    compressed_prompt: &str,
    output_compressed: &str,
) -> String {
    format!(
        r#"You are evaluating prompt compression quality by comparing outputs from original and compressed prompts.

Original Prompt:
{}

Output from Original Prompt:
{}

Compressed Prompt (expanded):
{}

Output from Compressed Prompt:
{}

Rate how well the compressed prompt preserves the original's behavior on a scale of 0-100 for:
1. Output Equivalence: Do both outputs convey the same meaning and information?
2. Instruction Compliance: Do both outputs follow the same instructions correctly?
3. Information Completeness: Does the compressed prompt's output contain all critical information from the original's output?
4. Quality Preservation: Is the quality of the compressed prompt's output equivalent to the original?
5. Overall Fidelity: Would the compressed prompt be acceptable as a replacement for the original?

CRITICAL: You MUST respond with ONLY valid JSON. Do not include markdown code blocks, explanations, or any other text. Return ONLY the JSON object below.

Required JSON format:
{{
  "output_equivalence": <0-100>,
  "instruction_compliance": <0-100>,
  "information_completeness": <0-100>,
  "quality_preservation": <0-100>,
  "overall_fidelity": <0-100>,
  "justification": "<brief explanation>",
  "key_differences": ["<difference 1>", "<difference 2>"]
}}

Example valid response:
{{"output_equivalence": 85, "instruction_compliance": 90, "information_completeness": 88, "quality_preservation": 87, "overall_fidelity": 88, "justification": "Outputs are equivalent", "key_differences": []}}

Remember: Return ONLY the JSON object, nothing else."#,
        original_prompt, output_original, compressed_prompt, output_compressed
    )
}

/// Parse judge response (JSON with fallback to text)
#[cfg(all(feature = "compression", feature = "load-test"))]
fn parse_judge_response(
    response: &str,
    evaluation_model: &str,
    judge_model: &str,
) -> Result<LlmJudgeMetrics, AppError> {
    // Step 1: Try to extract JSON from markdown code blocks (Claude often wraps JSON in ```json blocks)
    let json_str = if response.contains("```json") {
        // Extract content between ```json and ```
        if let Some(start) = response.find("```json") {
            let after_start = &response[start + 7..];
            if let Some(end) = after_start.find("```") {
                after_start[..end].trim()
            } else {
                response.trim()
            }
        } else {
            response.trim()
        }
    } else if response.contains("```") {
        // Try generic code block
        if let Some(start) = response.find("```") {
            let after_start = &response[start + 3..];
            if let Some(end) = after_start.find("```") {
                after_start[..end].trim()
            } else {
                response.trim()
            }
        } else {
            response.trim()
        }
    } else {
        response.trim()
    };

    // Step 2: Try JSON parsing
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
        if let (Some(oe), Some(ic), Some(info), Some(qp), Some(of)) = (
            parsed["output_equivalence"].as_f64(),
            parsed["instruction_compliance"].as_f64(),
            parsed["information_completeness"].as_f64(),
            parsed["quality_preservation"].as_f64(),
            parsed["overall_fidelity"].as_f64(),
        ) {
            let justification = parsed["justification"]
                .as_str()
                .unwrap_or("No justification provided")
                .to_string();

            let key_differences = parsed["key_differences"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            return Ok(LlmJudgeMetrics {
                output_equivalence: oe.clamp(0.0, 100.0),
                instruction_compliance: ic.clamp(0.0, 100.0),
                information_completeness: info.clamp(0.0, 100.0),
                quality_preservation: qp.clamp(0.0, 100.0),
                overall_fidelity: of.clamp(0.0, 100.0),
                justification,
                key_differences,
                evaluation_model: evaluation_model.to_string(),
                judge_model: judge_model.to_string(),
                evaluation_cost: None, // Will be calculated separately
                original_output: None,
                compressed_output: None,
            });
        }
    }

    // Step 3: Fallback - Try to extract scores from text using regex
    use regex::Regex;

    // More flexible regex patterns
    let patterns = [
        r"(?i)(output\s+equivalence|instruction\s+compliance|information\s+completeness|quality\s+preservation|overall\s+fidelity)[:\s]+(\d+(?:\.\d+)?)",
        r#"(?i)("output_equivalence"|"instruction_compliance"|"information_completeness"|"quality_preservation"|"overall_fidelity")[:\s]+(\d+(?:\.\d+)?)"#,
        r"(?i)(output\s*equivalence|instruction\s*compliance|information\s*completeness|quality\s*preservation|overall\s*fidelity)[:\s=]+(\d+(?:\.\d+)?)",
    ];

    let mut output_equivalence = 0.0;
    let mut instruction_compliance = 0.0;
    let mut information_completeness = 0.0;
    let mut quality_preservation = 0.0;
    let mut overall_fidelity = 0.0;
    let mut found_any = false;

    for pattern in &patterns {
        if let Ok(re) = Regex::new(pattern) {
            for cap in re.captures_iter(response) {
                if let (Some(metric), Some(score_str)) = (cap.get(1), cap.get(2)) {
                    if let Ok(score) = score_str.as_str().parse::<f64>() {
                        found_any = true;
                        let metric_lower = metric.as_str().to_lowercase();
                        if metric_lower.contains("output") && metric_lower.contains("equivalence") {
                            output_equivalence = score.clamp(0.0, 100.0);
                        } else if metric_lower.contains("instruction")
                            && metric_lower.contains("compliance")
                        {
                            instruction_compliance = score.clamp(0.0, 100.0);
                        } else if metric_lower.contains("information")
                            && metric_lower.contains("completeness")
                        {
                            information_completeness = score.clamp(0.0, 100.0);
                        } else if metric_lower.contains("quality")
                            && metric_lower.contains("preservation")
                        {
                            quality_preservation = score.clamp(0.0, 100.0);
                        } else if metric_lower.contains("overall")
                            && metric_lower.contains("fidelity")
                        {
                            overall_fidelity = score.clamp(0.0, 100.0);
                        }
                    }
                }
            }
            // If we found scores with this pattern, we can break
            if found_any {
                break;
            }
        }
    }

    if !found_any {
        // Provide more helpful error message with response snippet
        let response_preview = if response.len() > 500 {
            format!("{}...", &response[..500])
        } else {
            response.to_string()
        };

        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
            format!(
                "Could not parse judge response: no scores found. Response preview: {}",
                response_preview
            ),
        )));
    }

    // If we found at least one score, use average for missing ones
    let avg_score = (output_equivalence
        + instruction_compliance
        + information_completeness
        + quality_preservation
        + overall_fidelity)
        / 5.0;

    if output_equivalence == 0.0 {
        output_equivalence = avg_score;
    }
    if instruction_compliance == 0.0 {
        instruction_compliance = avg_score;
    }
    if information_completeness == 0.0 {
        information_completeness = avg_score;
    }
    if quality_preservation == 0.0 {
        quality_preservation = avg_score;
    }
    if overall_fidelity == 0.0 {
        overall_fidelity = avg_score;
    }

    Ok(LlmJudgeMetrics {
        output_equivalence,
        instruction_compliance,
        information_completeness,
        quality_preservation,
        overall_fidelity,
        justification: "Parsed from text response (JSON parsing failed)".to_string(),
        key_differences: Vec::new(),
        evaluation_model: evaluation_model.to_string(),
        judge_model: judge_model.to_string(),
        evaluation_cost: None,
        original_output: None,
        compressed_output: None,
    })
}

/// Estimate cost based on token usage and model
#[cfg(all(feature = "compression", feature = "load-test"))]
fn estimate_cost(input_tokens: usize, output_tokens: usize, model: &str) -> f64 {
    // Rough cost estimates per 1K tokens (input/output)
    // These are approximate and should be updated based on actual pricing
    let (input_cost_per_1k, output_cost_per_1k) = if model.contains("gpt-4") {
        (0.03, 0.06) // GPT-4 pricing
    } else if model.contains("gpt-3.5") {
        (0.0015, 0.002) // GPT-3.5-turbo pricing
    } else if model.contains("claude-3-opus") {
        (0.015, 0.075) // Claude 3 Opus
    } else if model.contains("claude-3-sonnet") {
        (0.003, 0.015) // Claude 3 Sonnet
    } else if model.contains("claude-3-haiku") {
        (0.00025, 0.00125) // Claude 3 Haiku
    } else {
        (0.001, 0.002) // Default fallback
    };

    let input_cost = (input_tokens as f64 / 1000.0) * input_cost_per_1k;
    let output_cost = (output_tokens as f64 / 1000.0) * output_cost_per_1k;

    input_cost + output_cost
}

/// Evaluate compression quality using LLM-as-a-judge
///
/// This function:
/// 1. Sends original prompt to evaluation model → gets output A
/// 2. Sends compressed prompt to evaluation model → gets output B
/// 3. Uses judge model to compare outputs A and B
///
/// # Arguments
///
/// * `original_prompt` - The original uncompressed prompt
/// * `compressed_prompt` - The expanded compressed prompt
/// * `evaluation_model` - Model to use for generating outputs (normalized to OpenRouter format)
/// * `judge_model` - Model to use for judging outputs (normalized to OpenRouter format)
/// * `client` - LLM client (should be OpenRouter by default)
///
/// # Returns
///
/// `LlmJudgeMetrics` with scores and metadata
#[cfg(all(feature = "compression", feature = "load-test"))]
pub async fn evaluate_with_llm_judge(
    original_prompt: &str,
    compressed_prompt: &str,
    evaluation_model: &str,
    judge_model: &str,
    client: &dyn LlmClient,
) -> Result<LlmJudgeMetrics, AppError> {
    // Normalize model names to OpenRouter format
    let eval_model = normalize_model_name(evaluation_model);
    let judge_model_norm = normalize_model_name(judge_model);

    // Step 1: Generate output from original prompt
    let original_output_res = client.send_request(original_prompt, &eval_model).await;
    let (original_output, original_tokens) = match original_output_res {
        Ok(resp) => (resp.content, resp.total_tokens.unwrap_or(0)),
        Err(e) => {
            return Err(AppError::Api(format!(
                "Failed to generate output from original prompt: {}",
                e
            )));
        }
    };

    // Step 2: Generate output from compressed prompt (parallelizable)
    let compressed_output_res = client.send_request(compressed_prompt, &eval_model).await;
    let (compressed_output, compressed_tokens) = match compressed_output_res {
        Ok(resp) => (resp.content, resp.total_tokens.unwrap_or(0)),
        Err(e) => {
            return Err(AppError::Api(format!(
                "Failed to generate output from compressed prompt: {}",
                e
            )));
        }
    };

    // Step 3: Build evaluation prompt comparing outputs
    let evaluation_prompt = build_evaluation_prompt(
        original_prompt,
        &original_output,
        compressed_prompt,
        &compressed_output,
    );

    // Step 4: Call judge LLM API
    let judge_response = client
        .send_request(&evaluation_prompt, &judge_model_norm)
        .await?;
    let judge_tokens = judge_response.total_tokens.unwrap_or(0);

    // Step 5: Parse response
    let mut metrics =
        parse_judge_response(&judge_response.content, &eval_model, &judge_model_norm)?;

    // Calculate costs
    let eval_cost_1 = estimate_cost(
        original_tokens.saturating_sub(judge_tokens),
        original_output.len() / 4, // Rough estimate
        &eval_model,
    );
    let eval_cost_2 = estimate_cost(
        compressed_tokens.saturating_sub(judge_tokens),
        compressed_output.len() / 4,
        &eval_model,
    );
    let judge_cost = estimate_cost(
        evaluation_prompt.len() / 4, // Rough estimate
        judge_response.content.len() / 4,
        &judge_model_norm,
    );

    metrics.evaluation_cost = Some(eval_cost_1 + eval_cost_2 + judge_cost);
    metrics.original_output = Some(original_output);
    metrics.compressed_output = Some(compressed_output);

    Ok(metrics)
}

#[cfg(test)]
#[cfg(all(feature = "compression", feature = "load-test"))]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_model_name() {
        assert_eq!(normalize_model_name("gpt-4"), "openai/gpt-4");
        assert_eq!(
            normalize_model_name("gpt-3.5-turbo"),
            "openai/gpt-3.5-turbo"
        );
        assert_eq!(
            normalize_model_name("claude-3-sonnet"),
            "anthropic/claude-3-sonnet"
        );
        assert_eq!(normalize_model_name("openai/gpt-4"), "openai/gpt-4"); // Already normalized
        assert_eq!(
            normalize_model_name("anthropic/claude-3-opus"),
            "anthropic/claude-3-opus"
        );
    }

    #[test]
    fn test_parse_judge_response_json() {
        let json_response = r#"{
            "output_equivalence": 92.5,
            "instruction_compliance": 95.0,
            "information_completeness": 88.0,
            "quality_preservation": 90.0,
            "overall_fidelity": 91.0,
            "justification": "Outputs are semantically equivalent",
            "key_differences": ["Minor formatting differences"]
        }"#;

        let metrics = parse_judge_response(json_response, "openai/gpt-4", "openai/gpt-4").unwrap();
        assert_eq!(metrics.output_equivalence, 92.5);
        assert_eq!(metrics.instruction_compliance, 95.0);
        assert_eq!(metrics.overall_fidelity, 91.0);
        assert_eq!(metrics.justification, "Outputs are semantically equivalent");
        assert_eq!(metrics.key_differences.len(), 1);
    }

    #[test]
    fn test_parse_judge_response_text() {
        let text_response =
            "Output Equivalence: 85\nInstruction Compliance: 90\nOverall Fidelity: 87.5";
        let metrics = parse_judge_response(text_response, "openai/gpt-4", "openai/gpt-4").unwrap();
        assert_eq!(metrics.output_equivalence, 85.0);
        assert_eq!(metrics.instruction_compliance, 90.0);
        assert!(metrics.overall_fidelity > 0.0);
    }

    #[test]
    fn test_llm_judge_metrics_rating() {
        let metrics = LlmJudgeMetrics {
            output_equivalence: 95.0,
            instruction_compliance: 95.0,
            information_completeness: 95.0,
            quality_preservation: 95.0,
            overall_fidelity: 95.0,
            justification: "Excellent".to_string(),
            key_differences: Vec::new(),
            evaluation_model: "openai/gpt-4".to_string(),
            judge_model: "openai/gpt-4".to_string(),
            evaluation_cost: Some(0.01),
            original_output: None,
            compressed_output: None,
        };

        assert_eq!(metrics.rating(), "Excellent");
        assert!(metrics.is_acceptable());
    }
}
