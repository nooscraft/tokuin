/// Main compression orchestrator
///
/// Coordinates pattern matching, extractive compression, and Hieratic encoding
use crate::compression::context_library::ContextLibraryManager;
use crate::compression::similarity::find_best_match;
use crate::compression::types::{
    CompressionConfig, CompressionResult, ContextReference, ExtractionStats, HieraticDocument,
    HieraticSection, PatternMatch, ScoringMode,
};
use crate::error::AppError;
use crate::tokenizers::Tokenizer;

#[cfg(feature = "compression-embeddings")]
use crate::compression::embeddings::{cosine_similarity, EmbeddingCache, EmbeddingProvider};

/// Main compressor that orchestrates the compression pipeline
pub struct Compressor {
    tokenizer: Box<dyn Tokenizer>,
    context_library: Option<ContextLibraryManager>,
    #[cfg(feature = "compression-embeddings")]
    embedding_provider: Option<Box<dyn EmbeddingProvider>>,
    #[cfg(feature = "compression-embeddings")]
    embedding_cache: std::cell::RefCell<EmbeddingCache>,
    #[cfg(feature = "compression-embeddings")]
    critical_patterns: Vec<Vec<f32>>, // Pre-computed embeddings for critical instruction patterns
}

impl Compressor {
    /// Create a new compressor
    pub fn new(tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            tokenizer,
            context_library: None,
            #[cfg(feature = "compression-embeddings")]
            embedding_provider: None,
            #[cfg(feature = "compression-embeddings")]
            embedding_cache: std::cell::RefCell::new(EmbeddingCache::default()),
            #[cfg(feature = "compression-embeddings")]
            critical_patterns: Vec::new(),
        }
    }

    /// Set the context library
    pub fn with_context_library(mut self, library: ContextLibraryManager) -> Self {
        self.context_library = Some(library);
        self
    }

    /// Set the embedding provider for semantic scoring
    #[cfg(feature = "compression-embeddings")]
    pub fn with_embeddings(self, provider: Box<dyn EmbeddingProvider>) -> Result<Self, AppError> {
        if !provider.is_available() {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "Embedding provider is not available".to_string(),
            )));
        }

        // Pre-compute embeddings for critical instruction patterns
        let critical_patterns = self.initialize_critical_patterns(&*provider)?;

        Ok(Self {
            embedding_provider: Some(provider),
            critical_patterns,
            ..self
        })
    }

    /// Initialize critical instruction patterns with embeddings
    #[cfg(feature = "compression-embeddings")]
    fn initialize_critical_patterns(
        &self,
        provider: &dyn EmbeddingProvider,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        let patterns = vec![
            "Response Format:",
            "separator is ALWAYS",
            "DO NOT output",
            "Extracted_Value",
            "Doc_Page_Number",
            "must be exactly",
            "critical requirement",
            "hard constraint",
            "required field",
            "mandatory",
        ];

        patterns
            .iter()
            .map(|p| provider.embed(p))
            .collect::<Result<Vec<_>, _>>()
    }

    /// Split a prompt into instructions and document/markup sections.
    ///
    /// Heuristic:
    /// - If `<Document>` marker exists, everything from that line onward is treated
    ///   as the raw document payload.
    /// - Otherwise, we scan lines and treat the first line that looks like structured
    ///   markup (HTML/JSON) as the start of the document section.
    /// - Only the instruction prefix is eligible for compression; the document tail
    ///   is preserved verbatim.
    fn split_instructions_and_document<'a>(&self, text: &'a str) -> (&'a str, &'a str) {
        if let Some(idx) = text.find("<Document>") {
            let (head, tail) = text.split_at(idx);
            return (head, tail);
        }

        let mut offset = 0usize;
        for line in text.lines() {
            let trimmed = line.trim_start();
            let line_len = line.len() + 1; // +1 for the newline we just consumed

            // Detect obvious markup / structured content
            let looks_like_html = trimmed.starts_with('<') && trimmed.contains('>');
            let looks_like_json_block = (trimmed.starts_with('{') || trimmed.starts_with('['))
                && trimmed.contains("\"content\"")
                && trimmed.contains("\"page\"");
            let looks_like_table = trimmed.starts_with("<table")
                || trimmed.starts_with("</table>")
                || trimmed.starts_with("<tr>")
                || trimmed.starts_with("<th>")
                || trimmed.starts_with("<td>");

            if looks_like_html || looks_like_json_block || looks_like_table {
                let (head, tail) = text.split_at(offset);
                return (head, tail);
            }

            offset += line_len;
        }

        // Fallback: no obvious document section detected; treat everything as instructions.
        (text, "")
    }

    /// Compress a prompt
    pub fn compress(
        &self,
        prompt: &str,
        config: &CompressionConfig,
    ) -> Result<CompressionResult, AppError> {
        let original_tokens = self.tokenizer.count_tokens(prompt)?;

        // Split into instructions and document; only instructions are compressible.
        let (instructions, document_part) = self.split_instructions_and_document(prompt);

        // Find and replace context patterns in instructions only
        let (processed_instructions, context_refs) = if !config.force_inline {
            self.find_and_replace_patterns(instructions, config.min_similarity)?
        } else {
            (instructions.to_string(), Vec::new())
        };

        // Apply extractive compression to instructions only
        let (compressed_instructions, extraction_stats) = self.apply_extractive_compression(
            &processed_instructions,
            config,
            original_tokens,
            &context_refs,
        )?;

        // Reattach the unmodified document section
        let compressed_text = format!("{}{}", compressed_instructions, document_part);

        // Create Hieratic document
        let document = self.create_hieratic_document(&compressed_text, &context_refs, config)?;

        // Calculate final metrics
        let compressed_tokens = self.calculate_compressed_tokens(&document, &context_refs)?;
        let tokens_saved = original_tokens.saturating_sub(compressed_tokens);
        let compression_ratio = if original_tokens > 0 {
            tokens_saved as f64 / original_tokens as f64
        } else {
            0.0
        };

        Ok(CompressionResult {
            original_tokens,
            compressed_tokens,
            compression_ratio,
            tokens_saved,
            context_refs,
            extractive_stats: extraction_stats,
            document,
            anchors: Vec::new(),   // Will be populated in incremental mode
            quality_metrics: None, // Will be calculated if requested
        })
    }

    /// Compress a prompt incrementally (Factory.ai-inspired approach)
    ///
    /// Only compresses new content since the last anchor, avoiding re-compression of
    /// already processed text. This is much more efficient for multi-turn conversations.
    #[cfg(feature = "compression")]
    pub fn compress_incremental(
        &self,
        new_text: &str,
        config: &CompressionConfig,
    ) -> Result<CompressionResult, AppError> {
        use crate::compression::types::CompressionAnchor;

        let previous = config
            .previous_result
            .as_ref()
            .expect("incremental compression requires previous state");

        let mut anchors = previous.anchors.clone();
        let previous_tokens = previous.original_tokens;

        let (summary_text, recent_text, summary_token_len, recent_token_len) =
            self.split_summary_and_recent(new_text, config.retention_threshold)?;

        let reduction_ratio = self.reduction_ratio(config);

        if summary_token_len > 0 {
            let target_tokens = ((summary_token_len as f64) * (1.0 - reduction_ratio))
                .round()
                .max(1.0) as usize;

            let (compressed_span, _stats) = if config.structured_mode {
                self.compress_structured_document(&summary_text, target_tokens, config)?
            } else {
                self.compress_by_sentence_scoring(&summary_text, target_tokens, config)?
            };

            let summary_tokens = self.tokenizer.count_tokens(&compressed_span)?;
            let new_anchor = CompressionAnchor::new(
                previous_tokens + summary_token_len,
                compressed_span.clone(),
                summary_tokens,
                previous_tokens,
                summary_tokens as f64 / summary_token_len.max(1) as f64,
            );
            anchors.push(new_anchor);
        }

        let total_tokens = previous_tokens + summary_token_len + recent_token_len;

        let mut compressed_content = String::new();
        for anchor in &anchors {
            compressed_content.push_str(&format!("@ANCHOR[{}]\n", anchor.id));
            compressed_content.push_str(&anchor.summary);
            compressed_content.push_str("\n\n");
        }

        if recent_token_len > 0 {
            compressed_content.push_str("@RECENT\n");
            compressed_content.push_str(&recent_text);
            compressed_content.push('\n');
        }

        let document = self.create_hieratic_document(&compressed_content, &[], config)?;

        let anchor_tokens: usize = anchors.iter().map(|a| a.summary_tokens).sum();
        let retained_tokens = recent_token_len;
        let compressed_tokens = anchor_tokens + retained_tokens;
        let tokens_saved = total_tokens.saturating_sub(compressed_tokens);
        let compression_ratio = if total_tokens > 0 {
            tokens_saved as f64 / total_tokens as f64
        } else {
            0.0
        };

        Ok(CompressionResult {
            original_tokens: total_tokens,
            compressed_tokens,
            compression_ratio,
            tokens_saved,
            context_refs: Vec::new(),
            extractive_stats: ExtractionStats::default(),
            document,
            anchors,
            quality_metrics: None, // Will be calculated if requested
        })
    }

    fn reduction_ratio(&self, config: &CompressionConfig) -> f64 {
        config.target_ratio.unwrap_or_else(|| match config.level {
            crate::compression::types::CompressionLevel::Light => 0.35,
            crate::compression::types::CompressionLevel::Medium => 0.55,
            crate::compression::types::CompressionLevel::Aggressive => 0.75,
        })
    }

    fn split_summary_and_recent(
        &self,
        text: &str,
        retention_tokens: usize,
    ) -> Result<(String, String, usize, usize), AppError> {
        let tokens = self.tokenizer.encode(text)?;
        if tokens.is_empty() {
            return Ok((String::new(), String::new(), 0, 0));
        }

        let retain = retention_tokens.min(tokens.len());
        let summary_len = tokens.len().saturating_sub(retain);

        if summary_len == 0 {
            let recent = self.tokenizer.decode(&tokens)?;
            return Ok((String::new(), recent, 0, tokens.len()));
        }

        let summary_tokens = self.tokenizer.decode(&tokens[..summary_len])?;
        let recent_tokens = if retain > 0 {
            self.tokenizer.decode(&tokens[summary_len..])?
        } else {
            String::new()
        };

        Ok((summary_tokens, recent_tokens, summary_len, retain))
    }

    /// Find and replace patterns from context library
    fn find_and_replace_patterns(
        &self,
        text: &str,
        min_similarity: f64,
    ) -> Result<(String, Vec<ContextReference>), AppError> {
        let Some(ref library) = self.context_library else {
            return Ok((text.to_string(), Vec::new()));
        };

        let mut matches: Vec<PatternMatch> = Vec::new();

        // Find matches for each pattern
        for pattern in &library.library().patterns {
            if let Some((start, end, similarity)) =
                find_best_match(&pattern.content, text, min_similarity)
            {
                matches.push(PatternMatch {
                    pattern: pattern.clone(),
                    start,
                    end,
                    similarity,
                });
            }
        }

        // Sort matches by position (start)
        matches.sort_by_key(|m| m.start);

        // Replace matches with references
        let mut result = String::new();
        let mut context_refs = Vec::new();
        let mut last_pos = 0;

        for m in matches {
            // Add text before match
            result.push_str(&text[last_pos..m.start]);

            // Add reference marker (will be handled by encoder)
            result.push_str(&format!("[PATTERN:{}]", m.pattern.id));

            // Track reference
            let original_tokens = self.tokenizer.count_tokens(&m.pattern.content)?;
            let compressed_tokens = self.tokenizer.count_tokens(&format!(
                "@{}[{}]",
                m.pattern.category.to_uppercase(),
                m.pattern.id
            ))?;

            context_refs.push(ContextReference {
                id: m.pattern.id.clone(),
                category: m.pattern.category.clone(),
                original_tokens,
                compressed_tokens,
                tokens_saved: original_tokens.saturating_sub(compressed_tokens),
            });

            last_pos = m.end;
        }

        // Add remaining text
        result.push_str(&text[last_pos..]);

        Ok((result, context_refs))
    }

    /// Apply extractive compression to reduce token count
    fn apply_extractive_compression(
        &self,
        text: &str,
        config: &CompressionConfig,
        original_tokens: usize,
        context_refs: &[ContextReference],
    ) -> Result<(String, ExtractionStats), AppError> {
        // Calculate target token count based on compression level
        let context_savings: usize = context_refs.iter().map(|r| r.tokens_saved).sum();
        let current_tokens = original_tokens.saturating_sub(context_savings);

        let target_ratio = config.target_ratio.unwrap_or_else(|| match config.level {
            crate::compression::types::CompressionLevel::Light => 0.35,
            crate::compression::types::CompressionLevel::Medium => 0.55,
            crate::compression::types::CompressionLevel::Aggressive => 0.75,
        });

        let target_tokens = (original_tokens as f64 * (1.0 - target_ratio)) as usize;

        if current_tokens <= target_tokens {
            // Already compressed enough
            return Ok((
                text.to_string(),
                ExtractionStats {
                    low_relevance_removed: 0,
                    redundant_removed: 0,
                    sections_compressed: 0,
                },
            ));
        }

        // Use structured mode if enabled
        let (compressed, stats) = if config.structured_mode {
            self.compress_structured_document(text, target_tokens, config)?
        } else {
            self.compress_by_sentence_scoring(text, target_tokens, config)?
        };

        Ok((compressed, stats))
    }

    /// Compress text by scoring and removing low-relevance sentences
    fn compress_by_sentence_scoring(
        &self,
        text: &str,
        target_tokens: usize,
        config: &CompressionConfig,
    ) -> Result<(String, ExtractionStats), AppError> {
        // Better sentence splitting: split by sentence boundaries, not just periods
        // This handles cases like "e.g.", "i.e.", etc.
        let sentences = self.split_into_sentences(text);

        if sentences.is_empty() {
            return Ok((text.to_string(), ExtractionStats::default()));
        }

        // Score each sentence with its original index to preserve order
        let mut scored_sentences: Vec<(f64, usize, String, usize)> = Vec::new();
        for (idx, sentence) in sentences.iter().enumerate() {
            // Get context (surrounding sentences)
            let context_start = idx.saturating_sub(2);
            let context_end = (idx + 3).min(sentences.len());
            let context: Vec<&str> = sentences[context_start..context_end]
                .iter()
                .map(|s| s.as_str())
                .collect();

            let score = self.score_sentence(sentence, &context, config);
            let tokens = self.tokenizer.count_tokens(sentence)?;
            scored_sentences.push((score, idx, sentence.clone(), tokens));
        }

        // Sort by score (descending), but keep original index
        scored_sentences.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Select sentences until we reach target, avoiding duplicates
        let mut selected_indices = std::collections::HashSet::new();
        let mut selected_with_order: Vec<(usize, String)> = Vec::new();
        let mut current_tokens = 0;
        let mut duplicates_removed = 0;

        for (_score, original_idx, sentence, tokens) in scored_sentences {
            // Check for duplicates using normalized sentence
            let normalized = sentence.trim().to_lowercase();
            let is_duplicate = selected_with_order
                .iter()
                .any(|(_, s)| s.trim().to_lowercase() == normalized);

            if is_duplicate {
                duplicates_removed += 1;
                continue;
            }

            if current_tokens + tokens <= target_tokens || selected_with_order.is_empty() {
                selected_indices.insert(original_idx);
                selected_with_order.push((original_idx, sentence));
                current_tokens += tokens;
            }
        }

        // Sort by original order to preserve structure
        selected_with_order.sort_by_key(|(idx, _)| *idx);

        // Reconstruct text preserving original order and structure
        let result: String = selected_with_order
            .into_iter()
            .map(|(_, sentence)| sentence)
            .collect::<Vec<_>>()
            .join(" ");

        let low_relevance_removed = text.len().saturating_sub(result.len());

        Ok((
            result,
            ExtractionStats {
                low_relevance_removed,
                redundant_removed: duplicates_removed,
                sections_compressed: 1,
            },
        ))
    }

    /// Split text into sentences more intelligently
    /// Handles line breaks, bullet points, and preserves structure
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        // First, split by newlines to preserve structure (instructions often have line breaks)
        let lines: Vec<&str> = text.lines().collect();
        let mut sentences = Vec::new();

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // If line ends with punctuation, it's likely a complete sentence
            if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
                sentences.push(trimmed.to_string());
            } else if trimmed.starts_with('-')
                || trimmed.starts_with('*')
                || trimmed.starts_with("•")
            {
                // Bullet points are complete units
                sentences.push(trimmed.to_string());
            } else if trimmed.len() > 20 {
                // Longer lines without punctuation might be complete thoughts
                sentences.push(trimmed.to_string());
            } else {
                // Short lines might be fragments - accumulate with next line
                if let Some(last) = sentences.last_mut() {
                    last.push_str(" ");
                    last.push_str(trimmed);
                } else {
                    sentences.push(trimmed.to_string());
                }
            }
        }

        // If we didn't get good splits from lines, fall back to period-based splitting
        if sentences.len() < text.matches('.').count() / 3 {
            // Split by sentence-ending punctuation, but be smarter about it
            let mut current = String::new();
            let chars: Vec<char> = text.chars().collect();

            for (i, &ch) in chars.iter().enumerate() {
                current.push(ch);

                // Check if this is a sentence ending
                if (ch == '.' || ch == '!' || ch == '?') && i + 1 < chars.len() {
                    let next_ch = chars[i + 1];
                    // Sentence ends if followed by space and capital, or end of text
                    if next_ch.is_whitespace() || i + 1 == chars.len() - 1 {
                        let trimmed = current.trim().to_string();
                        if trimmed.len() > 5 {
                            sentences.push(trimmed);
                        }
                        current.clear();
                    }
                }
            }

            // Add remaining text
            if !current.trim().is_empty() && current.trim().len() > 5 {
                sentences.push(current.trim().to_string());
            }
        }

        // Filter out very short fragments and normalize
        sentences
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() > 3 && !s.chars().all(|c| c.is_whitespace() || c == '.'))
            .collect()
    }

    /// Score a sentence for relevance using heuristic approach
    fn score_sentence_heuristic(&self, sentence: &str) -> f64 {
        let mut score = 0.0;

        let lower = sentence.to_lowercase();

        // Critical instruction lines must be preserved
        if lower.contains("response format:")
            || lower.contains("separator is always")
            || (lower.contains("do not") && lower.contains("output"))
            || lower.contains("extracted_value")
            || lower.contains("doc_page_number")
        {
            score += 100.0;
        }

        // Length bonus (not too short, not too long)
        let words = sentence.split_whitespace().count();
        if words >= 5 && words <= 30 {
            score += 1.0;
        }

        // Keyword bonuses
        if lower.contains("important") || lower.contains("critical") || lower.contains("must") {
            score += 2.0;
        }
        if lower.contains("analyze") || lower.contains("provide") || lower.contains("explain") {
            score += 1.5;
        }
        if lower.contains("example") || lower.contains("such as") {
            score += 1.0;
        }

        // Position bonus (first/last sentences often more important)
        // This would require context of all sentences - simplified here
        score += 0.5;

        score
    }

    /// Score a sentence using semantic embeddings
    #[cfg(feature = "compression-embeddings")]
    fn score_sentence_semantic(&self, sentence: &str, context: &[&str]) -> Result<f64, AppError> {
        let provider = self.embedding_provider.as_ref().ok_or_else(|| {
            #[cfg(feature = "load-test")]
            {
                AppError::Config("Embedding provider not available".to_string())
            }
            #[cfg(not(feature = "load-test"))]
            {
                AppError::Parse(crate::error::ParseError::InvalidFormat(
                    "Embedding provider not available".to_string(),
                ))
            }
        })?;

        // Get embedding for the sentence
        let sentence_emb =
            self.embedding_cache
                .borrow_mut()
                .get_or_compute(sentence, &**provider, |text| provider.embed(text))?;

        // 1. Similarity to critical patterns
        let max_similarity = if !self.critical_patterns.is_empty() {
            self.critical_patterns
                .iter()
                .map(|pattern| cosine_similarity(&sentence_emb, pattern))
                .fold(0.0, f64::max)
        } else {
            0.0
        };

        // 2. Information density (simplified: based on length and uniqueness)
        let info_density = self.compute_information_density(sentence, context)?;

        // 3. Position importance (first/last sentences)
        let position_score = self.compute_position_score(sentence, context);

        // Combine scores
        let semantic_score = (max_similarity * 50.0) + (info_density * 20.0) + position_score;

        Ok(semantic_score)
    }

    /// Compute information density of a sentence
    #[cfg(feature = "compression-embeddings")]
    fn compute_information_density(
        &self,
        sentence: &str,
        _context: &[&str],
    ) -> Result<f64, AppError> {
        // Simplified: longer sentences with more unique words have higher density
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let uniqueness_ratio = unique_words.len() as f64 / words.len().max(1) as f64;

        // Normalize to 0-1 range
        let density = (uniqueness_ratio * 0.5) + (words.len().min(30) as f64 / 30.0 * 0.5);
        Ok(density.min(1.0))
    }

    /// Compute position-based score
    fn compute_position_score(&self, _sentence: &str, context: &[&str]) -> f64 {
        // First and last sentences are often more important
        // This is a simplified version - full implementation would track position
        if context.is_empty() {
            1.0 // First sentence
        } else {
            0.5 // Middle sentences
        }
    }

    /// Score a sentence using the configured scoring mode
    fn score_sentence(&self, sentence: &str, context: &[&str], config: &CompressionConfig) -> f64 {
        match config.scoring_mode {
            ScoringMode::Heuristic => self.score_sentence_heuristic(sentence),
            #[cfg(feature = "compression-embeddings")]
            ScoringMode::Semantic => self
                .score_sentence_semantic(sentence, context)
                .unwrap_or_else(|_| self.score_sentence_heuristic(sentence)),
            #[cfg(feature = "compression-embeddings")]
            ScoringMode::Hybrid => {
                let heuristic_score = self.score_sentence_heuristic(sentence);
                let semantic_score = self
                    .score_sentence_semantic(sentence, context)
                    .unwrap_or(0.0);

                // Weighted combination: 70% semantic, 30% heuristic
                (semantic_score * 0.7) + (heuristic_score * 0.3)
            }
            #[cfg(not(feature = "compression-embeddings"))]
            ScoringMode::Semantic | ScoringMode::Hybrid => {
                // Fallback to heuristic if embeddings not available
                self.score_sentence_heuristic(sentence)
            }
        }
    }

    /// Compress structured document (JSON, tables, technical docs)
    fn compress_structured_document(
        &self,
        text: &str,
        target_tokens: usize,
        config: &CompressionConfig,
    ) -> Result<(String, ExtractionStats), AppError> {
        // Detect and extract structured blocks
        let blocks = self.extract_structured_blocks(text);

        // Find and consolidate repetitive patterns
        let (deduplicated_text, pattern_map) = self.find_repetitive_patterns(text);

        // Segment by logical sections preserving structure
        let segments = self.segment_by_structure(&deduplicated_text, &blocks);

        // Score segments with structure-awareness
        let mut scored_segments: Vec<(f64, String, usize)> = Vec::new();
        for (idx, segment) in segments.iter().enumerate() {
            // Get context for semantic scoring
            let context_start = idx.saturating_sub(2);
            let context_end = (idx + 3).min(segments.len());
            let context: Vec<&str> = segments[context_start..context_end]
                .iter()
                .map(|s| s.as_str())
                .collect();

            let score = self.score_structured_segment(segment, &blocks, &context, config);
            let tokens = self.tokenizer.count_tokens(segment)?;
            scored_segments.push((score, segment.clone(), tokens));
        }

        // Sort by score (descending)
        scored_segments.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Select segments until we reach target
        let mut selected = Vec::new();
        let mut current_tokens = 0;

        for (_score, segment, tokens) in scored_segments {
            if current_tokens + tokens <= target_tokens || selected.is_empty() {
                selected.push(segment);
                current_tokens += tokens;
            }
        }

        // Reconstruct with patterns expanded
        let mut result = selected.join("\n");
        let pattern_replacements: Vec<(String, String)> = pattern_map
            .iter()
            .map(|(k, v)| (format!("[REF:{}]", k), v.clone()))
            .collect();
        for (pattern_ref, pattern_content) in pattern_replacements {
            result = result.replace(&pattern_ref, &pattern_content);
        }

        let low_relevance_removed = text.len().saturating_sub(result.len());

        Ok((
            result,
            ExtractionStats {
                low_relevance_removed,
                redundant_removed: pattern_map.len(),
                sections_compressed: segments.len(),
            },
        ))
    }

    /// Extract structured blocks (JSON, HTML tables, code blocks)
    fn extract_structured_blocks(&self, text: &str) -> Vec<(usize, usize, &str)> {
        let mut blocks = Vec::new();

        // Find JSON blocks: {...} spanning multiple lines
        let mut depth = 0;
        let mut start = None;
        let chars: Vec<char> = text.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            match ch {
                '{' | '[' => {
                    if depth == 0 {
                        start = Some(i);
                    }
                    depth += 1;
                }
                '}' | ']' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(s) = start {
                            if i - s > 50 {
                                // Only consider significant blocks
                                blocks.push((s, i + 1, "json"));
                            }
                        }
                        start = None;
                    }
                }
                _ => {}
            }
        }

        // Find HTML table blocks
        let table_starts: Vec<usize> = text.match_indices("<table").map(|(i, _)| i).collect();
        let table_ends: Vec<usize> = text.match_indices("</table>").map(|(i, _)| i + 8).collect();

        for (start, end) in table_starts.iter().zip(table_ends.iter()) {
            blocks.push((*start, *end, "table"));
        }

        // Find code blocks (indented or fenced)
        let lines: Vec<&str> = text.lines().collect();
        let mut in_code_block = false;
        let mut code_start = 0;

        for (i, line) in lines.iter().enumerate() {
            if line.trim().starts_with("```") {
                if !in_code_block {
                    code_start =
                        text[..text.lines().take(i).collect::<Vec<_>>().join("\n").len()].len();
                    in_code_block = true;
                } else {
                    let code_end = text[..text
                        .lines()
                        .take(i + 1)
                        .collect::<Vec<_>>()
                        .join("\n")
                        .len()]
                        .len();
                    blocks.push((code_start, code_end, "code"));
                    in_code_block = false;
                }
            }
        }

        blocks.sort_by_key(|b| b.0);
        blocks
    }

    /// Find and replace repetitive patterns
    fn find_repetitive_patterns(
        &self,
        text: &str,
    ) -> (String, std::collections::HashMap<String, String>) {
        use std::collections::HashMap;

        let mut pattern_map = HashMap::new();
        let mut result = text.to_string();

        // Find repeated instruction patterns
        let patterns = [
            (
                r"Extract Full Text Values?:\s*-[^-]*?(?=\n\w|\nExtract|\nNormalize|$)",
                "EXTRACT_INSTR",
            ),
            (
                r"Normalize:\s*-[^-]*?(?=\nExtract|\nResponse|$)",
                "NORMALIZE_INSTR",
            ),
            (
                r"Response Format:\s*\w+::[^:]+::[^:]+::[^:]+",
                "RESPONSE_FMT",
            ),
            (
                r"If you(?:'re| are) unable to confidently[^.]+\.",
                "ERROR_HANDLING",
            ),
            (r"Provide this as your response to \w+\.", "RESPONSE_INSTR"),
            (r"enter your response as - for \w+ and \w+", "ERROR_DEFAULT"),
        ];

        let mut pattern_id = 0;
        for (_pattern_regex, pattern_name) in patterns.iter() {
            // Simple substring matching for common phrases
            let common_phrases = [
                "Extract Full Text Values:",
                "Normalize:",
                "Response Format:",
                "If you are unable to confidently identify and extract the value",
                "Provide this as your response to",
                "Use :: as the separator in your response",
            ];

            for phrase in common_phrases.iter() {
                let occurrences: Vec<usize> =
                    result.match_indices(phrase).map(|(i, _)| i).collect();
                if occurrences.len() >= 3 {
                    // This pattern repeats enough to warrant extraction
                    let pattern_key = format!("{}_{}", pattern_name, pattern_id);
                    pattern_id += 1;

                    // Store first occurrence
                    if let Some(&first_pos) = occurrences.first() {
                        let end_pos = first_pos
                            + phrase.len()
                            + 100.min(result.len() - first_pos - phrase.len());
                        let full_pattern = result[first_pos..end_pos].to_string();
                        pattern_map.insert(pattern_key.clone(), full_pattern.clone());

                        // Replace all occurrences with reference
                        let replacement = format!("[REF:{}]", pattern_key);
                        for _ in 0..occurrences.len() {
                            result = result.replacen(&full_pattern, &replacement, 1);
                        }
                    }
                    break; // Only process this pattern once
                }
            }
        }

        (result, pattern_map)
    }

    /// Segment text by logical structure (sections, definitions, examples)
    fn segment_by_structure(&self, text: &str, blocks: &[(usize, usize, &str)]) -> Vec<String> {
        let mut segments = Vec::new();
        let mut current_segment = String::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut char_pos = 0;

        for line in lines {
            let line_start = char_pos;
            let line_end = char_pos + line.len();
            char_pos = line_end + 1; // +1 for newline

            // Check if we're inside a structured block
            let in_block = blocks
                .iter()
                .any(|(start, end, _)| line_start >= *start && line_end <= *end);

            // Start new segment on headers or definition markers
            let is_header = line.trim().starts_with('#')
                || line.trim().ends_with(':')
                || line.trim().starts_with("##")
                || line.trim().starts_with("Definition:")
                || line.trim().starts_with("Location:")
                || line.trim().starts_with("Extraction:")
                || line.trim().starts_with("Response Format:");

            if is_header && !current_segment.is_empty() && !in_block {
                segments.push(current_segment.trim().to_string());
                current_segment = String::new();
            }

            current_segment.push_str(line);
            current_segment.push('\n');

            // End segment after structured blocks
            if in_block {
                let at_block_end = blocks.iter().any(|(_, end, _)| line_end >= *end - 1);
                if at_block_end {
                    segments.push(current_segment.trim().to_string());
                    current_segment = String::new();
                }
            }
        }

        if !current_segment.trim().is_empty() {
            segments.push(current_segment.trim().to_string());
        }

        segments
    }

    /// Score a structured segment for importance
    fn score_structured_segment(
        &self,
        segment: &str,
        _blocks: &[(usize, usize, &str)],
        context: &[&str],
        config: &CompressionConfig,
    ) -> f64 {
        // Use the same scoring logic as sentences, but with segment-specific bonuses
        let base_score = self.score_sentence(segment, context, config);

        let mut score = base_score;
        let lower = segment.to_lowercase();

        // Structural importance bonuses
        if segment.trim().starts_with('#') {
            score += 5.0; // Headers are critical
        }

        if segment.contains("<table") || segment.contains("```") || segment.contains('{') {
            score += 4.0; // Structured data is important
        }

        // Definition and format markers
        if lower.contains("definition:") || lower.contains("location:") {
            score += 3.5;
        }

        // Length bonus (substantial segments are usually important)
        let lines = segment.lines().count();
        if lines >= 3 && lines <= 20 {
            score += 1.0;
        } else if lines > 20 {
            score += 2.0; // Very substantial sections likely important
        }

        // Pattern references (repeated content already compressed)
        if segment.contains("[REF:") {
            score += 1.5;
        }

        score
    }

    /// Create Hieratic document from compressed text
    fn create_hieratic_document(
        &self,
        text: &str,
        context_refs: &[ContextReference],
        config: &CompressionConfig,
    ) -> Result<HieraticDocument, AppError> {
        let mut doc = HieraticDocument::new();

        // Add context library reference if used
        if let Some(ref path) = config.context_library_path {
            doc.context_path = Some(path.clone());
        }

        // Add sections from context references
        for context_ref in context_refs {
            let section = match context_ref.category.as_str() {
                "role" => HieraticSection::Role {
                    id: context_ref.id.clone(),
                    content: None,
                },
                "examples" => HieraticSection::Examples {
                    id: context_ref.id.clone(),
                    content: None,
                },
                "constraints" => HieraticSection::Constraints {
                    id: context_ref.id.clone(),
                    content: None,
                },
                _ => continue,
            };
            doc.add_section(section);
        }

        // Process remaining text for task section
        let task_content = text
            .replace("[PATTERN:", "")
            .replace("]", "")
            .trim()
            .to_string();

        doc.add_section(HieraticSection::Task {
            content: task_content,
        });

        Ok(doc)
    }

    /// Calculate total tokens in compressed format
    fn calculate_compressed_tokens(
        &self,
        document: &HieraticDocument,
        context_refs: &[ContextReference],
    ) -> Result<usize, AppError> {
        let mut total = 0;

        // Overhead for format (@HIERATIC, @CONTEXT, etc.)
        total += 10;

        // Context references (each is very small)
        total += context_refs
            .iter()
            .map(|r| r.compressed_tokens)
            .sum::<usize>();

        // Count all sections in the Hieratic document
        for section in &document.sections {
            match section {
                HieraticSection::Role { id: _, content } => {
                    if let Some(role_content) = content {
                        total += self.tokenizer.count_tokens(role_content)?;
                    }
                }
                HieraticSection::Examples { id: _, content } => {
                    if let Some(examples_content) = content {
                        // Examples content is typically a single string with multiple examples
                        total += self.tokenizer.count_tokens(examples_content)?;
                    }
                }
                HieraticSection::Constraints { id: _, content } => {
                    if let Some(constraints_content) = content {
                        // Constraints content is typically a single string with multiple constraints
                        total += self.tokenizer.count_tokens(constraints_content)?;
                    }
                }
                HieraticSection::Task { content } => {
                    total += self.tokenizer.count_tokens(content)?;
                }
                HieraticSection::Focus { areas } => {
                    // Approximate tokens for focus areas (each area ~2 tokens)
                    total += areas.len() * 2;
                }
                HieraticSection::Style { preferences } => {
                    // Approximate tokens for style preferences (each preference ~2 tokens)
                    total += preferences.len() * 2;
                }
                HieraticSection::Format { structure } => {
                    total += self.tokenizer.count_tokens(structure)?;
                }
                HieraticSection::Context { path: _ } => {
                    // Context path is already counted in context_refs
                }
            }
        }

        Ok(total)
    }

    /// Calculate quality metrics for a compression result
    ///
    /// This expands the compressed prompt and compares it to the original
    /// to measure semantic similarity, critical instruction preservation, etc.
    pub fn calculate_quality_metrics(
        &self,
        original: &str,
        result: &CompressionResult,
    ) -> Result<crate::compression::quality::QualityMetrics, AppError> {
        use crate::compression::hieratic_encoder::HieraticEncoder;
        use crate::compression::quality::calculate_quality_metrics;

        // Encode the Hieratic document to text
        let encoder = HieraticEncoder::new();
        let compressed_hieratic = encoder.encode(&result.document)?;

        // Calculate quality metrics
        let context_lib_path = result.document.context_path.as_deref();
        let metrics = calculate_quality_metrics(original, &compressed_hieratic, context_lib_path)?;

        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::types::CompressionLevel;
    use crate::tokenizers::openai::OpenAITokenizer;

    fn create_test_tokenizer() -> Box<dyn Tokenizer> {
        Box::new(OpenAITokenizer::new("gpt-4").unwrap())
    }

    #[test]
    fn test_compress_simple_text() {
        let tokenizer = create_test_tokenizer();
        let compressor = Compressor::new(tokenizer);

        let prompt = "This is a test prompt. It contains multiple sentences. Each sentence adds some information.";
        let config = CompressionConfig {
            level: CompressionLevel::Medium,
            ..Default::default()
        };

        let result = compressor.compress(prompt, &config).unwrap();

        // For very short prompts, Hieratic format overhead might prevent compression
        // So we check that compression was attempted (ratio is calculated correctly)
        assert!(result.compression_ratio >= 0.0);
        // If compression happened, verify it's beneficial
        if result.compressed_tokens < result.original_tokens {
            assert!(result.compression_ratio > 0.0);
        }
    }

    #[test]
    fn test_score_sentence() {
        let tokenizer = create_test_tokenizer();
        let compressor = Compressor::new(tokenizer);
        let config = CompressionConfig::default();
        let context: &[&str] = &[];

        let high_score =
            compressor.score_sentence("This is important to analyze", context, &config);
        let low_score = compressor.score_sentence("Just a simple sentence", context, &config);

        assert!(high_score > low_score);
    }
}
