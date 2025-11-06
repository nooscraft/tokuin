/// Markdown processing utilities.
#[cfg(feature = "markdown")]
use pulldown_cmark::{html, Options, Parser};

/// Strip markdown formatting from text to reduce token count.
///
/// This removes markdown syntax while preserving the text content,
/// which can significantly reduce token count for prompts with heavy
/// markdown formatting.
///
/// # Arguments
///
/// * `text` - The markdown text to strip.
///
/// # Returns
///
/// Plain text with markdown formatting removed.
///
/// # Example
///
/// ```rust
/// use tokuin::utils::markdown::strip_markdown;
///
/// let markdown = "# Title\n\nThis is **bold** text.";
/// let plain = strip_markdown(markdown);
/// assert_eq!(plain.trim(), "Title\n\nThis is bold text.");
/// ```
#[cfg(feature = "markdown")]
pub fn strip_markdown(text: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);

    let parser = Parser::new_ext(text, options);
    let mut html_output = String::new();
    html::push_html(&mut html_output, parser);

    // Convert HTML to plain text by removing tags
    // This is a simple approach - for production, consider using a proper HTML parser
    let mut plain = String::new();
    let mut in_tag = false;

    for ch in html_output.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => plain.push(ch),
            _ => {}
        }
    }

    // Decode HTML entities
    plain = plain.replace("&lt;", "<");
    plain = plain.replace("&gt;", ">");
    plain = plain.replace("&amp;", "&");
    plain = plain.replace("&quot;", "\"");
    plain = plain.replace("&apos;", "'");

    plain
}

/// Calculate token savings from stripping markdown.
///
/// # Arguments
///
/// * `original` - The original markdown text.
/// * `stripped` - The text after stripping markdown.
///
/// # Returns
///
/// The number of characters saved (approximate token savings).
#[cfg(feature = "markdown")]
pub fn calculate_savings(original: &str, stripped: &str) -> usize {
    original.len().saturating_sub(stripped.len())
}

#[cfg(test)]
#[cfg(feature = "markdown")]
mod tests {
    use super::*;

    #[test]
    fn test_strip_markdown() {
        let markdown = "# Title\n\nThis is **bold** text.";
        let plain = strip_markdown(markdown);
        assert!(plain.contains("Title"));
        assert!(plain.contains("bold"));
    }

    #[test]
    fn test_calculate_savings() {
        let original = "# Title\n\n**Bold** text.";
        let stripped = strip_markdown(original);
        let savings = calculate_savings(original, &stripped);
        assert!(savings > 0);
    }
}
