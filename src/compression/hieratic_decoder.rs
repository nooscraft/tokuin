/// Hieratic format decoder
///
/// Expands Hieratic documents back to full prompts
use crate::compression::context_library::ContextLibraryManager;
use crate::compression::parser::HieraticParser;
use crate::compression::types::HieraticSection;
use crate::error::AppError;
use std::path::Path;

/// Decoder for Hieratic format
pub struct HieraticDecoder {
    parser: HieraticParser,
    context_library: Option<ContextLibraryManager>,
}

impl HieraticDecoder {
    /// Create a new decoder
    pub fn new() -> Result<Self, AppError> {
        Ok(Self {
            parser: HieraticParser::new()?,
            context_library: None,
        })
    }

    /// Set context library
    pub fn with_context_library(mut self, library: ContextLibraryManager) -> Self {
        self.context_library = Some(library);
        self
    }

    /// Load context library from file
    pub fn load_context_library<P: AsRef<Path>>(mut self, path: P) -> Result<Self, AppError> {
        let library = ContextLibraryManager::load_from_file(path)?;
        self.context_library = Some(library);
        Ok(self)
    }

    /// Decode a Hieratic document to full prompt
    pub fn decode(&self, hieratic_text: &str) -> Result<String, AppError> {
        // Parse the Hieratic document
        let document = self.parser.parse(hieratic_text)?;

        // Validate
        self.parser.validate(&document)?;

        // Load context library if referenced
        let context_library = if let Some(ref path) = document.context_path {
            if self.context_library.is_none() {
                // Try to load from specified path
                Some(ContextLibraryManager::load_from_file(path)?)
            } else {
                self.context_library.clone()
            }
        } else {
            self.context_library.clone()
        };

        // Expand sections
        let mut output = String::new();

        for section in &document.sections {
            self.expand_section(&mut output, section, context_library.as_ref())?;
            output.push_str("\n\n");
        }

        Ok(output.trim().to_string())
    }

    /// Expand a single section
    fn expand_section(
        &self,
        output: &mut String,
        section: &HieraticSection,
        context_library: Option<&ContextLibraryManager>,
    ) -> Result<(), AppError> {
        match section {
            HieraticSection::Role { id, content } => {
                if let Some(content) = content {
                    // Inline content
                    output.push_str(content);
                } else if let Some(lib) = context_library {
                    // Referenced content
                    if let Some(pattern) = lib.get_pattern(id) {
                        output.push_str(&pattern.content);
                    } else {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            format!("Context pattern not found: {}", id),
                        )));
                    }
                } else {
                    return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                        format!(
                            "No context library available for referenced pattern: {}",
                            id
                        ),
                    )));
                }
            }
            HieraticSection::Examples { id, content } => {
                if let Some(content) = content {
                    output.push_str(content);
                } else if let Some(lib) = context_library {
                    if let Some(pattern) = lib.get_pattern(id) {
                        output.push_str(&pattern.content);
                    } else {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            format!("Context pattern not found: {}", id),
                        )));
                    }
                } else {
                    return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                        format!(
                            "No context library available for referenced pattern: {}",
                            id
                        ),
                    )));
                }
            }
            HieraticSection::Constraints { id, content } => {
                if let Some(content) = content {
                    output.push_str(content);
                } else if let Some(lib) = context_library {
                    if let Some(pattern) = lib.get_pattern(id) {
                        output.push_str(&pattern.content);
                    } else {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            format!("Context pattern not found: {}", id),
                        )));
                    }
                } else {
                    return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                        format!(
                            "No context library available for referenced pattern: {}",
                            id
                        ),
                    )));
                }
            }
            HieraticSection::Task { content } => {
                output.push_str(content);
            }
            HieraticSection::Focus { areas } => {
                output.push_str("Focus on: ");
                output.push_str(&areas.join(", "));
            }
            HieraticSection::Style { preferences } => {
                output.push_str("Style: ");
                output.push_str(&preferences.join(", "));
            }
            HieraticSection::Format { structure } => {
                output.push_str("Format: ");
                output.push_str(structure);
            }
            HieraticSection::Context { .. } => {
                // Context directive doesn't contribute to output
            }
        }

        Ok(())
    }
}

impl Default for HieraticDecoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default decoder")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::types::ContextPattern;

    #[test]
    fn test_decode_inline_document() {
        let hieratic = r#"@HIERATIC v1.0

@ROLE[inline]
"Expert programmer"

@TASK
Write some code
"#;

        let decoder = HieraticDecoder::new().unwrap();
        let decoded = decoder.decode(hieratic).unwrap();

        assert!(decoded.contains("Expert programmer"));
        assert!(decoded.contains("Write some code"));
    }

    #[test]
    fn test_decode_with_context_library() {
        let mut lib_manager = ContextLibraryManager::new();
        lib_manager.add_pattern(ContextPattern::new(
            "role_001".to_string(),
            "You are an expert developer".to_string(),
            "role".to_string(),
        ));

        let hieratic = r#"@HIERATIC v1.0

@ROLE[role_001]

@TASK
Review this code
"#;

        let decoder = HieraticDecoder::new()
            .unwrap()
            .with_context_library(lib_manager);
        let decoded = decoder.decode(hieratic).unwrap();

        assert!(decoded.contains("You are an expert developer"));
        assert!(decoded.contains("Review this code"));
    }

    #[test]
    fn test_decode_missing_pattern() {
        let hieratic = r#"@HIERATIC v1.0

@ROLE[nonexistent]

@TASK
Test
"#;

        let decoder = HieraticDecoder::new().unwrap();
        let result = decoder.decode(hieratic);

        assert!(result.is_err());
    }
}
