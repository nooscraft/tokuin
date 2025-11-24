/// Hieratic format encoder
///
/// Converts Hieratic documents to text format
use crate::compression::types::{HieraticDocument, HieraticSection};
use crate::error::AppError;

/// Encoder for Hieratic format
pub struct HieraticEncoder;

impl HieraticEncoder {
    /// Create a new encoder
    pub fn new() -> Self {
        Self
    }

    /// Encode a Hieratic document to string
    pub fn encode(&self, document: &HieraticDocument) -> Result<String, AppError> {
        let mut output = String::new();

        // Header
        output.push_str(&format!("@HIERATIC v{}\n", document.version));

        // Context library reference
        if let Some(ref path) = document.context_path {
            output.push_str(&format!("@CONTEXT: {}\n", path));
        }

        output.push('\n');

        // Sections
        for section in &document.sections {
            self.encode_section(&mut output, section)?;
            output.push('\n');
        }

        Ok(output)
    }

    /// Encode a single section
    fn encode_section(
        &self,
        output: &mut String,
        section: &HieraticSection,
    ) -> Result<(), AppError> {
        match section {
            HieraticSection::Role { id, content } => {
                output.push_str(&format!("@ROLE[{}]\n", id));
                if let Some(content) = content {
                    output.push_str(content);
                    output.push('\n');
                }
            }
            HieraticSection::Examples { id, content } => {
                output.push_str(&format!("@EXAMPLES[{}]\n", id));
                if let Some(content) = content {
                    output.push_str(content);
                    output.push('\n');
                }
            }
            HieraticSection::Constraints { id, content } => {
                output.push_str(&format!("@CONSTRAINTS[{}]\n", id));
                if let Some(content) = content {
                    output.push_str(content);
                    output.push('\n');
                }
            }
            HieraticSection::Task { content } => {
                output.push_str("@TASK\n");
                output.push_str(content);
                output.push('\n');
            }
            HieraticSection::Focus { areas } => {
                output.push_str(&format!("@FOCUS: {}\n", areas.join(", ")));
            }
            HieraticSection::Style { preferences } => {
                output.push_str(&format!("@STYLE: {}\n", preferences.join(", ")));
            }
            HieraticSection::Format { structure } => {
                output.push_str(&format!("@FORMAT: {}\n", structure));
            }
            HieraticSection::Context { path } => {
                // Already handled in header
                output.push_str(&format!("@CONTEXT: {}\n", path));
            }
        }

        Ok(())
    }
}

impl Default for HieraticEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_simple_document() {
        let mut doc = HieraticDocument::new();
        doc.add_section(HieraticSection::Role {
            id: "inline".to_string(),
            content: Some("Expert programmer".to_string()),
        });
        doc.add_section(HieraticSection::Task {
            content: "Write code".to_string(),
        });

        let encoder = HieraticEncoder::new();
        let output = encoder.encode(&doc).unwrap();

        assert!(output.contains("@HIERATIC v1.0"));
        assert!(output.contains("@ROLE[inline]"));
        assert!(output.contains("Expert programmer"));
        assert!(output.contains("@TASK"));
        assert!(output.contains("Write code"));
    }

    #[test]
    fn test_encode_with_context() {
        let mut doc = HieraticDocument::new();
        doc.context_path = Some("contexts.toml".to_string());
        doc.add_section(HieraticSection::Role {
            id: "role_001".to_string(),
            content: None,
        });
        doc.add_section(HieraticSection::Task {
            content: "Test task".to_string(),
        });

        let encoder = HieraticEncoder::new();
        let output = encoder.encode(&doc).unwrap();

        assert!(output.contains("@CONTEXT: contexts.toml"));
        assert!(output.contains("@ROLE[role_001]"));
    }

    #[test]
    fn test_encode_with_directives() {
        let mut doc = HieraticDocument::new();
        doc.add_section(HieraticSection::Task {
            content: "Main task".to_string(),
        });
        doc.add_section(HieraticSection::Focus {
            areas: vec!["performance".to_string(), "security".to_string()],
        });
        doc.add_section(HieraticSection::Style {
            preferences: vec!["concise".to_string(), "actionable".to_string()],
        });

        let encoder = HieraticEncoder::new();
        let output = encoder.encode(&doc).unwrap();

        assert!(output.contains("@FOCUS: performance, security"));
        assert!(output.contains("@STYLE: concise, actionable"));
    }
}
