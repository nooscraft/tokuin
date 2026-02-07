/// Hieratic format parser
///
/// Parses .hieratic files and validates syntax
use crate::compression::types::{HieraticDocument, HieraticSection};
use crate::error::AppError;
use regex::Regex;

/// Parser for Hieratic format
pub struct HieraticParser {
    version_regex: Regex,
    directive_regex: Regex,
}

impl HieraticParser {
    /// Create a new parser
    pub fn new() -> Result<Self, AppError> {
        Ok(Self {
            version_regex: Regex::new(r"^@HIERATIC\s+(v?\d+\.\d+)").map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(e.to_string()))
            })?,
            directive_regex: Regex::new(r"^@([A-Z_]+)(\[([^\]]+)\])?:\s*(.*)$").map_err(|e| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(e.to_string()))
            })?,
        })
    }

    /// Parse a Hieratic document from text
    pub fn parse(&self, content: &str) -> Result<HieraticDocument, AppError> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "Empty Hieratic document".to_string(),
            )));
        }

        // Parse version from first line
        let version = self.parse_version(lines[0])?;
        let mut doc = HieraticDocument {
            version,
            context_path: None,
            sections: Vec::new(),
        };

        let mut i = 1;
        while i < lines.len() {
            let line = lines[i].trim();

            // Skip empty lines
            if line.is_empty() {
                i += 1;
                continue;
            }

            // Parse directive
            if line.starts_with('@') {
                let (section, lines_consumed) = self.parse_directive(&lines[i..])?;

                // Handle CONTEXT directive specially
                if let HieraticSection::Context { path } = &section {
                    doc.context_path = Some(path.clone());
                } else {
                    doc.sections.push(section);
                }

                i += lines_consumed;
            } else {
                i += 1;
            }
        }

        Ok(doc)
    }

    /// Parse version from first line
    fn parse_version(&self, line: &str) -> Result<String, AppError> {
        let captures = self.version_regex.captures(line).ok_or_else(|| {
            AppError::Parse(crate::error::ParseError::InvalidFormat(
                "Missing or invalid @HIERATIC version declaration".to_string(),
            ))
        })?;

        let version = captures
            .get(1)
            .map(|m| m.as_str().trim_start_matches('v').to_string())
            .ok_or_else(|| {
                AppError::Parse(crate::error::ParseError::InvalidFormat(
                    "Could not extract version number".to_string(),
                ))
            })?;

        Ok(version)
    }

    /// Parse a directive and its content
    fn parse_directive(&self, lines: &[&str]) -> Result<(HieraticSection, usize), AppError> {
        if lines.is_empty() {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "Expected directive but found end of document".to_string(),
            )));
        }

        let first_line = lines[0].trim();

        // Handle single-line directives (CONTEXT, FOCUS, STYLE, FORMAT)
        if let Some(captures) = self.directive_regex.captures(first_line) {
            let directive = captures
                .get(1)
                .ok_or_else(|| {
                    AppError::Parse(crate::error::ParseError::InvalidFormat(
                        "Missing directive name in Hieratic section".to_string(),
                    ))
                })?
                .as_str();
            let _id = captures.get(3).map(|m| m.as_str().to_string());
            let value = captures
                .get(4)
                .ok_or_else(|| {
                    AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                        "Missing value for @{} directive",
                        directive
                    )))
                })?
                .as_str()
                .trim();

            match directive {
                "CONTEXT" => {
                    return Ok((
                        HieraticSection::Context {
                            path: value.to_string(),
                        },
                        1,
                    ));
                }
                "FOCUS" => {
                    let areas = value
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    return Ok((HieraticSection::Focus { areas }, 1));
                }
                "STYLE" => {
                    let preferences = value
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    return Ok((HieraticSection::Style { preferences }, 1));
                }
                "FORMAT" => {
                    return Ok((
                        HieraticSection::Format {
                            structure: value.to_string(),
                        },
                        1,
                    ));
                }
                _ => {
                    // Multi-line directive - continue below
                }
            }
        }

        // Handle multi-line directives (ROLE, EXAMPLES, CONSTRAINTS, TASK)
        let directive_line = first_line;
        if !directive_line.starts_with('@') {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                format!("Expected directive, found: {}", directive_line),
            )));
        }

        // Extract directive name and optional ID
        let (directive, id) = if let Some(bracket_pos) = directive_line.find('[') {
            let directive = &directive_line[1..bracket_pos];
            if let Some(end_bracket) = directive_line.find(']') {
                let id = &directive_line[bracket_pos + 1..end_bracket];
                (directive, Some(id.to_string()))
            } else {
                return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                    "Unclosed bracket in directive".to_string(),
                )));
            }
        } else {
            let directive = directive_line.trim_start_matches('@').trim();
            (directive, None)
        };

        // Collect content until next directive or end
        let mut content_lines = Vec::new();
        let mut lines_consumed = 1;

        for line in lines.iter().skip(1) {
            let trimmed = line.trim();
            if trimmed.starts_with('@') && !trimmed.starts_with("@ ") {
                // Next directive found
                break;
            }
            content_lines.push(*line);
            lines_consumed += 1;
        }

        let content = content_lines.join("\n").trim().to_string();

        // Create appropriate section
        let section = match directive {
            "ROLE" => HieraticSection::Role {
                id: id.unwrap_or_else(|| "inline".to_string()),
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
            },
            "EXAMPLES" => HieraticSection::Examples {
                id: id.unwrap_or_else(|| "inline".to_string()),
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
            },
            "CONSTRAINTS" => HieraticSection::Constraints {
                id: id.unwrap_or_else(|| "inline".to_string()),
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
            },
            "TASK" => HieraticSection::Task { content },
            _ => {
                return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                    format!("Unknown directive: {}", directive),
                )));
            }
        };

        Ok((section, lines_consumed))
    }

    /// Validate a Hieratic document
    pub fn validate(&self, doc: &HieraticDocument) -> Result<(), AppError> {
        // Check version
        if doc.version.is_empty() {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "Missing version".to_string(),
            )));
        }

        // Check that TASK section exists
        if doc.get_task().is_none() {
            return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                "Missing required @TASK section".to_string(),
            )));
        }

        // Validate that referenced sections have either content or valid ID
        for section in &doc.sections {
            match section {
                HieraticSection::Role { id, content } => {
                    if id != "inline" && content.is_some() {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            "Referenced ROLE should not have inline content".to_string(),
                        )));
                    }
                    if id == "inline" && content.is_none() {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            "Inline ROLE must have content".to_string(),
                        )));
                    }
                }
                HieraticSection::Examples { id, content } => {
                    if id != "inline" && content.is_some() {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            "Referenced EXAMPLES should not have inline content".to_string(),
                        )));
                    }
                    if id == "inline" && content.is_none() {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            "Inline EXAMPLES must have content".to_string(),
                        )));
                    }
                }
                HieraticSection::Constraints { id, content } => {
                    if id != "inline" && content.is_some() {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            "Referenced CONSTRAINTS should not have inline content".to_string(),
                        )));
                    }
                    if id == "inline" && content.is_none() {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            "Inline CONSTRAINTS must have content".to_string(),
                        )));
                    }
                }
                HieraticSection::Task { content } => {
                    if content.is_empty() {
                        return Err(AppError::Parse(crate::error::ParseError::InvalidFormat(
                            "TASK section cannot be empty".to_string(),
                        )));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

impl Default for HieraticParser {
    fn default() -> Self {
        Self::new().expect("Failed to create default parser")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_version() {
        let parser = HieraticParser::new().unwrap();
        assert_eq!(parser.parse_version("@HIERATIC v1.0").unwrap(), "1.0");
        assert_eq!(parser.parse_version("@HIERATIC 1.0").unwrap(), "1.0");
        assert!(parser.parse_version("HIERATIC 1.0").is_err());
    }

    #[test]
    fn test_parse_simple_document() {
        let parser = HieraticParser::new().unwrap();
        let content = r#"@HIERATIC v1.0

@ROLE[inline]
"Test role"

@TASK
Do something
"#;

        let doc = parser.parse(content).unwrap();
        assert_eq!(doc.version, "1.0");
        assert_eq!(doc.sections.len(), 2);
        assert_eq!(doc.get_task(), Some("Do something"));
    }

    #[test]
    fn test_parse_with_context() {
        let parser = HieraticParser::new().unwrap();
        let content = r#"@HIERATIC v1.0
@CONTEXT: contexts.toml

@ROLE[role_001]

@TASK
Test task
"#;

        let doc = parser.parse(content).unwrap();
        assert_eq!(doc.context_path, Some("contexts.toml".to_string()));
    }

    #[test]
    fn test_parse_with_directives() {
        let parser = HieraticParser::new().unwrap();
        let content = r#"@HIERATIC v1.0

@TASK
Main task

@FOCUS: performance, security
@STYLE: concise, actionable
"#;

        let doc = parser.parse(content).unwrap();
        assert_eq!(doc.sections.len(), 3);

        // Check FOCUS
        if let Some(HieraticSection::Focus { areas }) = doc.sections.get(1) {
            assert_eq!(areas.len(), 2);
            assert_eq!(areas[0], "performance");
        } else {
            panic!("Expected FOCUS section");
        }
    }

    #[test]
    fn test_validate_missing_task() {
        let parser = HieraticParser::new().unwrap();
        let doc = HieraticDocument {
            version: "1.0".to_string(),
            context_path: None,
            sections: vec![],
        };

        assert!(parser.validate(&doc).is_err());
    }

    #[test]
    fn test_validate_valid_document() {
        let parser = HieraticParser::new().unwrap();
        let doc = HieraticDocument {
            version: "1.0".to_string(),
            context_path: None,
            sections: vec![HieraticSection::Task {
                content: "Test".to_string(),
            }],
        };

        assert!(parser.validate(&doc).is_ok());
    }
}
