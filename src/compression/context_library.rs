/// Context library management
///
/// Handles loading, saving, and querying context libraries
use crate::compression::types::{ContextLibrary, ContextPattern};
use crate::error::AppError;
use std::fs;
use std::path::Path;

/// Manager for context libraries
#[derive(Clone)]
pub struct ContextLibraryManager {
    library: ContextLibrary,
}

impl ContextLibraryManager {
    /// Create a new empty library manager
    pub fn new() -> Self {
        Self {
            library: ContextLibrary::new(),
        }
    }

    /// Load a context library from a TOML file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, AppError> {
        let content = fs::read_to_string(&path).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to read context library '{}': {}",
                path.as_ref().display(),
                e
            )))
        })?;

        let library: ContextLibrary = toml::from_str(&content).map_err(|e| {
            AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                "Failed to parse context library: {}",
                e
            )))
        })?;

        Ok(Self { library })
    }

    /// Save the context library to a TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), AppError> {
        let content = toml::to_string_pretty(&self.library).map_err(|e| {
            AppError::Parse(crate::error::ParseError::InvalidFormat(format!(
                "Failed to serialize context library: {}",
                e
            )))
        })?;

        fs::write(&path, content).map_err(|e| {
            AppError::Io(std::io::Error::other(format!(
                "Failed to write context library '{}': {}",
                path.as_ref().display(),
                e
            )))
        })?;

        Ok(())
    }

    /// Get a reference to the underlying library
    pub fn library(&self) -> &ContextLibrary {
        &self.library
    }

    /// Get a mutable reference to the underlying library
    pub fn library_mut(&mut self) -> &mut ContextLibrary {
        &mut self.library
    }

    /// Add a new pattern to the library
    pub fn add_pattern(&mut self, pattern: ContextPattern) {
        self.library.add_pattern(pattern);
    }

    /// Get a pattern by ID
    pub fn get_pattern(&self, id: &str) -> Option<&ContextPattern> {
        self.library.get_pattern(id)
    }

    /// Find patterns similar to given content
    pub fn find_similar_patterns(
        &self,
        content: &str,
        min_similarity: f64,
    ) -> Vec<&ContextPattern> {
        use crate::compression::similarity::normalized_similarity;

        self.library
            .patterns
            .iter()
            .filter_map(|pattern| {
                let similarity = normalized_similarity(&pattern.content, content);
                if similarity >= min_similarity {
                    Some(pattern)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all patterns of a category
    pub fn patterns_by_category(&self, category: &str) -> Vec<&ContextPattern> {
        self.library.patterns_by_category(category)
    }

    /// Update metadata
    pub fn set_source_directory<P: AsRef<Path>>(&mut self, path: P) {
        self.library.metadata.source_directory = Some(path.as_ref().display().to_string());
    }

    /// Merge another library into this one
    pub fn merge(&mut self, other: &ContextLibrary) {
        for pattern in &other.patterns {
            // Check if pattern already exists
            if self.library.get_pattern(&pattern.id).is_none() {
                self.library.add_pattern(pattern.clone());
            }
        }
    }

    /// Remove patterns with frequency below threshold
    pub fn prune_low_frequency(&mut self, min_frequency: usize) {
        self.library
            .patterns
            .retain(|p| p.frequency >= min_frequency);
        self.library.metadata.total_patterns = self.library.patterns.len();
    }

    /// Sort patterns by frequency (descending)
    pub fn sort_by_frequency(&mut self) {
        self.library
            .patterns
            .sort_by(|a, b| b.frequency.cmp(&a.frequency));
    }

    /// Get library statistics
    pub fn statistics(&self) -> LibraryStatistics {
        let total_patterns = self.library.patterns.len();
        let total_tokens: usize = self.library.patterns.iter().map(|p| p.avg_tokens).sum();
        let avg_tokens_per_pattern = if total_patterns > 0 {
            total_tokens / total_patterns
        } else {
            0
        };

        let mut category_counts = std::collections::HashMap::new();
        for pattern in &self.library.patterns {
            *category_counts.entry(pattern.category.clone()).or_insert(0) += 1;
        }

        LibraryStatistics {
            total_patterns,
            total_tokens,
            avg_tokens_per_pattern,
            category_counts,
        }
    }
}

impl Default for ContextLibraryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a context library
#[derive(Debug, Clone)]
pub struct LibraryStatistics {
    pub total_patterns: usize,
    pub total_tokens: usize,
    pub avg_tokens_per_pattern: usize,
    pub category_counts: std::collections::HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_create_and_add_pattern() {
        let mut manager = ContextLibraryManager::new();
        let pattern = ContextPattern::new(
            "test_001".to_string(),
            "Test content".to_string(),
            "role".to_string(),
        );

        manager.add_pattern(pattern);
        assert_eq!(manager.library().patterns.len(), 1);
        assert!(manager.get_pattern("test_001").is_some());
    }

    #[test]
    fn test_save_and_load() {
        let mut manager = ContextLibraryManager::new();
        let pattern = ContextPattern::new(
            "test_001".to_string(),
            "Test content".to_string(),
            "role".to_string(),
        );
        manager.add_pattern(pattern);

        // Save to temp file
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();

        manager.save_to_file(&temp_path).unwrap();

        // Load back
        let loaded = ContextLibraryManager::load_from_file(&temp_path).unwrap();
        assert_eq!(loaded.library().patterns.len(), 1);
        assert!(loaded.get_pattern("test_001").is_some());
    }

    #[test]
    fn test_patterns_by_category() {
        let mut manager = ContextLibraryManager::new();
        manager.add_pattern(ContextPattern::new(
            "role_001".to_string(),
            "Role 1".to_string(),
            "role".to_string(),
        ));
        manager.add_pattern(ContextPattern::new(
            "example_001".to_string(),
            "Example 1".to_string(),
            "examples".to_string(),
        ));
        manager.add_pattern(ContextPattern::new(
            "role_002".to_string(),
            "Role 2".to_string(),
            "role".to_string(),
        ));

        let roles = manager.patterns_by_category("role");
        assert_eq!(roles.len(), 2);

        let examples = manager.patterns_by_category("examples");
        assert_eq!(examples.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let mut manager = ContextLibraryManager::new();
        let mut pattern1 = ContextPattern::new(
            "test_001".to_string(),
            "Test 1".to_string(),
            "role".to_string(),
        );
        pattern1.avg_tokens = 100;

        let mut pattern2 = ContextPattern::new(
            "test_002".to_string(),
            "Test 2".to_string(),
            "examples".to_string(),
        );
        pattern2.avg_tokens = 200;

        manager.add_pattern(pattern1);
        manager.add_pattern(pattern2);

        let stats = manager.statistics();
        assert_eq!(stats.total_patterns, 2);
        assert_eq!(stats.total_tokens, 300);
        assert_eq!(stats.avg_tokens_per_pattern, 150);
    }

    #[test]
    fn test_prune_low_frequency() {
        let mut manager = ContextLibraryManager::new();

        let mut pattern1 = ContextPattern::new(
            "test_001".to_string(),
            "Test 1".to_string(),
            "role".to_string(),
        );
        pattern1.frequency = 5;

        let mut pattern2 = ContextPattern::new(
            "test_002".to_string(),
            "Test 2".to_string(),
            "role".to_string(),
        );
        pattern2.frequency = 1;

        manager.add_pattern(pattern1);
        manager.add_pattern(pattern2);

        assert_eq!(manager.library().patterns.len(), 2);

        manager.prune_low_frequency(3);
        assert_eq!(manager.library().patterns.len(), 1);
        assert!(manager.get_pattern("test_001").is_some());
        assert!(manager.get_pattern("test_002").is_none());
    }
}
