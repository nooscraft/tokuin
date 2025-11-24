/// Hieratic prompt compression module
///
/// This module provides functionality for compressing LLM prompts using the Hieratic format,
/// a structured, LLM-parseable serialization format that enables 70-90% token reduction.
pub mod types;

#[cfg(feature = "compression")]
pub mod similarity;

#[cfg(feature = "compression")]
pub mod parser;

#[cfg(feature = "compression")]
pub mod context_library;

#[cfg(feature = "compression")]
pub mod pattern_extractor;

#[cfg(feature = "compression")]
pub mod compressor;

#[cfg(feature = "compression")]
pub mod hieratic_encoder;

#[cfg(feature = "compression")]
pub mod hieratic_decoder;

#[cfg(feature = "compression-embeddings")]
pub mod embeddings;

// Re-export main types
pub use types::{
    CompressionConfig, CompressionLevel, CompressionResult, ContextLibrary, ContextPattern,
    HieraticDocument, HieraticSection, OutputFormat, ScoringMode,
};

#[cfg(feature = "compression")]
pub use compressor::Compressor;

#[cfg(feature = "compression")]
pub use hieratic_encoder::HieraticEncoder;

#[cfg(feature = "compression")]
pub use hieratic_decoder::HieraticDecoder;

#[cfg(feature = "compression")]
pub use context_library::ContextLibraryManager;

#[cfg(feature = "compression")]
pub use pattern_extractor::PatternExtractor;
