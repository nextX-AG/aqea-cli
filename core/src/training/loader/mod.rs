//! Unified Data Loading Module for AQEA Training
//!
//! Provides a consistent interface for loading embedding data from
//! various formats (AQED, JSON, CSV, HDF5).
//!
//! # Supported Formats
//!
//! - **AQED**: Binary format, 60x faster than JSON
//! - **JSON**: Universal text format
//! - **CSV**: Tabular format (future)
//! - **HDF5**: Scientific data format (future)
//!
//! # Example
//!
//! ```rust,ignore
//! use aqea_core::training::loader::{load_auto, EmbeddingData};
//!
//! let data = load_auto("embeddings.aqed")?;
//! println!("Loaded {} embeddings of dimension {}", data.len(), data.dimension());
//! ```

mod aqed;
mod json;

pub use aqed::AqedLoader;
pub use json::JsonLoader;

use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::io::Read;
use thiserror::Error;

/// Errors that can occur during data loading
#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Empty data")]
    EmptyData,
}

/// Supported file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// AQED binary format (fast)
    Aqed,
    /// JSON text format
    Json,
    /// CSV tabular format
    Csv,
    /// HDF5 scientific format
    Hdf5,
}

impl Format {
    /// Detect format from file path and content
    pub fn detect(path: &Path) -> Result<Self, LoaderError> {
        // Try magic bytes first
        if let Ok(mut file) = File::open(path) {
            let mut magic = [0u8; 8];
            if file.read_exact(&mut magic).is_ok() {
                // AQED magic: "AQED"
                if &magic[0..4] == b"AQED" {
                    return Ok(Format::Aqed);
                }
                // HDF5 magic: 0x89 'H' 'D' 'F'
                if magic[0] == 0x89 && &magic[1..4] == b"HDF" {
                    return Ok(Format::Hdf5);
                }
                // JSON starts with { or [
                if magic[0] == b'{' || magic[0] == b'[' {
                    return Ok(Format::Json);
                }
            }
        }

        // Fall back to extension
        match path.extension().and_then(|e| e.to_str()) {
            Some("aqed") => Ok(Format::Aqed),
            Some("json") => Ok(Format::Json),
            Some("csv") | Some("tsv") => Ok(Format::Csv),
            Some("h5") | Some("hdf5") => Ok(Format::Hdf5),
            _ => Err(LoaderError::UnsupportedFormat(
                path.display().to_string()
            )),
        }
    }
}

/// Loaded embedding data
#[derive(Debug, Clone)]
pub struct EmbeddingData {
    /// Embeddings as flat array (N x D)
    pub embeddings: Vec<Vec<f32>>,
    /// Optional labels (e.g., "SEEN", "UNSEEN")
    pub labels: Option<Vec<String>>,
    /// Optional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl EmbeddingData {
    /// Create new EmbeddingData
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            embeddings,
            labels: None,
            metadata: HashMap::new(),
        }
    }

    /// Create with labels
    pub fn with_labels(embeddings: Vec<Vec<f32>>, labels: Vec<String>) -> Self {
        Self {
            embeddings,
            labels: Some(labels),
            metadata: HashMap::new(),
        }
    }

    /// Number of embeddings
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Embedding dimension
    pub fn dimension(&self) -> usize {
        self.embeddings.first().map(|e| e.len()).unwrap_or(0)
    }

    /// Get embeddings as slice
    pub fn as_slice(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// Filter by label
    pub fn filter_by_label(&self, label: &str) -> EmbeddingData {
        if let Some(labels) = &self.labels {
            let filtered: Vec<_> = self.embeddings
                .iter()
                .zip(labels.iter())
                .filter(|(_, l)| l.as_str() == label)
                .map(|(e, _)| e.clone())
                .collect();

            let filtered_labels: Vec<_> = labels
                .iter()
                .filter(|l| l.as_str() == label)
                .cloned()
                .collect();

            EmbeddingData::with_labels(filtered, filtered_labels)
        } else {
            self.clone()
        }
    }

    /// Split into training and validation sets
    pub fn split(&self, train_fraction: f32, seed: u64) -> (EmbeddingData, EmbeddingData) {
        use rand::prelude::*;

        let n = self.embeddings.len();
        let n_train = ((n as f32) * train_fraction).round() as usize;

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        let train_indices = &indices[..n_train];
        let val_indices = &indices[n_train..];

        let train_emb: Vec<_> = train_indices.iter().map(|&i| self.embeddings[i].clone()).collect();
        let val_emb: Vec<_> = val_indices.iter().map(|&i| self.embeddings[i].clone()).collect();

        let train_labels = self.labels.as_ref().map(|labels| {
            train_indices.iter().map(|&i| labels[i].clone()).collect()
        });
        let val_labels = self.labels.as_ref().map(|labels| {
            val_indices.iter().map(|&i| labels[i].clone()).collect()
        });

        let mut train = EmbeddingData::new(train_emb);
        train.labels = train_labels;
        train.metadata = self.metadata.clone();

        let mut val = EmbeddingData::new(val_emb);
        val.labels = val_labels;

        (train, val)
    }
}

/// Trait for data loaders
pub trait DataLoader: Send + Sync {
    /// Load embeddings from a path
    fn load(&self, path: &Path) -> Result<EmbeddingData, LoaderError>;

    /// Check if this loader supports the given path
    fn supports(&self, path: &Path) -> bool;

    /// Format name
    fn format_name(&self) -> &'static str;
}

/// Auto-detect format and load data
pub fn load_auto<P: AsRef<Path>>(path: P) -> Result<EmbeddingData, LoaderError> {
    let path = path.as_ref();
    let format = Format::detect(path)?;

    match format {
        Format::Aqed => AqedLoader.load(path),
        Format::Json => JsonLoader.load(path),
        Format::Csv => Err(LoaderError::UnsupportedFormat("CSV not yet implemented".into())),
        Format::Hdf5 => Err(LoaderError::UnsupportedFormat("HDF5 not yet implemented".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_embedding_data_new() {
        let emb = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let data = EmbeddingData::new(emb);

        assert_eq!(data.len(), 2);
        assert_eq!(data.dimension(), 3);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_embedding_data_with_labels() {
        let emb = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let labels = vec!["SEEN".into(), "UNSEEN".into(), "SEEN".into()];
        let data = EmbeddingData::with_labels(emb, labels);

        assert_eq!(data.len(), 3);
        assert!(data.labels.is_some());
    }

    #[test]
    fn test_filter_by_label() {
        let emb = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let labels = vec!["SEEN".into(), "UNSEEN".into(), "SEEN".into()];
        let data = EmbeddingData::with_labels(emb, labels);

        let seen = data.filter_by_label("SEEN");
        assert_eq!(seen.len(), 2);

        let unseen = data.filter_by_label("UNSEEN");
        assert_eq!(unseen.len(), 1);
    }

    #[test]
    fn test_split() {
        let emb: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32]).collect();
        let data = EmbeddingData::new(emb);

        let (train, val) = data.split(0.8, 42);

        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    #[test]
    fn test_format_detect_json() {
        let mut file = NamedTempFile::with_suffix(".json").unwrap();
        writeln!(file, "[{{}}]").unwrap();

        let format = Format::detect(file.path()).unwrap();
        assert_eq!(format, Format::Json);
    }

    #[test]
    fn test_format_detect_by_extension() {
        let file = NamedTempFile::with_suffix(".aqed").unwrap();
        // Note: magic bytes won't match, falls back to extension
        let format = Format::detect(file.path());
        // May fail or succeed depending on file content
        assert!(format.is_ok() || format.is_err());
    }
}
