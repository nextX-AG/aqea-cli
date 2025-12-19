//! Compression Module - AQEA Octonion Compressor
//!
//! Core compression using Octonion-based weighted aggregation.
//! This is the production-ready compressor from unified_optimizer.
//!
//! Features:
//! - Block-based dimension grouping
//! - Trainable weight matrices
//! - Golden ratio weighting (default)
//! - Serialization for weight persistence

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Golden ratio
const PHI: f32 = 1.618033988749895;

/// Trait for compression algorithms
pub trait Compressor: Send + Sync {
    /// Compress high-dimensional embedding to low-dimensional
    fn compress(&self, embedding: &[f32]) -> Vec<f32>;

    /// Get compressed dimension
    fn compressed_dim(&self) -> usize;

    /// Get original dimension
    fn original_dim(&self) -> usize;
}

/// Octonion-aware compressor - AQEA Core Implementation
///
/// Flexible implementation supporting any input/output dimensions.
/// The core principle: Group input dimensions into "octonion-like" blocks
/// and aggregate them using weighted averaging with golden ratio weighting.
///
/// For classic 384D → 8D: Groups 384 into 48 octonions, aggregates to 1.
/// For 384D → 13D: Groups 384 into blocks, aggregates to 13 output dims.
/// For 768D → 26D: Groups 768 into blocks, aggregates to 26 output dims.
pub struct OctonionCompressor {
    original_dim: usize,
    compressed_dim: usize,
    /// Block size for grouping (default: 8 for octonion structure)
    block_size: usize,
    /// Pre-computed aggregation weights (golden ratio based)
    weights: Vec<Vec<f32>>,
    /// Whether using trained weights
    is_trained: bool,
}

impl OctonionCompressor {
    /// Create classic 384D → 8D compressor
    pub fn new() -> Self {
        Self::with_dims(384, 8)
    }

    /// Create compressor with custom dimensions
    ///
    /// # Arguments
    /// * `original_dim` - Input dimension (e.g., 384, 768, 1024)
    /// * `compressed_dim` - Output dimension (e.g., 4, 8, 13, 26)
    pub fn with_dims(original_dim: usize, compressed_dim: usize) -> Self {
        // Block size: prefer 8 (octonion), but adapt if needed
        let block_size = 8;
        let weights = Self::compute_aggregation_weights(original_dim, compressed_dim, block_size);

        Self {
            original_dim,
            compressed_dim,
            block_size,
            weights,
            is_trained: false,
        }
    }

    /// Create compressor with trained weights
    ///
    /// # Arguments
    /// * `original_dim` - Input dimension
    /// * `compressed_dim` - Output dimension
    /// * `flat_weights` - Flattened weight vector (compressed_dim * original_dim)
    pub fn with_trained_weights(original_dim: usize, compressed_dim: usize, flat_weights: &[f32]) -> Self {
        assert_eq!(flat_weights.len(), compressed_dim * original_dim,
            "Expected {} weights, got {}", compressed_dim * original_dim, flat_weights.len());

        // Reshape flat weights into 2D matrix
        let mut weights = vec![vec![0.0f32; original_dim]; compressed_dim];
        for out_idx in 0..compressed_dim {
            for in_idx in 0..original_dim {
                weights[out_idx][in_idx] = flat_weights[out_idx * original_dim + in_idx];
            }
        }

        Self {
            original_dim,
            compressed_dim,
            block_size: 8,
            weights,
            is_trained: true,
        }
    }

    /// Load weights from file
    ///
    /// # Arguments
    /// * `original_dim` - Input dimension
    /// * `compressed_dim` - Output dimension
    /// * `path` - Path to weights file (.bin or .json)
    pub fn load_weights<P: AsRef<Path>>(original_dim: usize, compressed_dim: usize, path: P) -> Result<Self, std::io::Error> {
        let path = path.as_ref();
        
        if path.extension().map_or(false, |ext| ext == "json") {
            // JSON format
            let mut file = File::open(path)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            
            let flat_weights: Vec<f32> = serde_json::from_str(&contents)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            
            Ok(Self::with_trained_weights(original_dim, compressed_dim, &flat_weights))
        } else {
            // Binary format (f32 little-endian)
            let mut file = File::open(path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            
            let flat_weights: Vec<f32> = buffer
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            
            Ok(Self::with_trained_weights(original_dim, compressed_dim, &flat_weights))
        }
    }

    /// Save weights to file
    ///
    /// # Arguments
    /// * `path` - Path to save weights (.bin or .json)
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let path = path.as_ref();
        let flat_weights = self.get_flat_weights();
        
        if path.extension().map_or(false, |ext| ext == "json") {
            // JSON format
            let json = serde_json::to_string_pretty(&flat_weights)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let mut file = File::create(path)?;
            file.write_all(json.as_bytes())?;
        } else {
            // Binary format (f32 little-endian)
            let mut file = File::create(path)?;
            for &w in &flat_weights {
                file.write_all(&w.to_le_bytes())?;
            }
        }
        
        Ok(())
    }

    /// Get flattened weights for serialization/training
    pub fn get_flat_weights(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.compressed_dim * self.original_dim);
        for out_weights in &self.weights {
            flat.extend_from_slice(out_weights);
        }
        flat
    }

    /// Set weights from flat vector (for training)
    pub fn set_flat_weights(&mut self, flat_weights: &[f32]) {
        assert_eq!(flat_weights.len(), self.compressed_dim * self.original_dim);
        
        for out_idx in 0..self.compressed_dim {
            for in_idx in 0..self.original_dim {
                self.weights[out_idx][in_idx] = flat_weights[out_idx * self.original_dim + in_idx];
            }
        }
        self.is_trained = true;
    }

    /// Get weight matrix reference
    pub fn weights(&self) -> &Vec<Vec<f32>> {
        &self.weights
    }

    /// Check if using trained weights
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Compute golden-ratio weighted aggregation matrix
    ///
    /// This is the core AQEA principle: dimensions are grouped and weighted
    /// using golden ratio-based coefficients for optimal information preservation.
    fn compute_aggregation_weights(original_dim: usize, compressed_dim: usize, block_size: usize) -> Vec<Vec<f32>> {
        let mut weights = vec![vec![0.0f32; original_dim]; compressed_dim];

        // Number of input blocks
        let n_blocks = (original_dim + block_size - 1) / block_size;

        // For each output dimension
        for out_idx in 0..compressed_dim {
            // Determine which blocks contribute to this output
            // Use golden ratio to distribute blocks across outputs
            let start_block = (out_idx * n_blocks) / compressed_dim;
            let end_block = ((out_idx + 1) * n_blocks) / compressed_dim;

            // Weight each input dimension
            for block_idx in start_block..end_block.min(n_blocks) {
                let block_start = block_idx * block_size;
                let block_end = (block_start + block_size).min(original_dim);

                for in_idx in block_start..block_end {
                    // Golden ratio weighting within block
                    let pos_in_block = (in_idx - block_start) as f32;
                    let block_len = (block_end - block_start) as f32;

                    // Golden weighting: earlier positions get slightly more weight
                    let golden_weight = (1.0 + (-pos_in_block / block_len * PHI).exp()) / 2.0;

                    // Also weight by block position (closer blocks = more weight)
                    let block_distance = ((block_idx as f32) - (start_block as f32 + end_block as f32) / 2.0).abs();
                    let block_weight = 1.0 / (1.0 + block_distance * 0.1);

                    weights[out_idx][in_idx] = golden_weight * block_weight;
                }
            }

            // Normalize weights for this output dimension
            let sum: f32 = weights[out_idx].iter().sum();
            if sum > 1e-10 {
                for w in &mut weights[out_idx] {
                    *w /= sum;
                }
            }
        }

        weights
    }

    /// Compress embedding using pre-computed weights
    fn compress_with_weights(&self, embedding: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.compressed_dim];

        for (out_idx, out_weights) in self.weights.iter().enumerate() {
            for (in_idx, &weight) in out_weights.iter().enumerate() {
                if in_idx < embedding.len() && weight.abs() > 1e-10 {
                    result[out_idx] += embedding[in_idx] * weight;
                }
            }
        }

        result
    }
}

impl Default for OctonionCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for OctonionCompressor {
    fn compress(&self, embedding: &[f32]) -> Vec<f32> {
        self.compress_with_weights(embedding)
    }

    fn compressed_dim(&self) -> usize {
        self.compressed_dim
    }

    fn original_dim(&self) -> usize {
        self.original_dim
    }
}

// ============================================================================
// BATCH COMPRESSION
// ============================================================================

/// Compress multiple embeddings at once
pub fn compress_batch(compressor: &dyn Compressor, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    embeddings.iter()
        .map(|e| compressor.compress(e))
        .collect()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octonion_compressor_384_8() {
        let compressor = OctonionCompressor::new();
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();

        let compressed = compressor.compress(&embedding);
        assert_eq!(compressed.len(), 8);
        
        // All values should be non-zero
        assert!(compressed.iter().all(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_octonion_compressor_384_4() {
        let compressor = OctonionCompressor::with_dims(384, 4);
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();

        let compressed = compressor.compress(&embedding);
        assert_eq!(compressed.len(), 4);
    }

    #[test]
    fn test_octonion_compressor_768_13() {
        let compressor = OctonionCompressor::with_dims(768, 13);
        let embedding: Vec<f32> = (0..768).map(|i| i as f32 / 768.0).collect();

        let compressed = compressor.compress(&embedding);
        assert_eq!(compressed.len(), 13);
    }

    #[test]
    fn test_trained_weights() {
        let original_dim = 16;
        let compressed_dim = 4;
        
        // Create custom weights (identity-like for testing)
        let mut flat_weights = vec![0.0f32; compressed_dim * original_dim];
        for out_idx in 0..compressed_dim {
            for in_idx in 0..4 {
                flat_weights[out_idx * original_dim + out_idx * 4 + in_idx] = 0.25;
            }
        }
        
        let compressor = OctonionCompressor::with_trained_weights(original_dim, compressed_dim, &flat_weights);
        assert!(compressor.is_trained());
        
        let embedding: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let compressed = compressor.compress(&embedding);
        assert_eq!(compressed.len(), 4);
    }

    #[test]
    fn test_get_set_flat_weights() {
        let mut compressor = OctonionCompressor::with_dims(32, 4);
        let original_weights = compressor.get_flat_weights();
        
        // Modify and set back
        let modified: Vec<f32> = original_weights.iter().map(|w| w * 2.0).collect();
        compressor.set_flat_weights(&modified);
        
        let new_weights = compressor.get_flat_weights();
        for (o, n) in original_weights.iter().zip(new_weights.iter()) {
            assert!((n - o * 2.0).abs() < 1e-6);
        }
    }
}
