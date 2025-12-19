//! Stage 1: 8-bit Scalar Quantization
//!
//! Converts float32 embeddings to uint8 with min/max metadata.
//! Nearly lossless for ranking tasks (~100% Spearman preserved).
//!
//! # Compression
//! - Input: float32 (4 bytes per value)
//! - Output: uint8 (1 byte per value) + 8 bytes metadata
//! - Ratio: ~4x
//!
//! # Usage
//! ```rust
//! use aqea_core::quantization::{quantize_8bit, dequantize_8bit};
//!
//! let embedding = vec![0.1, 0.5, -0.3, 0.8];
//! let (quantized, min, max) = quantize_8bit(&embedding);
//! let restored = dequantize_8bit(&quantized, min, max);
//! // restored ≈ embedding (with small quantization error)
//! ```

use serde::{Deserialize, Serialize};

/// Quantized embedding with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedEmbedding {
    /// Quantized values (0-255)
    pub data: Vec<u8>,
    /// Minimum value from original
    pub min: f32,
    /// Maximum value from original
    pub max: f32,
}

impl QuantizedEmbedding {
    /// Create from raw components
    pub fn new(data: Vec<u8>, min: f32, max: f32) -> Self {
        Self { data, min, max }
    }

    /// Get the dimension
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Calculate byte size (data + metadata)
    pub fn byte_size(&self) -> usize {
        self.data.len() + 8 // 8 bytes for min/max (2 * f32)
    }

    /// Dequantize to float32
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_8bit(&self.data, self.min, self.max)
    }
}

/// Quantize float32 data to uint8
///
/// Maps values from [min, max] to [0, 255].
/// Returns (quantized_data, min, max).
///
/// # Arguments
/// * `data` - Float32 values to quantize
///
/// # Returns
/// * `Vec<u8>` - Quantized values (0-255)
/// * `f32` - Minimum value
/// * `f32` - Maximum value
///
/// # Example
/// ```rust
/// use aqea_core::quantization::quantize_8bit;
///
/// let data = vec![0.0, 0.5, 1.0];
/// let (quantized, min, max) = quantize_8bit(&data);
/// assert_eq!(min, 0.0);
/// assert_eq!(max, 1.0);
/// assert_eq!(quantized, vec![0, 127, 255]);
/// ```
pub fn quantize_8bit(data: &[f32]) -> (Vec<u8>, f32, f32) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }

    // Find min/max
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for &val in data {
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }

    // Handle edge case where all values are the same
    if (max - min).abs() < f32::EPSILON {
        return (vec![128u8; data.len()], min, max);
    }

    // Quantize to 0-255
    let scale = 255.0 / (max - min);
    let quantized: Vec<u8> = data
        .iter()
        .map(|&val| ((val - min) * scale).round().clamp(0.0, 255.0) as u8)
        .collect();

    (quantized, min, max)
}

/// Dequantize uint8 data back to float32
///
/// Maps values from [0, 255] back to [min, max].
///
/// # Arguments
/// * `data` - Quantized uint8 values
/// * `min` - Original minimum value
/// * `max` - Original maximum value
///
/// # Returns
/// * `Vec<f32>` - Restored float32 values
///
/// # Example
/// ```rust
/// use aqea_core::quantization::dequantize_8bit;
///
/// let quantized = vec![0, 127, 255];
/// let restored = dequantize_8bit(&quantized, 0.0, 1.0);
/// // restored ≈ [0.0, 0.498, 1.0]
/// ```
pub fn dequantize_8bit(data: &[u8], min: f32, max: f32) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }

    // Handle edge case where min == max
    if (max - min).abs() < f32::EPSILON {
        return vec![min; data.len()];
    }

    let scale = (max - min) / 255.0;
    data.iter()
        .map(|&val| min + (val as f32) * scale)
        .collect()
}

/// 8-bit Quantizer with configurable options
#[derive(Debug, Clone)]
pub struct EightBitQuantizer {
    /// Whether quantization is enabled
    enabled: bool,
}

impl Default for EightBitQuantizer {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl EightBitQuantizer {
    /// Create new quantizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Create disabled quantizer (pass-through)
    pub fn disabled() -> Self {
        Self { enabled: false }
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Quantize embedding
    pub fn quantize(&self, embedding: &[f32]) -> QuantizedEmbedding {
        let (data, min, max) = quantize_8bit(embedding);
        QuantizedEmbedding::new(data, min, max)
    }

    /// Process embedding (quantize then dequantize)
    /// Returns the processed embedding ready for next stage
    pub fn process(&self, embedding: &[f32]) -> Vec<f32> {
        if !self.enabled {
            return embedding.to_vec();
        }

        let (quantized, min, max) = quantize_8bit(embedding);
        dequantize_8bit(&quantized, min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_roundtrip() {
        let original = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let (quantized, min, max) = quantize_8bit(&original);
        let restored = dequantize_8bit(&quantized, min, max);

        // Check values are close (within quantization error)
        for (orig, rest) in original.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 0.01, "orig={}, rest={}", orig, rest);
        }
    }

    #[test]
    fn test_quantize_negative_values() {
        let original = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let (quantized, min, max) = quantize_8bit(&original);

        assert_eq!(min, -1.0);
        assert_eq!(max, 1.0);
        assert_eq!(quantized[0], 0);   // -1.0 -> 0
        assert_eq!(quantized[4], 255); // 1.0 -> 255
    }

    #[test]
    fn test_quantize_same_values() {
        let original = vec![0.5, 0.5, 0.5];
        let (quantized, min, max) = quantize_8bit(&original);
        let restored = dequantize_8bit(&quantized, min, max);

        for rest in &restored {
            assert!((rest - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantize_empty() {
        let original: Vec<f32> = vec![];
        let (quantized, _, _) = quantize_8bit(&original);
        assert!(quantized.is_empty());
    }

    #[test]
    fn test_quantizer_process() {
        let quantizer = EightBitQuantizer::new();
        let original = vec![0.1, 0.5, 0.9];
        let processed = quantizer.process(&original);

        assert_eq!(processed.len(), original.len());
        for (orig, proc) in original.iter().zip(processed.iter()) {
            assert!((orig - proc).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantizer_disabled() {
        let quantizer = EightBitQuantizer::disabled();
        let original = vec![0.1, 0.5, 0.9];
        let processed = quantizer.process(&original);

        // Should be identical when disabled
        assert_eq!(original, processed);
    }

    #[test]
    fn test_quantized_embedding_byte_size() {
        let emb = QuantizedEmbedding::new(vec![0u8; 384], 0.0, 1.0);
        assert_eq!(emb.byte_size(), 384 + 8); // data + min/max
    }

    #[test]
    fn test_spearman_preserved() {
        // Simulate similarity ranking preservation
        let embeddings: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..384)
                    .map(|j| ((i * 13 + j * 7) as f32 / 1000.0).sin())
                    .collect()
            })
            .collect();

        // Compute original cosine similarities
        let mut orig_sims = Vec::new();
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                orig_sims.push(cosine_similarity(&embeddings[i], &embeddings[j]));
            }
        }

        // Quantize and compute similarities
        let quantizer = EightBitQuantizer::new();
        let processed: Vec<Vec<f32>> = embeddings.iter().map(|e| quantizer.process(e)).collect();

        let mut quant_sims = Vec::new();
        for i in 0..processed.len() {
            for j in (i + 1)..processed.len() {
                quant_sims.push(cosine_similarity(&processed[i], &processed[j]));
            }
        }

        // Compute Spearman correlation (simplified - check ranking)
        let spearman = compute_spearman(&orig_sims, &quant_sims);
        assert!(
            spearman > 0.999,
            "8-bit quantization should preserve ranking, got {}",
            spearman
        );
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    fn compute_spearman(x: &[f32], y: &[f32]) -> f32 {
        // Simplified Spearman: check if rankings are identical
        let mut x_ranked: Vec<(usize, f32)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        let mut y_ranked: Vec<(usize, f32)> = y.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        x_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        y_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n = x.len() as f32;
        let sum_d2: f32 = x_ranked
            .iter()
            .zip(y_ranked.iter())
            .enumerate()
            .map(|(i, (xr, yr))| {
                let rx = x_ranked.iter().position(|r| r.0 == i).unwrap() as f32;
                let ry = y_ranked.iter().position(|r| r.0 == i).unwrap() as f32;
                (rx - ry).powi(2)
            })
            .sum();

        1.0 - (6.0 * sum_d2) / (n * (n * n - 1.0))
    }
}

