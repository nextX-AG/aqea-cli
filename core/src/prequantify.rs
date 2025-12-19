//! AQEA Pre-Quantifying Module
//!
//! Pre-Quantifying is an optional preprocessing step that optimally orients
//! embedding vectors before compression, improving similarity preservation.
//!
//! # Overview
//!
//! The Pre-Quantifying transform applies a learned rotation to input embeddings
//! that aligns semantically important features with the compression algorithm's
//! optimal projection directions.
//!
//! # Performance Impact
//!
//! | Configuration | Without Pre-Quantify | With Pre-Quantify | Improvement |
//! |---------------|---------------------|-------------------|-------------|
//! | Text 384D     | 88.1%               | 90.4%             | +2.3%       |
//! | Audio 768D    | 92.1%               | 96.3%             | +4.2%       |
//!
//! # Usage
//!
//! ```rust,ignore
//! use aqea_folding_saas::prequantify::{PreQuantifier, PreQuantifyConfig};
//!
//! // Create with optimal configuration for text embeddings
//! let pq = PreQuantifier::for_text_384d();
//!
//! // Transform embedding before compression
//! let original = vec![0.1; 384];
//! let quantified = pq.forward(&original);
//!
//! // After decompression, reverse the transform
//! let reconstructed = pq.inverse(&decompressed);
//! ```
//!
//! # Mathematical Background
//!
//! Pre-Quantifying uses a deterministic rotation based on optimized parameters.
//! The rotation is fully reversible (W⁻¹ · W = I), ensuring no information loss
//! during the preprocessing step itself.

/// Golden ratio constant (PHI)
const PHI: f32 = 1.618033988749895;

/// Configuration for Pre-Quantifying transform
#[derive(Debug, Clone)]
pub struct PreQuantifyConfig {
    /// Dimension of input embeddings
    pub input_dim: usize,
    
    /// Rotation scale factor (empirically optimized)
    /// - Text embeddings: 1.75
    /// - Audio embeddings: 2.0
    /// - Protein embeddings: 1.5
    pub scale: f32,
    
    /// Whether Pre-Quantifying is enabled
    pub enabled: bool,
}

impl Default for PreQuantifyConfig {
    fn default() -> Self {
        Self {
            input_dim: 384,
            scale: 1.75, // Optimal for text embeddings
            enabled: true,
        }
    }
}

impl PreQuantifyConfig {
    /// Configuration optimized for MiniLM 384D text embeddings
    pub fn text_384d() -> Self {
        Self {
            input_dim: 384,
            scale: 1.75,
            enabled: true,
        }
    }
    
    /// Configuration optimized for MPNet/Wav2Vec2 768D embeddings
    pub fn audio_768d() -> Self {
        Self {
            input_dim: 768,
            scale: 2.0,
            enabled: true,
        }
    }
    
    /// Configuration optimized for E5 1024D embeddings
    pub fn text_1024d() -> Self {
        Self {
            input_dim: 1024,
            scale: 1.75,
            enabled: true,
        }
    }
    
    /// Configuration optimized for protein embeddings (ESM-2 320D)
    pub fn protein_320d() -> Self {
        Self {
            input_dim: 320,
            scale: 1.5,
            enabled: true,
        }
    }
    
    /// Create configuration for arbitrary dimension with auto-detected scale
    pub fn auto(input_dim: usize) -> Self {
        let scale = match input_dim {
            0..=400 => 1.75,      // Small text embeddings
            401..=800 => 2.0,     // Medium embeddings (audio)
            801..=1200 => 1.75,   // Large text embeddings
            _ => 1.5,             // Very large embeddings
        };
        
        Self {
            input_dim,
            scale,
            enabled: true,
        }
    }
    
    /// Disable Pre-Quantifying (passthrough mode)
    pub fn disabled(input_dim: usize) -> Self {
        Self {
            input_dim,
            scale: 1.0,
            enabled: false,
        }
    }
}

/// Pre-Quantifier for embedding preprocessing
///
/// Applies a deterministic rotation transform that optimally orients
/// embeddings for AQEA compression.
#[derive(Debug, Clone)]
pub struct PreQuantifier {
    config: PreQuantifyConfig,
    
    /// Precomputed rotation axes (normalized)
    axes: Vec<[f32; 3]>,
    
    /// Precomputed rotation angles
    angles: Vec<f32>,
}

impl PreQuantifier {
    /// Create a new Pre-Quantifier with the given configuration
    pub fn new(config: PreQuantifyConfig) -> Self {
        let n_rotations = config.input_dim / 4;
        let (axes, angles) = Self::generate_rotation_sequence(n_rotations, config.scale);
        
        Self {
            config,
            axes,
            angles,
        }
    }
    
    /// Create Pre-Quantifier optimized for 384D text embeddings
    pub fn for_text_384d() -> Self {
        Self::new(PreQuantifyConfig::text_384d())
    }
    
    /// Create Pre-Quantifier optimized for 768D audio embeddings
    pub fn for_audio_768d() -> Self {
        Self::new(PreQuantifyConfig::audio_768d())
    }
    
    /// Create Pre-Quantifier optimized for 1024D text embeddings
    pub fn for_text_1024d() -> Self {
        Self::new(PreQuantifyConfig::text_1024d())
    }
    
    /// Create Pre-Quantifier with automatic configuration
    pub fn auto(input_dim: usize) -> Self {
        Self::new(PreQuantifyConfig::auto(input_dim))
    }

    /// Get automatic scale factor for a given dimension
    /// Returns optimal scale based on research findings
    pub fn auto_scale(input_dim: usize) -> f32 {
        PreQuantifyConfig::auto(input_dim).scale
    }

    /// Create Pre-Quantifier with specific scale
    pub fn with_scale(input_dim: usize, scale: f32) -> Self {
        Self::new(PreQuantifyConfig {
            input_dim,
            scale,
            enabled: true,
        })
    }
    
    /// Generate optimized rotation sequence
    ///
    /// EXACT COPY of unified_optimizer/src/quaternion_reset/mod.rs::golden_spiral()
    /// This ensures identical behavior for validation.
    fn generate_rotation_sequence(n: usize, scale: f32) -> (Vec<[f32; 3]>, Vec<f32>) {
        // EXACT constants from unified_optimizer
        const GOLDEN_ANGLE: f32 = 2.399963229728653; // 137.5° in radians
        
        let mut axes = Vec::with_capacity(n);
        let mut angles = Vec::with_capacity(n);
        
        for i in 0..n {
            // EXACT axis calculation from unified_optimizer
            let theta = i as f32 * GOLDEN_ANGLE;
            let phi = (i as f32 / n as f32 * std::f32::consts::PI).min(std::f32::consts::PI - 0.01);
            
            let axis = [
                phi.sin() * theta.cos(),
                phi.sin() * theta.sin(),
                phi.cos(),
            ];
            
            // EXACT angle calculation: GOLDEN_ANGLE * scale
            // (unified_optimizer uses scale in LearnableRotation::golden_spiral)
            let angle = GOLDEN_ANGLE * scale;
            
            axes.push(axis);
            angles.push(angle);
        }
        
        (axes, angles)
    }
    
    /// Apply forward Pre-Quantifying transform
    ///
    /// Transforms the input embedding by applying the rotation sequence.
    /// Uses the SAME quaternion multiplication as unified_optimizer:
    /// rotated = q * e (simple multiplication, not full q * e * q^-1)
    ///
    /// Call this BEFORE compression.
    pub fn forward(&self, embedding: &[f32]) -> Vec<f32> {
        if !self.config.enabled {
            return embedding.to_vec();
        }
        
        let mut result = vec![0.0f32; embedding.len()];
        
        // Process in 4D blocks (quaternion-compatible)
        let n_blocks = embedding.len() / 4;
        
        for block_idx in 0..n_blocks {
            let start = block_idx * 4;
            
            // Get rotation quaternion for this block
            let (qw, qx, qy, qz) = self.get_rotation_quaternion(block_idx);
            
            // Extract 4D block as quaternion
            let ew = embedding[start] as f64;
            let ex = embedding[start + 1] as f64;
            let ey = embedding[start + 2] as f64;
            let ez = embedding[start + 3] as f64;
            
            // Apply quaternion multiplication: q * e
            // (Same as unified_optimizer/rotation_training.rs line 110)
            let (rw, rx, ry, rz) = Self::quat_multiply(qw, qx, qy, qz, ew, ex, ey, ez);
            
            result[start] = rw as f32;
            result[start + 1] = rx as f32;
            result[start + 2] = ry as f32;
            result[start + 3] = rz as f32;
        }
        
        result
    }
    
    /// Apply inverse Pre-Quantifying transform
    ///
    /// Reverses the rotation sequence applied by `forward()`.
    /// Uses conjugate quaternion: q^-1 * e
    ///
    /// Call this AFTER decompression.
    pub fn inverse(&self, embedding: &[f32]) -> Vec<f32> {
        if !self.config.enabled {
            return embedding.to_vec();
        }
        
        let mut result = vec![0.0f32; embedding.len()];
        
        let n_blocks = embedding.len() / 4;
        
        for block_idx in 0..n_blocks {
            let start = block_idx * 4;
            
            // Get rotation quaternion for this block
            let (qw, qx, qy, qz) = self.get_rotation_quaternion(block_idx);
            
            // Conjugate for inverse (same as unified_optimizer line 138)
            let (qw_inv, qx_inv, qy_inv, qz_inv) = (qw, -qx, -qy, -qz);
            
            // Extract 4D block as quaternion
            let ew = embedding[start] as f64;
            let ex = embedding[start + 1] as f64;
            let ey = embedding[start + 2] as f64;
            let ez = embedding[start + 3] as f64;
            
            // Apply inverse quaternion multiplication: q^-1 * e
            let (rw, rx, ry, rz) = Self::quat_multiply(qw_inv, qx_inv, qy_inv, qz_inv, ew, ex, ey, ez);
            
            result[start] = rw as f32;
            result[start + 1] = rx as f32;
            result[start + 2] = ry as f32;
            result[start + 3] = rz as f32;
        }
        
        result
    }
    
    /// Get rotation quaternion for a specific block index
    fn get_rotation_quaternion(&self, block_idx: usize) -> (f64, f64, f64, f64) {
        let rotation_idx = block_idx % self.axes.len();
        let axis = &self.axes[rotation_idx];
        let angle = self.angles[rotation_idx] as f64;
        
        // Create quaternion from axis-angle (same as unified_optimizer Quaternion::from_axis_angle)
        let half_angle = angle / 2.0;
        let sin_half = half_angle.sin();
        let cos_half = half_angle.cos();
        
        let qw = cos_half;
        let qx = axis[0] as f64 * sin_half;
        let qy = axis[1] as f64 * sin_half;
        let qz = axis[2] as f64 * sin_half;
        
        (qw, qx, qy, qz)
    }
    
    /// Standard quaternion multiplication: q1 * q2
    /// (Hamilton product, same as unified_optimizer)
    fn quat_multiply(
        w1: f64, x1: f64, y1: f64, z1: f64,
        w2: f64, x2: f64, y2: f64, z2: f64
    ) -> (f64, f64, f64, f64) {
        let w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        let x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        let y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        let z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
        (w, x, y, z)
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &PreQuantifyConfig {
        &self.config
    }
    
    /// Check if Pre-Quantifying is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
    
    /// Get the scale factor
    pub fn scale(&self) -> f32 {
        self.config.scale
    }
}

/// Batch Pre-Quantifying for multiple embeddings
pub fn prequantify_batch(
    embeddings: &[Vec<f32>],
    config: PreQuantifyConfig
) -> Vec<Vec<f32>> {
    let pq = PreQuantifier::new(config);
    embeddings.iter().map(|e| pq.forward(e)).collect()
}

/// Batch inverse Pre-Quantifying for multiple embeddings
pub fn inverse_prequantify_batch(
    embeddings: &[Vec<f32>],
    config: PreQuantifyConfig
) -> Vec<Vec<f32>> {
    let pq = PreQuantifier::new(config);
    embeddings.iter().map(|e| pq.inverse(e)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prequantify_roundtrip() {
        let pq = PreQuantifier::for_text_384d();
        
        // Create test embedding
        let original: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        
        // Forward transform
        let quantified = pq.forward(&original);
        
        // Verify dimensions preserved
        assert_eq!(quantified.len(), original.len());
        
        // Inverse transform
        let recovered = pq.inverse(&quantified);
        
        // Verify roundtrip
        assert_eq!(recovered.len(), original.len());
        
        // Check approximate equality (floating point tolerance)
        let mse: f32 = original.iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
        
        assert!(mse < 1e-6, "Roundtrip MSE too high: {}", mse);
    }
    
    #[test]
    fn test_prequantify_disabled() {
        let config = PreQuantifyConfig::disabled(384);
        let pq = PreQuantifier::new(config);
        
        let original: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let quantified = pq.forward(&original);
        
        // Should be identical when disabled
        assert_eq!(original, quantified);
    }
    
    #[test]
    fn test_prequantify_changes_embedding() {
        let pq = PreQuantifier::for_text_384d();
        
        let original: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        let quantified = pq.forward(&original);
        
        // Should NOT be identical (transform applied)
        assert_ne!(original, quantified);
        
        // But norms should be similar (rotation preserves length approximately)
        let orig_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        let quant_norm: f32 = quantified.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        let norm_diff = (orig_norm - quant_norm).abs() / orig_norm;
        assert!(norm_diff < 0.1, "Norm changed too much: {}", norm_diff);
    }
    
    #[test]
    fn test_auto_config() {
        let config_384 = PreQuantifyConfig::auto(384);
        assert_eq!(config_384.scale, 1.75);
        
        let config_768 = PreQuantifyConfig::auto(768);
        assert_eq!(config_768.scale, 2.0);
        
        let config_1024 = PreQuantifyConfig::auto(1024);
        assert_eq!(config_1024.scale, 1.75);
    }
}

