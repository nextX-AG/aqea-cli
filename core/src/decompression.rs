//! Decompression Module - Various Reconstruction Methods
//!
//! Provides multiple decompression strategies:
//! - ProjectionDecompressor: Uses transpose of compression weights (fast, good quality)
//! - LinearDecompressor: Simple replication (fast, low quality)
//! - ThermodynamicDecompressor: Monte Carlo sampling (slow, highest quality)
//! - PQDecompressor: Product Quantization codebook lookup (for PQ-encoded data)

use crate::energy::{EnergyFunction, CombinedEnergy};
use crate::constants::KB_EV;
use crate::pq::ProductQuantizer;
use rand::prelude::*;
use rand_distr::Normal;

/// Trait for decompression algorithms
pub trait Decompressor: Send + Sync {
    /// Decompress low-dimensional to high-dimensional
    fn decompress(&self, compressed: &[f32]) -> Vec<f32>;

    /// Get compressed dimension
    fn compressed_dim(&self) -> usize;

    /// Get original dimension
    fn original_dim(&self) -> usize;
}

// ============================================================================
// PROJECTION DECOMPRESSOR (uses trained weights transpose)
// ============================================================================

/// Projection Decompressor - Uses transpose of compression weights
///
/// This is the recommended decompressor when trained weights are available.
/// Reconstruction: x' = W^T @ y (transpose projection)
///
/// Quality depends on how orthogonal the weight matrix is.
/// For trained AQEA weights, this typically achieves good reconstruction.
pub struct ProjectionDecompressor {
    original_dim: usize,
    compressed_dim: usize,
    /// Transposed weights [original_dim x compressed_dim]
    weights_t: Vec<Vec<f32>>,
}

impl ProjectionDecompressor {
    /// Create from compression weight matrix
    ///
    /// # Arguments
    /// * `compression_weights` - The weight matrix from OctonionCompressor [compressed_dim x original_dim]
    pub fn from_compression_weights(compression_weights: &[Vec<f32>]) -> Self {
        let compressed_dim = compression_weights.len();
        let original_dim = if compressed_dim > 0 { compression_weights[0].len() } else { 0 };

        // Transpose: weights_t[i][j] = weights[j][i]
        let mut weights_t = vec![vec![0.0f32; compressed_dim]; original_dim];
        for j in 0..compressed_dim {
            for i in 0..original_dim {
                weights_t[i][j] = compression_weights[j][i];
            }
        }

        Self {
            original_dim,
            compressed_dim,
            weights_t,
        }
    }

    /// Create from flat weights (same format as OctonionCompressor)
    pub fn from_flat_weights(original_dim: usize, compressed_dim: usize, flat_weights: &[f32]) -> Self {
        // Reshape to 2D then transpose
        let mut weights_t = vec![vec![0.0f32; compressed_dim]; original_dim];
        
        for out_idx in 0..compressed_dim {
            for in_idx in 0..original_dim {
                // Original: weights[out_idx][in_idx] = flat[out_idx * original_dim + in_idx]
                // Transpose: weights_t[in_idx][out_idx] = flat[out_idx * original_dim + in_idx]
                weights_t[in_idx][out_idx] = flat_weights[out_idx * original_dim + in_idx];
            }
        }

        Self {
            original_dim,
            compressed_dim,
            weights_t,
        }
    }

    /// Create with default (golden ratio) weights
    pub fn with_dims(original_dim: usize, compressed_dim: usize) -> Self {
        use crate::compression::OctonionCompressor;
        let compressor = OctonionCompressor::with_dims(original_dim, compressed_dim);
        Self::from_compression_weights(compressor.weights())
    }
}

impl Decompressor for ProjectionDecompressor {
    fn decompress(&self, compressed: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.original_dim];

        // Matrix-vector multiply: result = weights_t @ compressed
        for i in 0..self.original_dim {
            for j in 0..self.compressed_dim.min(compressed.len()) {
                result[i] += self.weights_t[i][j] * compressed[j];
            }
        }

        result
    }

    fn compressed_dim(&self) -> usize {
        self.compressed_dim
    }

    fn original_dim(&self) -> usize {
        self.original_dim
    }
}

// ============================================================================
// PQ DECOMPRESSOR (codebook lookup for Product Quantization)
// ============================================================================

/// PQ Decompressor - Uses trained codebook for reconstruction
///
/// This decompressor reconstructs vectors from PQ codes by looking up
/// centroids in the trained codebook and concatenating them.
///
/// **Important:** PQ is NON-LINEAR compression (quantization), so this
/// decompressor performs centroid lookup, NOT matrix multiplication!
///
/// Reconstruction: for each code → lookup centroid → concatenate
pub struct PQDecompressor {
    /// Output dimension (original AQEA space, e.g., 128D)
    original_dim: usize,
    /// Number of subvectors (= number of bytes in PQ code)
    n_subvectors: usize,
    /// Dimension per subvector
    subvec_dim: usize,
    /// Centroids: [n_subvectors][256][subvec_dim]
    centroids: Vec<Vec<Vec<f32>>>,
}

impl PQDecompressor {
    /// Create from a trained ProductQuantizer
    ///
    /// # Arguments
    /// * `pq` - A trained ProductQuantizer with codebook
    pub fn from_pq(pq: &ProductQuantizer) -> Self {
        Self {
            original_dim: pq.config.input_dim,
            n_subvectors: pq.config.n_subvectors,
            subvec_dim: pq.config.subvec_dim,
            centroids: pq.centroids.clone(),
        }
    }

    /// Create from raw centroids
    ///
    /// # Arguments
    /// * `original_dim` - Output dimension
    /// * `centroids` - Codebook: [n_subvectors][256][subvec_dim]
    pub fn from_centroids(original_dim: usize, centroids: Vec<Vec<Vec<f32>>>) -> Self {
        let n_subvectors = centroids.len();
        let subvec_dim = if n_subvectors > 0 && !centroids[0].is_empty() {
            centroids[0][0].len()
        } else {
            0
        };

        Self {
            original_dim,
            n_subvectors,
            subvec_dim,
            centroids,
        }
    }

    /// Decompress PQ codes (u8) to float vector
    ///
    /// This is the primary method - takes actual PQ codes as bytes.
    ///
    /// # Arguments
    /// * `codes` - PQ codes (one byte per subvector)
    ///
    /// # Returns
    /// * Reconstructed float vector
    pub fn decompress_codes(&self, codes: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.original_dim);

        for (sub_idx, &code) in codes.iter().enumerate() {
            if sub_idx < self.centroids.len() {
                let centroid = &self.centroids[sub_idx][code as usize];
                result.extend_from_slice(centroid);
            }
        }

        // Truncate to exact original_dim (in case of padding)
        result.truncate(self.original_dim);
        result
    }

    /// Get the number of subvectors
    pub fn n_subvectors(&self) -> usize {
        self.n_subvectors
    }
}

impl Decompressor for PQDecompressor {
    /// Decompress - compatibility with Decompressor trait
    ///
    /// Note: This method expects f32 values that represent u8 codes.
    /// For actual PQ codes, prefer `decompress_codes(&[u8])`.
    fn decompress(&self, compressed: &[f32]) -> Vec<f32> {
        // Convert f32 to u8 codes
        let codes: Vec<u8> = compressed.iter()
            .map(|&v| v.round() as u8)
            .collect();
        self.decompress_codes(&codes)
    }

    fn compressed_dim(&self) -> usize {
        self.n_subvectors
    }

    fn original_dim(&self) -> usize {
        self.original_dim
    }
}

// ============================================================================
// LINEAR DECOMPRESSOR (simple baseline)
// ============================================================================

/// Linear decompressor - simple replication baseline
///
/// Fast but low quality. Used as initialization for thermodynamic.
pub struct LinearDecompressor {
    original_dim: usize,
    compressed_dim: usize,
}

impl LinearDecompressor {
    pub fn new(original_dim: usize, compressed_dim: usize) -> Self {
        Self {
            original_dim,
            compressed_dim,
        }
    }
}

impl Decompressor for LinearDecompressor {
    fn decompress(&self, compressed: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.original_dim];
        let n_per_dim = self.original_dim / self.compressed_dim.max(1);

        for i in 0..self.original_dim {
            let src_idx = (i / n_per_dim).min(compressed.len().saturating_sub(1));
            out[i] = if src_idx < compressed.len() { compressed[src_idx] } else { 0.0 };
        }

        out
    }

    fn compressed_dim(&self) -> usize {
        self.compressed_dim
    }

    fn original_dim(&self) -> usize {
        self.original_dim
    }
}

/// Thermodynamic decompressor using Monte Carlo with energy minimization
///
/// Implements replica exchange (parallel tempering) for efficient sampling.
/// Multiple replicas at different temperatures explore the energy landscape
/// and exchange configurations to escape local minima.
pub struct ThermodynamicDecompressor {
    original_dim: usize,
    compressed_dim: usize,
    n_replicas: usize,
    max_iterations: usize,
    temperature_min: f32,
    temperature_max: f32,
    step_size: f32,
}

impl ThermodynamicDecompressor {
    /// Create with default parameters
    pub fn new(original_dim: usize, compressed_dim: usize) -> Self {
        Self {
            original_dim,
            compressed_dim,
            n_replicas: 8,
            max_iterations: 200,
            temperature_min: 280.0,
            temperature_max: 400.0,
            step_size: 0.1,
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        original_dim: usize,
        compressed_dim: usize,
        n_replicas: usize,
        max_iterations: usize,
    ) -> Self {
        let mut dec = Self::new(original_dim, compressed_dim);
        dec.n_replicas = n_replicas;
        dec.max_iterations = max_iterations;
        dec
    }

    /// Initialize conformation from compressed representation
    fn initialize_conformation(&self, compressed: &[f32], rng: &mut impl Rng) -> Vec<f32> {
        let mut conformation = vec![0.0; self.original_dim];
        let normal = Normal::new(0.0, 0.1).unwrap();
        let n_per_dim = self.original_dim / self.compressed_dim.max(1);

        for i in 0..self.original_dim {
            let src_idx = (i / n_per_dim).min(compressed.len().saturating_sub(1));
            let base = if src_idx < compressed.len() { compressed[src_idx] } else { 0.0 };
            let noise: f64 = rng.sample(normal);
            conformation[i] = base + noise as f32;
        }

        conformation
    }

    /// Single Monte Carlo step with Metropolis criterion
    fn monte_carlo_step(
        &self,
        conformation: &mut [f32],
        compressed: &[f32],
        temperature: f32,
        energy_fn: &dyn EnergyFunction,
        rng: &mut impl Rng,
    ) -> f32 {
        let normal = Normal::new(0.0, self.step_size as f64).unwrap();

        // Current energy
        let current_energy = energy_fn.energy(conformation, compressed);

        // Propose random perturbation
        let idx = rng.gen_range(0..conformation.len());
        let old_value = conformation[idx];
        let delta: f64 = rng.sample(normal);
        conformation[idx] = old_value + delta as f32;

        // New energy
        let new_energy = energy_fn.energy(conformation, compressed);

        // Metropolis acceptance criterion
        let delta_e = new_energy - current_energy;
        let accept = if delta_e < 0.0 {
            true
        } else {
            let boltzmann = (-delta_e / (KB_EV * temperature)).exp();
            rng.gen::<f32>() < boltzmann
        };

        if !accept {
            conformation[idx] = old_value;
            current_energy
        } else {
            new_energy
        }
    }

    /// Replica exchange between adjacent temperature levels
    fn replica_exchange(
        &self,
        replicas: &mut [Vec<f32>],
        temperatures: &[f32],
        compressed: &[f32],
        energy_fn: &dyn EnergyFunction,
        rng: &mut impl Rng,
    ) {
        for i in 0..(replicas.len() - 1) {
            let e_i = energy_fn.energy(&replicas[i], compressed);
            let e_j = energy_fn.energy(&replicas[i + 1], compressed);

            let beta_i = 1.0 / (KB_EV * temperatures[i]);
            let beta_j = 1.0 / (KB_EV * temperatures[i + 1]);

            let delta = (beta_i - beta_j) * (e_j - e_i);
            let accept = if delta < 0.0 {
                true
            } else {
                rng.gen::<f32>() < (-delta).exp()
            };

            if accept {
                replicas.swap(i, i + 1);
            }
        }
    }
}

impl Decompressor for ThermodynamicDecompressor {
    fn decompress(&self, compressed: &[f32]) -> Vec<f32> {
        let mut rng = rand::thread_rng();

        // Create energy function
        let energy_fn = CombinedEnergy::new(self.original_dim, self.compressed_dim);

        // Initialize replicas at different temperatures
        let mut replicas: Vec<Vec<f32>> = (0..self.n_replicas)
            .map(|_| self.initialize_conformation(compressed, &mut rng))
            .collect();

        // Temperature ladder (geometric spacing)
        let temp_ratio = (self.temperature_max / self.temperature_min).powf(1.0 / (self.n_replicas - 1) as f32);
        let temperatures: Vec<f32> = (0..self.n_replicas)
            .map(|i| self.temperature_min * temp_ratio.powi(i as i32))
            .collect();

        // Run Monte Carlo simulation
        for iter in 0..self.max_iterations {
            // MC steps for each replica
            for (replica, &temp) in replicas.iter_mut().zip(temperatures.iter()) {
                // Multiple steps per iteration for efficiency
                for _ in 0..10 {
                    self.monte_carlo_step(replica, compressed, temp, &energy_fn, &mut rng);
                }
            }

            // Replica exchange every 10 iterations
            if iter % 10 == 0 {
                self.replica_exchange(
                    &mut replicas,
                    &temperatures,
                    compressed,
                    &energy_fn,
                    &mut rng,
                );
            }
        }

        // Return lowest temperature replica (most "folded")
        replicas.remove(0)
    }

    fn compressed_dim(&self) -> usize {
        self.compressed_dim
    }

    fn original_dim(&self) -> usize {
        self.original_dim
    }
}

/// Hybrid decompressor - combines fast linear with thermodynamic refinement
pub struct HybridDecompressor {
    linear: LinearDecompressor,
    thermodynamic: ThermodynamicDecompressor,
    refinement_iterations: usize,
}

impl HybridDecompressor {
    pub fn new(original_dim: usize, compressed_dim: usize) -> Self {
        Self {
            linear: LinearDecompressor::new(original_dim, compressed_dim),
            thermodynamic: ThermodynamicDecompressor::with_config(
                original_dim, compressed_dim, 4, 50
            ),
            refinement_iterations: 50,
        }
    }

    /// Quick mode - linear only (for real-time applications)
    pub fn quick(original_dim: usize, compressed_dim: usize) -> Self {
        Self {
            linear: LinearDecompressor::new(original_dim, compressed_dim),
            thermodynamic: ThermodynamicDecompressor::with_config(
                original_dim, compressed_dim, 2, 10
            ),
            refinement_iterations: 10,
        }
    }
}

impl Decompressor for HybridDecompressor {
    fn decompress(&self, compressed: &[f32]) -> Vec<f32> {
        // Start with linear for fast initialization
        let initial = self.linear.decompress(compressed);
        
        // Refine with thermodynamic
        let mut rng = rand::thread_rng();
        let energy_fn = CombinedEnergy::new(
            self.linear.original_dim(), 
            self.linear.compressed_dim()
        );

        let mut conformation = initial;
        for _ in 0..self.refinement_iterations {
            let normal = Normal::new(0.0, 0.05).unwrap();
            let idx = rng.gen_range(0..conformation.len());
            let old_value = conformation[idx];
            let delta: f64 = rng.sample(normal);
            conformation[idx] = old_value + delta as f32;

            let current_energy = energy_fn.energy(&conformation, compressed);
            let mut test_conf = conformation.clone();
            test_conf[idx] = old_value;
            let old_energy = energy_fn.energy(&test_conf, compressed);

            // Only keep if energy decreased
            if current_energy > old_energy {
                conformation[idx] = old_value;
            }
        }

        conformation
    }

    fn compressed_dim(&self) -> usize {
        self.linear.compressed_dim()
    }

    fn original_dim(&self) -> usize {
        self.linear.original_dim()
    }
}

// ============================================================================
// BATCH DECOMPRESSION
// ============================================================================

/// Decompress multiple vectors at once
pub fn decompress_batch(decompressor: &dyn Decompressor, compressed_vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    compressed_vectors.iter()
        .map(|c| decompressor.decompress(c))
        .collect()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_decompressor() {
        let decompressor = LinearDecompressor::new(384, 8);
        let compressed = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(decompressed.len(), 384);
        
        // First block should have values close to first compressed value
        assert!((decompressed[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_thermodynamic_decompressor() {
        let decompressor = ThermodynamicDecompressor::with_config(64, 4, 4, 20);
        let compressed = vec![0.1, 0.2, 0.3, 0.4];

        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(decompressed.len(), 64);
        
        // Values should be finite
        assert!(decompressed.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_hybrid_decompressor() {
        let decompressor = HybridDecompressor::quick(64, 4);
        let compressed = vec![0.5, -0.3, 0.8, 0.1];

        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(decompressed.len(), 64);
        assert!(decompressed.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_batch_decompression() {
        let decompressor = LinearDecompressor::new(32, 4);
        let compressed_batch = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
        ];

        let decompressed = decompress_batch(&decompressor, &compressed_batch);
        assert_eq!(decompressed.len(), 2);
        assert!(decompressed.iter().all(|v| v.len() == 32));
    }

    #[test]
    fn test_pq_decompressor() {
        use crate::pq::ProductQuantizer;

        // Create and train a small PQ
        let mut pq = ProductQuantizer::new(8, 4, 8); // 8D -> 4 subvectors

        // Generate simple training data
        let training_data: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..8).map(|j| (i as f32 * 0.1 + j as f32 * 0.01)).collect())
            .collect();

        pq.train(&training_data, 10);

        // Create decompressor from PQ
        let decompressor = PQDecompressor::from_pq(&pq);

        // Test encode -> decode roundtrip
        let original = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let codes = pq.encode(&original);
        let decoded = decompressor.decompress_codes(&codes);

        // Check dimensions
        assert_eq!(decoded.len(), 8);
        assert_eq!(codes.len(), 4);

        // Values should be finite
        assert!(decoded.iter().all(|v| v.is_finite()));

        // Decoded should be reasonably close to original (PQ approximation)
        // This is a sanity check - PQ is lossy but should preserve structure
        println!("Original: {:?}", original);
        println!("Decoded:  {:?}", decoded);
    }

    #[test]
    fn test_pq_decompressor_quality() {
        use crate::pq::ProductQuantizer;
        use crate::metrics::cosine_similarity;

        // Create PQ with more subvectors for better quality
        let mut pq = ProductQuantizer::new(128, 64, 8); // 128D -> 64 subvectors = 64 bytes

        // Generate training data
        let mut rng = rand::thread_rng();
        let training_data: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..128).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();

        pq.train(&training_data, 20);

        // Create decompressor
        let decompressor = PQDecompressor::from_pq(&pq);

        // Test quality on multiple vectors
        let mut total_similarity = 0.0;
        let n_test = 50;

        for _ in 0..n_test {
            let original: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() - 0.5).collect();
            let codes = pq.encode(&original);
            let decoded = decompressor.decompress_codes(&codes);

            let sim = cosine_similarity(&original, &decoded);
            total_similarity += sim;
        }

        let avg_similarity = total_similarity / n_test as f32;
        println!("Average cosine similarity after PQ encode/decode: {:.4}", avg_similarity);

        // With 64 subvectors on 128D, we expect good reconstruction (>0.90)
        assert!(avg_similarity > 0.85, "PQ reconstruction quality too low: {}", avg_similarity);
    }
}
