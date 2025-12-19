//! Energy Functions (Priors) for Thermodynamic Decompression
//!
//! Implements energy-based priors for Monte Carlo sampling.
//! These are used in the thermodynamic decompressor to guide
//! the reconstruction of high-dimensional vectors.
//!
//! Key priors:
//! - OctonionPrior: Block-based consistency
//! - LocalConsistencyPrior: Neighborhood smoothness

// Note: Constants imported here for future use in energy calculations

/// Trait for all energy functions
pub trait EnergyFunction: Send + Sync {
    /// Calculate total energy for a given conformation
    fn energy(&self, conformation: &[f32], compressed: &[f32]) -> f32;

    /// Get number of weight parameters
    fn param_count(&self) -> usize;

    /// Get default weights
    fn default_weights(&self) -> Vec<f32>;
}

/// Octonion-aware energy prior
///
/// Enforces consistency within octonion-like blocks.
/// Energy is lower when reconstructed values are consistent
/// with their corresponding compressed values.
pub struct OctonionPrior {
    original_dim: usize,
    compressed_dim: usize,
    block_size: usize,
    lambda_reconstruction: f32,
    lambda_smoothness: f32,
}

impl OctonionPrior {
    pub fn new(original_dim: usize, compressed_dim: usize) -> Self {
        Self {
            original_dim,
            compressed_dim,
            block_size: 8,
            lambda_reconstruction: 0.5,
            lambda_smoothness: 0.3,
        }
    }

    /// Reconstruction energy - how well does conformation match compressed?
    fn reconstruction_energy(&self, conformation: &[f32], compressed: &[f32]) -> f32 {
        let mut energy = 0.0;
        let n_blocks = (self.original_dim + self.block_size - 1) / self.block_size;
        let blocks_per_output = n_blocks / self.compressed_dim.max(1);

        for out_idx in 0..self.compressed_dim {
            let target = if out_idx < compressed.len() { compressed[out_idx] } else { 0.0 };

            // Average of corresponding block values
            let start_block = out_idx * blocks_per_output;
            let end_block = ((out_idx + 1) * blocks_per_output).min(n_blocks);

            let mut block_sum = 0.0f32;
            let mut block_count = 0;

            for block_idx in start_block..end_block {
                let block_start = block_idx * self.block_size;
                let block_end = (block_start + self.block_size).min(self.original_dim);

                for i in block_start..block_end {
                    if i < conformation.len() {
                        block_sum += conformation[i];
                        block_count += 1;
                    }
                }
            }

            let block_avg = if block_count > 0 { block_sum / block_count as f32 } else { 0.0 };
            let deviation = (block_avg - target).abs();
            energy += deviation * deviation;
        }

        energy * self.lambda_reconstruction
    }

    /// Smoothness energy - adjacent values should be similar
    fn smoothness_energy(&self, conformation: &[f32]) -> f32 {
        if conformation.len() < 2 {
            return 0.0;
        }

        let mut energy = 0.0;
        for i in 1..conformation.len() {
            let diff = conformation[i] - conformation[i - 1];
            energy += diff * diff;
        }

        energy * self.lambda_smoothness / conformation.len() as f32
    }

    /// Block consistency - values within same block should be similar
    fn block_consistency_energy(&self, conformation: &[f32]) -> f32 {
        let mut energy = 0.0;
        let n_blocks = (conformation.len() + self.block_size - 1) / self.block_size;

        for block_idx in 0..n_blocks {
            let block_start = block_idx * self.block_size;
            let block_end = (block_start + self.block_size).min(conformation.len());

            // Compute block mean
            let block_sum: f32 = conformation[block_start..block_end].iter().sum();
            let block_len = (block_end - block_start) as f32;
            let block_mean = block_sum / block_len;

            // Variance within block
            for i in block_start..block_end {
                let diff = conformation[i] - block_mean;
                energy += diff * diff;
            }
        }

        energy * 0.1 / conformation.len() as f32
    }
}

impl EnergyFunction for OctonionPrior {
    fn energy(&self, conformation: &[f32], compressed: &[f32]) -> f32 {
        let recon_e = self.reconstruction_energy(conformation, compressed);
        let smooth_e = self.smoothness_energy(conformation);
        let block_e = self.block_consistency_energy(conformation);

        recon_e + smooth_e + block_e
    }

    fn param_count(&self) -> usize {
        self.compressed_dim + 2  // One per output dim + 2 lambdas
    }

    fn default_weights(&self) -> Vec<f32> {
        vec![0.1; self.param_count()]
    }
}

/// Local Consistency Prior
///
/// Based on Bernshteyn's LOCAL theorem: if every point is
/// consistent with its k neighbors, the global structure converges.
pub struct LocalConsistencyPrior {
    dim: usize,
    k_neighbors: usize,
    lambda_local: f32,
    lambda_gradient: f32,
}

impl LocalConsistencyPrior {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            k_neighbors: 3,
            lambda_local: 0.15,
            lambda_gradient: 0.05,
        }
    }

    /// Local neighborhood consistency
    fn local_consistency(&self, conformation: &[f32]) -> f32 {
        let mut energy = 0.0;
        let n = conformation.len();

        for i in 0..n {
            let x_i = conformation[i];
            let mut neighbor_sum = 0.0;

            for delta in 1..=self.k_neighbors {
                let j = (i + delta) % n;
                neighbor_sum += conformation[j];
            }

            let neighbor_mean = neighbor_sum / self.k_neighbors as f32;
            let deviation = (x_i - neighbor_mean).abs();
            energy += deviation;
        }

        energy * self.lambda_local / n as f32
    }

    /// Gradient smoothness (second derivative)
    fn gradient_consistency(&self, conformation: &[f32]) -> f32 {
        if conformation.len() < 3 {
            return 0.0;
        }

        let mut energy = 0.0;
        let n = conformation.len();

        for i in 1..(n - 1) {
            let curvature = conformation[i - 1] - 2.0 * conformation[i] + conformation[i + 1];
            energy += curvature.abs();
        }

        energy * self.lambda_gradient / n as f32
    }
}

impl EnergyFunction for LocalConsistencyPrior {
    fn energy(&self, conformation: &[f32], _compressed: &[f32]) -> f32 {
        let local_e = self.local_consistency(conformation);
        let gradient_e = self.gradient_consistency(conformation);

        local_e + gradient_e
    }

    fn param_count(&self) -> usize {
        self.dim + 2
    }

    fn default_weights(&self) -> Vec<f32> {
        vec![0.1; self.param_count()]
    }
}

/// Combined energy function with multiple priors
pub struct CombinedEnergy {
    octonion: OctonionPrior,
    local: LocalConsistencyPrior,
    weight_octonion: f32,
    weight_local: f32,
}

impl CombinedEnergy {
    pub fn new(original_dim: usize, compressed_dim: usize) -> Self {
        Self {
            octonion: OctonionPrior::new(original_dim, compressed_dim),
            local: LocalConsistencyPrior::new(original_dim),
            weight_octonion: 0.7,
            weight_local: 0.3,
        }
    }
}

impl EnergyFunction for CombinedEnergy {
    fn energy(&self, conformation: &[f32], compressed: &[f32]) -> f32 {
        let octonion_e = self.octonion.energy(conformation, compressed);
        let local_e = self.local.energy(conformation, compressed);

        self.weight_octonion * octonion_e + self.weight_local * local_e
    }

    fn param_count(&self) -> usize {
        self.octonion.param_count() + self.local.param_count()
    }

    fn default_weights(&self) -> Vec<f32> {
        let mut weights = self.octonion.default_weights();
        weights.extend(self.local.default_weights());
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octonion_prior() {
        let prior = OctonionPrior::new(384, 8);
        let conformation: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let compressed = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let energy = prior.energy(&conformation, &compressed);
        assert!(energy >= 0.0);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_local_consistency_prior() {
        let prior = LocalConsistencyPrior::new(384);
        let conformation: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let compressed = vec![0.1, 0.2, 0.3, 0.4];

        let energy = prior.energy(&conformation, &compressed);
        assert!(energy >= 0.0);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_combined_energy() {
        let energy_fn = CombinedEnergy::new(384, 8);
        let conformation: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let compressed = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let energy = energy_fn.energy(&conformation, &compressed);
        assert!(energy >= 0.0);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_param_count() {
        let octonion = OctonionPrior::new(384, 8);
        assert_eq!(octonion.param_count(), 10);  // 8 + 2

        let local = LocalConsistencyPrior::new(384);
        assert_eq!(local.param_count(), 386);  // 384 + 2
    }
}

