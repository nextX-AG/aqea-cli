//! Random Sampler - Baseline uniform sampling
//!
//! Simple random sampling without replacement.
//! Fast but doesn't guarantee good space coverage.

use super::Sampler;
use rand::prelude::*;

/// Random sampler - uniform sampling without replacement
#[derive(Debug, Clone)]
pub struct RandomSampler {
    seed: u64,
}

impl RandomSampler {
    /// Create a new random sampler with default seed
    pub fn new() -> Self {
        Self { seed: 42 }
    }

    /// Create a random sampler with specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for RandomSampler {
    fn sample(&self, embeddings: &[Vec<f32>], n_samples: usize) -> Vec<usize> {
        let n = embeddings.len();
        let n_samples = n_samples.min(n);

        if n_samples == n {
            return (0..n).collect();
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n_samples);
        indices.sort_unstable();
        indices
    }

    fn name(&self) -> &'static str {
        "random"
    }

    fn description(&self) -> &'static str {
        "Uniform random sampling without replacement"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_embeddings(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(123);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_random_sampler_basic() {
        let sampler = RandomSampler::new();
        let embeddings = generate_embeddings(100, 32);

        let indices = sampler.sample(&embeddings, 20);

        assert_eq!(indices.len(), 20);
        // Check all indices are valid
        for &idx in &indices {
            assert!(idx < 100);
        }
        // Check no duplicates
        let unique: std::collections::HashSet<_> = indices.iter().collect();
        assert_eq!(unique.len(), 20);
    }

    #[test]
    fn test_random_sampler_all() {
        let sampler = RandomSampler::new();
        let embeddings = generate_embeddings(50, 32);

        let indices = sampler.sample(&embeddings, 50);

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_random_sampler_more_than_available() {
        let sampler = RandomSampler::new();
        let embeddings = generate_embeddings(30, 32);

        let indices = sampler.sample(&embeddings, 100);

        assert_eq!(indices.len(), 30);
    }

    #[test]
    fn test_random_sampler_deterministic() {
        let embeddings = generate_embeddings(100, 32);

        let sampler1 = RandomSampler::with_seed(42);
        let sampler2 = RandomSampler::with_seed(42);

        let indices1 = sampler1.sample(&embeddings, 30);
        let indices2 = sampler2.sample(&embeddings, 30);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_random_sampler_different_seeds() {
        let embeddings = generate_embeddings(100, 32);

        let sampler1 = RandomSampler::with_seed(42);
        let sampler2 = RandomSampler::with_seed(123);

        let indices1 = sampler1.sample(&embeddings, 30);
        let indices2 = sampler2.sample(&embeddings, 30);

        assert_ne!(indices1, indices2);
    }

    #[test]
    fn test_random_sampler_name() {
        let sampler = RandomSampler::new();
        assert_eq!(sampler.name(), "random");
    }
}
