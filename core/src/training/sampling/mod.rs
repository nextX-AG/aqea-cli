//! Stratified Sampling Module for AQEA Training
//!
//! Provides intelligent sampling strategies to ensure training data
//! covers the embedding space uniformly, reducing generalization gap.
//!
//! # Available Samplers
//!
//! - [`RandomSampler`] - Uniform random sampling (baseline)
//! - [`KMeansSampler`] - K-Means++ based cluster sampling
//! - [`TsneGridSampler`] - t-SNE 2D grid sampling (best coverage)
//!
//! # Example
//!
//! ```rust,ignore
//! use aqea_core::training::sampling::{Sampler, KMeansSampler};
//!
//! let sampler = KMeansSampler::new(100); // 100 clusters
//! let indices = sampler.sample(&embeddings, 1000);
//! ```

mod random;
mod kmeans;
mod tsne_grid;

pub use random::RandomSampler;
pub use kmeans::KMeansSampler;
pub use tsne_grid::TsneGridSampler;

/// Trait for sampling strategies
///
/// Samplers select a subset of indices from embeddings that
/// best represent the full dataset for training.
pub trait Sampler: Send + Sync {
    /// Sample n indices from the embedding dataset
    ///
    /// # Arguments
    /// * `embeddings` - The full embedding dataset (N x D)
    /// * `n_samples` - Number of samples to select
    ///
    /// # Returns
    /// Vector of indices into the embeddings array
    fn sample(&self, embeddings: &[Vec<f32>], n_samples: usize) -> Vec<usize>;

    /// Name of the sampling strategy
    fn name(&self) -> &'static str;

    /// Description of the sampling strategy
    fn description(&self) -> &'static str {
        "Sampling strategy"
    }
}

/// Sampling strategy enum for configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SamplingStrategy {
    /// Uniform random sampling
    Random,
    /// K-Means++ cluster-based sampling
    #[serde(rename = "kmeans")]
    KMeans,
    /// t-SNE grid-based sampling
    #[serde(rename = "tsne-grid")]
    TsneGrid,
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Random
    }
}

impl std::fmt::Display for SamplingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Random => write!(f, "random"),
            Self::KMeans => write!(f, "kmeans"),
            Self::TsneGrid => write!(f, "tsne-grid"),
        }
    }
}

impl std::str::FromStr for SamplingStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "random" => Ok(Self::Random),
            "kmeans" | "kmeans++" | "k-means" => Ok(Self::KMeans),
            "tsne" | "tsne-grid" | "tsne_grid" | "grid" => Ok(Self::TsneGrid),
            _ => Err(format!("Unknown sampling strategy: {}. Valid: random, kmeans, tsne-grid", s)),
        }
    }
}

/// Create a sampler from a strategy
pub fn create_sampler(strategy: SamplingStrategy, seed: Option<u64>) -> Box<dyn Sampler> {
    let seed = seed.unwrap_or(42);
    match strategy {
        SamplingStrategy::Random => Box::new(RandomSampler::with_seed(seed)),
        SamplingStrategy::KMeans => Box::new(KMeansSampler::with_seed(seed)),
        SamplingStrategy::TsneGrid => Box::new(TsneGridSampler::with_seed(seed)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_from_str() {
        assert_eq!("random".parse::<SamplingStrategy>().unwrap(), SamplingStrategy::Random);
        assert_eq!("kmeans".parse::<SamplingStrategy>().unwrap(), SamplingStrategy::KMeans);
        assert_eq!("kmeans++".parse::<SamplingStrategy>().unwrap(), SamplingStrategy::KMeans);
        assert_eq!("tsne-grid".parse::<SamplingStrategy>().unwrap(), SamplingStrategy::TsneGrid);
        assert!("invalid".parse::<SamplingStrategy>().is_err());
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(SamplingStrategy::Random.to_string(), "random");
        assert_eq!(SamplingStrategy::KMeans.to_string(), "kmeans");
        assert_eq!(SamplingStrategy::TsneGrid.to_string(), "tsne-grid");
    }
}
