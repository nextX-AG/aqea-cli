//! Configuration for Auto-Training
//!
//! Defines all configurable parameters for the auto-training system.

use serde::{Deserialize, Serialize};
use super::sampling::SamplingStrategy;

/// Sample size progression strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SampleProgression {
    /// Fixed progression: 100 -> 200 -> 500 -> 1000 -> ...
    Fixed(Vec<usize>),

    /// Adaptive doubling: start at N, double until stagnation
    Doubling { start: usize, max: usize },
    
    /// Single run with fixed sample size (fast testing mode)
    /// Use absolute number or percentage of available data
    Single(usize),
    
    /// Single run with percentage of available data
    Percent(f32),
}

impl Default for SampleProgression {
    fn default() -> Self {
        // Start with smaller sizes to work with datasets of all sizes
        SampleProgression::Fixed(vec![100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    }
}

impl SampleProgression {
    /// Get the sample sizes to iterate through
    pub fn sizes(&self, max_available: usize) -> Vec<usize> {
        match self {
            SampleProgression::Fixed(sizes) => {
                sizes.iter()
                    .copied()
                    .filter(|&s| s <= max_available)
                    .collect()
            }
            SampleProgression::Doubling { start, max } => {
                let mut sizes = Vec::new();
                let mut current = *start;
                while current <= max_available && current <= *max {
                    sizes.push(current);
                    current *= 2;
                }
                sizes
            }
            SampleProgression::Single(n) => {
                // Single fixed size (capped to available)
                vec![(*n).min(max_available)]
            }
            SampleProgression::Percent(pct) => {
                // Percentage of available data
                let n = ((max_available as f32) * (*pct / 100.0)).round() as usize;
                vec![n.max(1).min(max_available)]
            }
        }
    }
    
    /// Create single-run mode with fixed sample count
    pub fn single(n: usize) -> Self {
        SampleProgression::Single(n)
    }
    
    /// Create single-run mode with percentage of data
    pub fn percent(pct: f32) -> Self {
        SampleProgression::Percent(pct)
    }
}

/// Configuration for the AutoTrainer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTrainerConfig {
    /// Fraction of data for validation (default: 0.2 = 20%)
    pub validation_fraction: f64,

    /// Maximum validation set size (to cap memory usage)
    pub max_validation_size: usize,

    /// Sample progression strategy
    pub progression: SampleProgression,

    /// Sampling strategy for selecting training samples
    pub sampling_strategy: SamplingStrategy,

    /// Patience: how many non-improving steps before stopping
    pub patience: usize,

    /// Minimum improvement to count as "improving" (default: 0.001 = 0.1%)
    pub min_improvement: f64,

    /// Overfitting threshold: train - val > threshold triggers warning
    pub overfit_threshold: f64,

    /// Overfitting threshold for hard stop
    pub overfit_stop_threshold: f64,

    /// CMA-ES generations per training step
    pub cmaes_generations: usize,

    /// CMA-ES population size
    pub cmaes_population: usize,

    /// Target compressed dimension
    pub compressed_dim: usize,

    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,

    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for AutoTrainerConfig {
    fn default() -> Self {
        Self {
            validation_fraction: 0.2,
            max_validation_size: 50_000,
            progression: SampleProgression::default(),
            sampling_strategy: SamplingStrategy::default(),
            patience: 3,
            min_improvement: 0.001,
            overfit_threshold: 0.02,      // 2% gap = warning
            overfit_stop_threshold: 0.05,  // 5% gap = stop
            cmaes_generations: 100,
            cmaes_population: 50,
            compressed_dim: 13,
            seed: Some(42),
            verbose: true,
        }
    }
}

impl AutoTrainerConfig {
    /// Create config for instant testing (minimal CMA-ES - ~2 seconds)
    /// Only 5 generations × 10 population = 50 evaluations
    pub fn instant() -> Self {
        Self {
            cmaes_generations: 5,
            cmaes_population: 10,
            patience: 1,
            progression: SampleProgression::Single(50),  // Only 50 samples!
            ..Default::default()
        }
    }
    
    /// Create config for quick training (fewer generations, smaller populations)
    pub fn quick() -> Self {
        Self {
            cmaes_generations: 50,
            cmaes_population: 30,
            // Start with smaller sizes to work with datasets of all sizes
            progression: SampleProgression::Fixed(vec![100, 200, 500, 1000, 2000, 5000]),
            ..Default::default()
        }
    }

    /// Create config for thorough training (more generations)
    pub fn thorough() -> Self {
        Self {
            cmaes_generations: 200,
            cmaes_population: 100,
            patience: 5,
            ..Default::default()
        }
    }

    /// Set compressed dimension
    pub fn with_compressed_dim(mut self, dim: usize) -> Self {
        self.compressed_dim = dim;
        self
    }

    /// Set patience
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Set single-run mode with fixed sample count (fast testing)
    pub fn with_single_samples(mut self, n: usize) -> Self {
        self.progression = SampleProgression::Single(n);
        self
    }
    
    /// Set single-run mode with percentage of data (fast testing)
    pub fn with_percent_samples(mut self, pct: f32) -> Self {
        self.progression = SampleProgression::Percent(pct);
        self
    }

    /// Disable verbose output
    pub fn quiet(mut self) -> Self {
        self.verbose = false;
        self
    }

    /// Set sampling strategy
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Use K-Means++ sampling for better space coverage
    pub fn with_kmeans_sampling(mut self) -> Self {
        self.sampling_strategy = SamplingStrategy::KMeans;
        self
    }

    /// Use t-SNE Grid sampling for uniform space coverage
    pub fn with_tsne_grid_sampling(mut self) -> Self {
        self.sampling_strategy = SamplingStrategy::TsneGrid;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AutoTrainerConfig::default();
        assert_eq!(config.validation_fraction, 0.2);
        assert_eq!(config.patience, 3);
        assert_eq!(config.compressed_dim, 13);
    }

    #[test]
    fn test_fixed_progression() {
        let prog = SampleProgression::Fixed(vec![1000, 2000, 5000, 10000]);

        // All fit
        assert_eq!(prog.sizes(20000), vec![1000, 2000, 5000, 10000]);

        // Some filtered
        assert_eq!(prog.sizes(3000), vec![1000, 2000]);

        // None fit
        assert_eq!(prog.sizes(500), Vec::<usize>::new());
    }

    #[test]
    fn test_doubling_progression() {
        let prog = SampleProgression::Doubling { start: 500, max: 10000 };

        assert_eq!(prog.sizes(20000), vec![500, 1000, 2000, 4000, 8000]);
        assert_eq!(prog.sizes(3000), vec![500, 1000, 2000]);
    }
}
