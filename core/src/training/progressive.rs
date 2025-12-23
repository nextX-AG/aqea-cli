//! Progressive Trainer
//!
//! Trains with progressively larger sample sizes to find optimal training amount.

use rand::prelude::*;
use crate::training::config::SampleProgression;

/// Result of one progressive training step
#[derive(Debug, Clone)]
pub struct ProgressStep {
    /// Number of samples used
    pub n_samples: usize,
    /// Training score achieved
    pub train_score: f64,
    /// Validation score achieved
    pub val_score: f64,
    /// Trained weights
    pub weights: Vec<f32>,
    /// Training time in seconds
    pub time_secs: f64,
}

/// Progressive trainer that increases sample size over iterations
pub struct ProgressiveTrainer {
    /// Sample progression strategy
    progression: SampleProgression,
    /// Current step index
    current_step: usize,
    /// Available sample sizes
    sample_sizes: Vec<usize>,
    /// Random seed
    seed: u64,
}

impl ProgressiveTrainer {
    /// Create new progressive trainer
    ///
    /// # Arguments
    /// * `progression` - Sample size progression strategy
    /// * `max_available` - Maximum number of samples available
    /// * `seed` - Random seed for sampling
    pub fn new(progression: SampleProgression, max_available: usize, seed: u64) -> Self {
        let sample_sizes = progression.sizes(max_available);

        Self {
            progression,
            current_step: 0,
            sample_sizes,
            seed,
        }
    }

    /// Get next sample size, or None if done
    pub fn next_sample_size(&mut self) -> Option<usize> {
        if self.current_step < self.sample_sizes.len() {
            let size = self.sample_sizes[self.current_step];
            self.current_step += 1;
            Some(size)
        } else {
            None
        }
    }

    /// Peek at next sample size without advancing
    pub fn peek_next(&self) -> Option<usize> {
        self.sample_sizes.get(self.current_step).copied()
    }

    /// Get current step number (0-indexed)
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get total number of steps
    pub fn total_steps(&self) -> usize {
        self.sample_sizes.len()
    }

    /// Check if there are more steps
    pub fn has_more(&self) -> bool {
        self.current_step < self.sample_sizes.len()
    }

    /// Get all sample sizes
    pub fn sample_sizes(&self) -> &[usize] {
        &self.sample_sizes
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Sample indices for training
    ///
    /// Returns random indices for the given sample size from the training set.
    pub fn sample_indices(&self, n_total: usize, n_sample: usize) -> Vec<usize> {
        if n_sample >= n_total {
            return (0..n_total).collect();
        }

        // Use step-based seed for reproducibility
        let step_seed = self.seed.wrapping_add(self.current_step as u64);
        let mut rng = StdRng::seed_from_u64(step_seed);

        let mut indices: Vec<usize> = (0..n_total).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n_sample);
        indices
    }

    /// Sample data for training
    pub fn sample_data<T: Clone>(&self, data: &[T], n_sample: usize) -> Vec<T> {
        let indices = self.sample_indices(data.len(), n_sample);
        indices.iter().map(|&i| data[i].clone()).collect()
    }

    /// Sample paired data for training
    pub fn sample_pairs<T: Clone, U: Clone>(
        &self,
        data1: &[T],
        data2: &[U],
        n_sample: usize,
    ) -> (Vec<T>, Vec<U>) {
        let n = data1.len().min(data2.len());
        let indices = self.sample_indices(n, n_sample);

        let sampled1 = indices.iter().map(|&i| data1[i].clone()).collect();
        let sampled2 = indices.iter().map(|&i| data2[i].clone()).collect();

        (sampled1, sampled2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_iteration() {
        let progression = SampleProgression::Fixed(vec![1000, 2000, 5000]);
        let mut trainer = ProgressiveTrainer::new(progression, 10000, 42);

        assert_eq!(trainer.total_steps(), 3);
        assert!(trainer.has_more());

        assert_eq!(trainer.next_sample_size(), Some(1000));
        assert_eq!(trainer.next_sample_size(), Some(2000));
        assert_eq!(trainer.next_sample_size(), Some(5000));
        assert_eq!(trainer.next_sample_size(), None);
        assert!(!trainer.has_more());
    }

    #[test]
    fn test_progressive_limited_by_available() {
        let progression = SampleProgression::Fixed(vec![1000, 2000, 5000, 10000]);
        let trainer = ProgressiveTrainer::new(progression, 3000, 42);

        // Only 1000 and 2000 should be available
        assert_eq!(trainer.sample_sizes(), &[1000, 2000]);
    }

    #[test]
    fn test_sample_indices() {
        let progression = SampleProgression::Fixed(vec![100]);
        let trainer = ProgressiveTrainer::new(progression, 1000, 42);

        let indices = trainer.sample_indices(1000, 100);
        assert_eq!(indices.len(), 100);

        // All indices should be unique and in range
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 100);
        assert!(sorted.iter().all(|&i| i < 1000));
    }

    #[test]
    fn test_sample_indices_all() {
        let progression = SampleProgression::Fixed(vec![100]);
        let trainer = ProgressiveTrainer::new(progression, 1000, 42);

        // Request more than available
        let indices = trainer.sample_indices(50, 100);
        assert_eq!(indices.len(), 50); // Returns all available
    }

    #[test]
    fn test_sample_data() {
        let progression = SampleProgression::Fixed(vec![5]);
        let trainer = ProgressiveTrainer::new(progression, 100, 42);

        let data: Vec<i32> = (0..20).collect();
        let sampled = trainer.sample_data(&data, 5);

        assert_eq!(sampled.len(), 5);
        for val in &sampled {
            assert!(*val >= 0 && *val < 20);
        }
    }

    #[test]
    fn test_sample_pairs() {
        let progression = SampleProgression::Fixed(vec![5]);
        let trainer = ProgressiveTrainer::new(progression, 100, 42);

        let data1: Vec<i32> = (0..20).collect();
        let data2: Vec<i32> = (100..120).collect();

        let (sampled1, sampled2) = trainer.sample_pairs(&data1, &data2, 5);

        assert_eq!(sampled1.len(), 5);
        assert_eq!(sampled2.len(), 5);

        // Check that pairs are consistent (same indices were used)
        for (v1, v2) in sampled1.iter().zip(sampled2.iter()) {
            assert_eq!(*v2, *v1 + 100);
        }
    }

    #[test]
    fn test_reset() {
        let progression = SampleProgression::Fixed(vec![1000, 2000]);
        let mut trainer = ProgressiveTrainer::new(progression, 10000, 42);

        trainer.next_sample_size();
        trainer.next_sample_size();
        assert!(!trainer.has_more());

        trainer.reset();
        assert!(trainer.has_more());
        assert_eq!(trainer.current_step(), 0);
    }

    #[test]
    fn test_peek() {
        let progression = SampleProgression::Fixed(vec![1000, 2000]);
        let mut trainer = ProgressiveTrainer::new(progression, 10000, 42);

        assert_eq!(trainer.peek_next(), Some(1000));
        assert_eq!(trainer.peek_next(), Some(1000)); // Still 1000

        trainer.next_sample_size();
        assert_eq!(trainer.peek_next(), Some(2000));
    }
}
