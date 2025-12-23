//! Data Splitter for Train/Validation Split
//!
//! Provides deterministic 80/20 splitting with optional stratification.

use rand::prelude::*;

/// Split data containing train and validation sets
#[derive(Debug, Clone)]
pub struct SplitData<T: Clone> {
    /// Training data
    pub train: Vec<T>,
    /// Validation data
    pub validation: Vec<T>,
}

impl<T: Clone> SplitData<T> {
    /// Get number of training samples
    pub fn train_size(&self) -> usize {
        self.train.len()
    }

    /// Get number of validation samples
    pub fn validation_size(&self) -> usize {
        self.validation.len()
    }
}

/// Data splitter for creating train/validation splits
pub struct DataSplitter {
    /// Fraction of data for validation
    validation_fraction: f64,
    /// Maximum validation set size
    max_validation_size: Option<usize>,
    /// Random seed for reproducibility
    seed: u64,
}

impl DataSplitter {
    /// Create a new DataSplitter with default 80/20 split
    pub fn new() -> Self {
        Self {
            validation_fraction: 0.2,
            max_validation_size: None,
            seed: 42,
        }
    }

    /// Set validation fraction (0.0 - 1.0)
    pub fn with_validation_fraction(mut self, fraction: f64) -> Self {
        self.validation_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set maximum validation size (for memory efficiency with large datasets)
    pub fn with_max_validation_size(mut self, max_size: usize) -> Self {
        self.max_validation_size = Some(max_size);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Split a single vector of data
    pub fn split<T: Clone>(&self, data: &[T]) -> SplitData<T> {
        let n = data.len();
        if n == 0 {
            return SplitData {
                train: Vec::new(),
                validation: Vec::new(),
            };
        }

        // Calculate validation size
        let mut val_size = (n as f64 * self.validation_fraction).round() as usize;
        val_size = val_size.max(1).min(n - 1); // At least 1, at most n-1

        // Cap validation size if specified
        if let Some(max) = self.max_validation_size {
            val_size = val_size.min(max);
        }

        let train_size = n - val_size;

        // Create shuffled indices
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        // Split
        let train: Vec<T> = indices[..train_size]
            .iter()
            .map(|&i| data[i].clone())
            .collect();

        let validation: Vec<T> = indices[train_size..]
            .iter()
            .map(|&i| data[i].clone())
            .collect();

        SplitData { train, validation }
    }

    /// Split paired data (embeddings1, embeddings2)
    /// Keeps pairs together during split
    pub fn split_pairs<T: Clone, U: Clone>(
        &self,
        data1: &[T],
        data2: &[U],
    ) -> (SplitData<T>, SplitData<U>) {
        let n = data1.len().min(data2.len());
        if n == 0 {
            return (
                SplitData { train: Vec::new(), validation: Vec::new() },
                SplitData { train: Vec::new(), validation: Vec::new() },
            );
        }

        // Calculate validation size
        let mut val_size = (n as f64 * self.validation_fraction).round() as usize;
        val_size = val_size.max(1).min(n - 1);

        if let Some(max) = self.max_validation_size {
            val_size = val_size.min(max);
        }

        let train_size = n - val_size;

        // Create shuffled indices
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        // Split data1
        let train1: Vec<T> = indices[..train_size]
            .iter()
            .map(|&i| data1[i].clone())
            .collect();
        let val1: Vec<T> = indices[train_size..]
            .iter()
            .map(|&i| data1[i].clone())
            .collect();

        // Split data2
        let train2: Vec<U> = indices[..train_size]
            .iter()
            .map(|&i| data2[i].clone())
            .collect();
        let val2: Vec<U> = indices[train_size..]
            .iter()
            .map(|&i| data2[i].clone())
            .collect();

        (
            SplitData { train: train1, validation: val1 },
            SplitData { train: train2, validation: val2 },
        )
    }

    /// Get the indices for train/validation split
    /// Useful when you need to apply the same split to multiple arrays
    pub fn get_split_indices(&self, n: usize) -> (Vec<usize>, Vec<usize>) {
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut val_size = (n as f64 * self.validation_fraction).round() as usize;
        val_size = val_size.max(1).min(n - 1);

        if let Some(max) = self.max_validation_size {
            val_size = val_size.min(max);
        }

        let train_size = n - val_size;

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let train_indices = indices[..train_size].to_vec();
        let val_indices = indices[train_size..].to_vec();

        (train_indices, val_indices)
    }
}

impl Default for DataSplitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_split() {
        let data: Vec<i32> = (0..100).collect();
        let splitter = DataSplitter::new();
        let split = splitter.split(&data);

        assert_eq!(split.train_size(), 80);
        assert_eq!(split.validation_size(), 20);
    }

    #[test]
    fn test_split_deterministic() {
        let data: Vec<i32> = (0..100).collect();
        let splitter = DataSplitter::new().with_seed(123);

        let split1 = splitter.split(&data);
        let split2 = splitter.split(&data);

        assert_eq!(split1.train, split2.train);
        assert_eq!(split1.validation, split2.validation);
    }

    #[test]
    fn test_split_pairs() {
        let data1: Vec<i32> = (0..100).collect();
        let data2: Vec<i32> = (100..200).collect();
        let splitter = DataSplitter::new();

        let (split1, split2) = splitter.split_pairs(&data1, &data2);

        assert_eq!(split1.train_size(), split2.train_size());
        assert_eq!(split1.validation_size(), split2.validation_size());
        assert_eq!(split1.train_size(), 80);
    }

    #[test]
    fn test_max_validation_size() {
        let data: Vec<i32> = (0..1000).collect();
        let splitter = DataSplitter::new().with_max_validation_size(50);
        let split = splitter.split(&data);

        assert_eq!(split.validation_size(), 50);
        assert_eq!(split.train_size(), 950);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<i32> = vec![];
        let splitter = DataSplitter::new();
        let split = splitter.split(&data);

        assert!(split.train.is_empty());
        assert!(split.validation.is_empty());
    }

    #[test]
    fn test_small_data() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let splitter = DataSplitter::new();
        let split = splitter.split(&data);

        // With 5 items: 1 goes to validation (20% rounded)
        assert_eq!(split.train_size() + split.validation_size(), 5);
        assert!(split.validation_size() >= 1);
    }

    #[test]
    fn test_split_indices() {
        let splitter = DataSplitter::new().with_seed(42);
        let (train_idx, val_idx) = splitter.get_split_indices(100);

        assert_eq!(train_idx.len(), 80);
        assert_eq!(val_idx.len(), 20);

        // No overlap
        for idx in &val_idx {
            assert!(!train_idx.contains(idx));
        }
    }
}
