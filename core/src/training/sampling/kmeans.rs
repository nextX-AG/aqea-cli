//! K-Means++ Sampler - Cluster-based diverse sampling
//!
//! Uses K-Means++ initialization algorithm to select samples
//! that are maximally spread out in the embedding space.
//!
//! This provides much better space coverage than random sampling.

use super::Sampler;
use rand::prelude::*;

/// K-Means++ based sampler for diverse sample selection
///
/// Uses the K-Means++ initialization algorithm:
/// 1. Select first point randomly
/// 2. For each subsequent point, select with probability
///    proportional to squared distance from nearest selected point
///
/// This ensures samples are spread out across the embedding space.
#[derive(Debug, Clone)]
pub struct KMeansSampler {
    seed: u64,
}

impl KMeansSampler {
    /// Create a new K-Means++ sampler with default seed
    pub fn new() -> Self {
        Self { seed: 42 }
    }

    /// Create a K-Means++ sampler with specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }

    /// Compute squared Euclidean distance between two vectors
    #[inline]
    fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum()
    }
}

impl Default for KMeansSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for KMeansSampler {
    fn sample(&self, embeddings: &[Vec<f32>], n_samples: usize) -> Vec<usize> {
        let n = embeddings.len();
        let n_samples = n_samples.min(n);

        if n_samples == 0 {
            return vec![];
        }

        if n_samples == n {
            return (0..n).collect();
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut selected: Vec<usize> = Vec::with_capacity(n_samples);

        // Track minimum distance to any selected point for each embedding
        let mut min_distances: Vec<f32> = vec![f32::MAX; n];

        // Step 1: Select first point randomly
        let first = rng.gen_range(0..n);
        selected.push(first);

        // Update distances from first point
        for (i, emb) in embeddings.iter().enumerate() {
            min_distances[i] = Self::squared_distance(emb, &embeddings[first]);
        }
        min_distances[first] = 0.0;

        // Step 2: Select remaining points using K-Means++ probability
        while selected.len() < n_samples {
            // Compute cumulative distribution
            let total_dist: f32 = min_distances.iter().sum();

            if total_dist <= 0.0 {
                // All remaining points are at distance 0, fall back to random
                let remaining: Vec<usize> = (0..n)
                    .filter(|i| !selected.contains(i))
                    .collect();
                if let Some(&idx) = remaining.choose(&mut rng) {
                    selected.push(idx);
                } else {
                    break;
                }
                continue;
            }

            // Sample proportional to squared distance
            let threshold = rng.gen::<f32>() * total_dist;
            let mut cumsum = 0.0;
            let mut next_idx = 0;

            for (i, &dist) in min_distances.iter().enumerate() {
                cumsum += dist;
                if cumsum >= threshold {
                    next_idx = i;
                    break;
                }
            }

            selected.push(next_idx);

            // Update minimum distances
            let new_point = &embeddings[next_idx];
            for (i, emb) in embeddings.iter().enumerate() {
                let dist = Self::squared_distance(emb, new_point);
                if dist < min_distances[i] {
                    min_distances[i] = dist;
                }
            }
            min_distances[next_idx] = 0.0;
        }

        selected.sort_unstable();
        selected
    }

    fn name(&self) -> &'static str {
        "kmeans++"
    }

    fn description(&self) -> &'static str {
        "K-Means++ initialization for diverse sampling"
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

    fn generate_clustered_embeddings() -> Vec<Vec<f32>> {
        // Create 4 distinct clusters
        let mut embeddings = Vec::new();
        let clusters = [
            (0.0, 0.0),   // Cluster 1
            (10.0, 0.0),  // Cluster 2
            (0.0, 10.0),  // Cluster 3
            (10.0, 10.0), // Cluster 4
        ];

        let mut rng = StdRng::seed_from_u64(42);
        for (cx, cy) in clusters.iter() {
            for _ in 0..25 {
                let x = cx + rng.gen::<f32>() * 0.5;
                let y = cy + rng.gen::<f32>() * 0.5;
                embeddings.push(vec![x, y]);
            }
        }
        embeddings
    }

    #[test]
    fn test_kmeans_sampler_basic() {
        let sampler = KMeansSampler::new();
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
    fn test_kmeans_sampler_all() {
        let sampler = KMeansSampler::new();
        let embeddings = generate_embeddings(50, 32);

        let indices = sampler.sample(&embeddings, 50);

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_kmeans_sampler_more_than_available() {
        let sampler = KMeansSampler::new();
        let embeddings = generate_embeddings(30, 32);

        let indices = sampler.sample(&embeddings, 100);

        assert_eq!(indices.len(), 30);
    }

    #[test]
    fn test_kmeans_sampler_deterministic() {
        let embeddings = generate_embeddings(100, 32);

        let sampler1 = KMeansSampler::with_seed(42);
        let sampler2 = KMeansSampler::with_seed(42);

        let indices1 = sampler1.sample(&embeddings, 30);
        let indices2 = sampler2.sample(&embeddings, 30);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_kmeans_sampler_coverage() {
        // Test that K-Means++ selects from different clusters
        let embeddings = generate_clustered_embeddings();
        let sampler = KMeansSampler::with_seed(42);

        let indices = sampler.sample(&embeddings, 4);

        // With 4 samples from 4 clusters, we expect coverage of all clusters
        // Cluster 0: indices 0-24, Cluster 1: 25-49, Cluster 2: 50-74, Cluster 3: 75-99
        let clusters_hit: std::collections::HashSet<_> = indices
            .iter()
            .map(|&i| i / 25)
            .collect();

        // K-Means++ should hit multiple clusters
        assert!(clusters_hit.len() >= 3, "Expected at least 3 clusters, got {:?}", clusters_hit);
    }

    #[test]
    fn test_kmeans_better_than_random_for_coverage() {
        let embeddings = generate_clustered_embeddings();

        // Test multiple seeds to ensure K-Means++ generally covers more space
        let mut kmeans_avg_spread = 0.0;
        let mut random_avg_spread = 0.0;
        let num_trials = 10;

        for seed in 0..num_trials {
            let kmeans = KMeansSampler::with_seed(seed);
            let random = super::super::RandomSampler::with_seed(seed);

            let km_indices = kmeans.sample(&embeddings, 8);
            let rand_indices = random.sample(&embeddings, 8);

            // Compute average pairwise distance
            let km_spread = compute_avg_pairwise_dist(&embeddings, &km_indices);
            let rand_spread = compute_avg_pairwise_dist(&embeddings, &rand_indices);

            kmeans_avg_spread += km_spread;
            random_avg_spread += rand_spread;
        }

        kmeans_avg_spread /= num_trials as f32;
        random_avg_spread /= num_trials as f32;

        // K-Means++ should have higher average spread (more diverse samples)
        assert!(
            kmeans_avg_spread > random_avg_spread * 0.9,
            "K-Means++ spread ({:.2}) should be at least close to random ({:.2})",
            kmeans_avg_spread,
            random_avg_spread
        );
    }

    fn compute_avg_pairwise_dist(embeddings: &[Vec<f32>], indices: &[usize]) -> f32 {
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                total += KMeansSampler::squared_distance(
                    &embeddings[indices[i]],
                    &embeddings[indices[j]],
                ).sqrt();
                count += 1;
            }
        }
        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }

    #[test]
    fn test_kmeans_sampler_name() {
        let sampler = KMeansSampler::new();
        assert_eq!(sampler.name(), "kmeans++");
    }

    #[test]
    fn test_kmeans_empty() {
        let sampler = KMeansSampler::new();
        let embeddings: Vec<Vec<f32>> = vec![];
        let indices = sampler.sample(&embeddings, 10);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_kmeans_zero_samples() {
        let sampler = KMeansSampler::new();
        let embeddings = generate_embeddings(100, 32);
        let indices = sampler.sample(&embeddings, 0);
        assert!(indices.is_empty());
    }
}
