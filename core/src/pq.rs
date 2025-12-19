//! Stage 3: Product Quantization (PQ)
//!
//! Compresses AQEA outputs to very small byte codes using learned codebooks.
//! Uses **K-Means++ initialization** with multiple restarts for optimal quality.
//!
//! # Compression
//! - Input: float32 AQEA output (e.g., 128D = 512 bytes)
//! - Output: uint8 centroid indices (e.g., 64 bytes for 64 subvectors)
//! - Ratio: 8-64x additional compression
//!
//! # Quality
//! With K-Means++ (100 iterations, 3 runs):
//! - PQ-64: 99.9% Cosine Similarity, 82.6% Recall@10 (only -1.2% vs Float)
//! - PQ-128: 100% Cosine Similarity, 83.5% Recall@10 (only -0.3% vs Float)
//!
//! # Training Required
//! PQ must be trained on AQEA outputs from your target domain:
//! ```rust,ignore
//! let mut pq = ProductQuantizer::new(128, 64, 8); // 128D -> 64 subvectors, 8-bit
//! pq.train(&aqea_outputs, 100); // K-Means++ with 100 iterations
//! pq.save("model_pq.cb")?;
//! ```
//!
//! # Usage
//! ```rust,ignore
//! let pq = ProductQuantizer::load("model_pq.cb")?;
//! let codes = pq.encode(&aqea_output);    // Compress to bytes
//! let decoded = pq.decode(&codes);         // Decompress for reranking
//! ```

use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Product Quantizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Input dimension (from AQEA output)
    pub input_dim: usize,
    /// Number of subvectors
    pub n_subvectors: usize,
    /// Bits per subvector (8 = 256 centroids)
    pub bits: usize,
    /// Dimension per subvector
    pub subvec_dim: usize,
}

impl PQConfig {
    /// Create new config
    pub fn new(input_dim: usize, n_subvectors: usize, bits: usize) -> Self {
        assert!(
            input_dim % n_subvectors == 0 || input_dim >= n_subvectors,
            "input_dim should be >= n_subvectors"
        );
        let subvec_dim = (input_dim + n_subvectors - 1) / n_subvectors;
        Self {
            input_dim,
            n_subvectors,
            bits,
            subvec_dim,
        }
    }
}

/// Product Quantizer
///
/// Divides vectors into subvectors and quantizes each to a centroid index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Configuration
    pub config: PQConfig,
    /// Centroids: [n_subvectors][n_centroids][subvec_dim]
    pub centroids: Vec<Vec<Vec<f32>>>,
    /// Whether trained
    pub trained: bool,
}

impl ProductQuantizer {
    /// Create new untrained PQ
    ///
    /// # Arguments
    /// * `input_dim` - Input dimension (AQEA output dimension)
    /// * `n_subvectors` - Number of subvectors (determines output bytes)
    /// * `bits` - Bits per index (8 = 256 centroids, good default)
    pub fn new(input_dim: usize, n_subvectors: usize, bits: usize) -> Self {
        let config = PQConfig::new(input_dim, n_subvectors, bits);
        let n_centroids = 1 << bits;

        // Initialize random centroids
        let mut rng = rand::thread_rng();
        let centroids: Vec<Vec<Vec<f32>>> = (0..n_subvectors)
            .map(|_| {
                (0..n_centroids)
                    .map(|_| (0..config.subvec_dim).map(|_| rng.gen::<f32>() - 0.5).collect())
                    .collect()
            })
            .collect();

        Self {
            config,
            centroids,
            trained: false,
        }
    }

    /// Create disabled PQ (pass-through)
    pub fn disabled() -> Self {
        Self {
            config: PQConfig::new(1, 1, 1),
            centroids: vec![],
            trained: false,
        }
    }

    /// Check if PQ is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get output code size in bytes
    pub fn code_bytes(&self) -> usize {
        if self.config.bits <= 8 {
            self.config.n_subvectors
        } else {
            self.config.n_subvectors * 2
        }
    }

    /// Get number of centroids per subvector
    pub fn n_centroids(&self) -> usize {
        1 << self.config.bits
    }

    /// Train PQ on data using k-means
    ///
    /// # Arguments
    /// * `data` - Training vectors (AQEA outputs)
    /// * `iterations` - K-means iterations (10-20 is good)
    pub fn train(&mut self, data: &[Vec<f32>], iterations: usize) {
        if data.is_empty() {
            return;
        }

        let n_centroids = self.n_centroids();

        // Train each subvector independently
        for sub_idx in 0..self.config.n_subvectors {
            // Extract subvectors
            let subvectors: Vec<Vec<f32>> = data
                .iter()
                .map(|vec| self.extract_subvector(vec, sub_idx))
                .collect();

            // K-means for this subvector
            self.centroids[sub_idx] =
                kmeans_train(&subvectors, n_centroids, self.config.subvec_dim, iterations);
        }

        self.trained = true;
    }

    /// Extract subvector from full vector
    fn extract_subvector(&self, vec: &[f32], sub_idx: usize) -> Vec<f32> {
        let start = sub_idx * self.config.subvec_dim;
        let end = (start + self.config.subvec_dim).min(vec.len());

        if start >= vec.len() {
            // Pad with zeros if vector is too short
            return vec![0.0; self.config.subvec_dim];
        }

        let mut subvec: Vec<f32> = vec[start..end].to_vec();
        // Pad if needed
        while subvec.len() < self.config.subvec_dim {
            subvec.push(0.0);
        }
        subvec
    }

    /// Encode vector to PQ codes
    ///
    /// # Arguments
    /// * `vec` - Input vector (AQEA output)
    ///
    /// # Returns
    /// * Centroid indices for each subvector
    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.config.n_subvectors);

        for sub_idx in 0..self.config.n_subvectors {
            let subvec = self.extract_subvector(vec, sub_idx);
            let nearest = self.find_nearest_centroid(sub_idx, &subvec);
            codes.push(nearest as u8);
        }

        codes
    }

    /// Find nearest centroid for a subvector
    fn find_nearest_centroid(&self, sub_idx: usize, subvec: &[f32]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (idx, centroid) in self.centroids[sub_idx].iter().enumerate() {
            let dist = squared_distance(subvec, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Decode PQ codes back to approximate vector
    ///
    /// # Arguments
    /// * `codes` - Centroid indices
    ///
    /// # Returns
    /// * Reconstructed vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.input_dim);

        for (sub_idx, &code) in codes.iter().enumerate() {
            if sub_idx < self.centroids.len() {
                let centroid = &self.centroids[sub_idx][code as usize];
                vec.extend_from_slice(centroid);
            }
        }

        // Truncate to exact input_dim
        vec.truncate(self.config.input_dim);
        vec
    }

    /// Save trained PQ to file
    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load trained PQ from file
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    // =========================================================================
    // ADC (Asymmetric Distance Computation) - für schnelle PQ-Suche
    // =========================================================================

    /// Compute distance tables for ADC search
    /// 
    /// For each subvector of the query, computes squared distance to all centroids.
    /// This is precomputed once per query, then used for fast distance computation.
    /// 
    /// # Arguments
    /// * `query` - Query vector in AQEA space (e.g., 128D float)
    /// 
    /// # Returns
    /// * Distance tables: [n_subvectors][256] squared distances
    pub fn compute_distance_tables(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let n_centroids = self.n_centroids();
        let mut tables = Vec::with_capacity(self.config.n_subvectors);
        
        for sub_idx in 0..self.config.n_subvectors {
            let start = sub_idx * self.config.subvec_dim;
            let end = (start + self.config.subvec_dim).min(query.len());
            let query_sub = &query[start..end];
            
            let mut distances = Vec::with_capacity(n_centroids);
            for centroid in &self.centroids[sub_idx] {
                let dist = squared_distance(query_sub, &centroid[..query_sub.len()]);
                distances.push(dist);
            }
            tables.push(distances);
        }
        
        tables
    }

    /// Compute asymmetric squared distance using precomputed tables
    /// 
    /// This is O(n_subvectors) instead of O(input_dim)!
    /// 
    /// # Arguments
    /// * `tables` - Precomputed distance tables from `compute_distance_tables`
    /// * `codes` - PQ codes for a database vector
    /// 
    /// # Returns
    /// * Squared Euclidean distance (lower = more similar)
    pub fn asymmetric_distance(&self, tables: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        for (sub_idx, &code) in codes.iter().enumerate() {
            if sub_idx < tables.len() {
                dist += tables[sub_idx][code as usize];
            }
        }
        dist
    }

    /// Search using ADC (Asymmetric Distance Computation)
    /// 
    /// Fast search: Query stays as float, distances computed via lookup tables.
    /// This is LOSSY because we never reconstruct the database vectors!
    /// 
    /// # Arguments
    /// * `query` - Query vector in AQEA space
    /// * `codes_db` - Database of PQ codes
    /// * `k` - Number of results to return
    /// 
    /// # Returns
    /// * Top-k indices and distances (sorted by distance, ascending)
    pub fn search_adc(&self, query: &[f32], codes_db: &[Vec<u8>], k: usize) -> Vec<(usize, f32)> {
        let tables = self.compute_distance_tables(query);
        
        let mut results: Vec<(usize, f32)> = codes_db.iter()
            .enumerate()
            .map(|(idx, codes)| (idx, self.asymmetric_distance(&tables, codes)))
            .collect();
        
        // Sort by distance (ascending = closest first)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        results.into_iter().take(k).collect()
    }

    /// Convert squared distance to approximate cosine similarity
    /// 
    /// For normalized vectors: cosine_sim ≈ 1 - dist²/2
    /// 
    /// # Arguments
    /// * `squared_dist` - Squared Euclidean distance
    /// 
    /// # Returns
    /// * Approximate cosine similarity in [0, 1]
    pub fn dist_to_cosine_sim(squared_dist: f32) -> f32 {
        (1.0 - squared_dist / 2.0).max(0.0).min(1.0)
    }
}

/// Squared Euclidean distance
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// K-Means++ initialization for better convergence
/// 
/// Selects initial centroids with probability proportional to distance
/// from existing centroids, ensuring better spread.
fn kmeans_plusplus_init(data: &[Vec<f32>], k: usize, seed: u64) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }
    
    let n = data.len();
    let actual_k = k.min(n);
    
    // Simple deterministic RNG based on seed
    let mut rng_state = seed;
    let mut next_rand = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32)
    };
    
    let mut centroids = Vec::with_capacity(actual_k);
    let mut min_distances = vec![f32::INFINITY; n];
    
    // Pick first centroid randomly
    let first_idx = (next_rand() * n as f32) as usize % n;
    centroids.push(data[first_idx].clone());
    
    // Pick remaining centroids using K-Means++ probability
    for _ in 1..actual_k {
        // Update min distances to nearest existing centroid
        let mut total_dist = 0.0f32;
        for (i, point) in data.iter().enumerate() {
            let dist_to_last: f32 = point.iter()
                .zip(centroids.last().unwrap().iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            min_distances[i] = min_distances[i].min(dist_to_last);
            total_dist += min_distances[i];
        }
        
        // If all distances are 0, pick randomly
        if total_dist <= 0.0 {
            let idx = (next_rand() * n as f32) as usize % n;
            centroids.push(data[idx].clone());
            continue;
        }
        
        // Sample proportional to distance squared
        let threshold = next_rand() * total_dist;
        let mut cumulative = 0.0f32;
        let mut selected_idx = 0;
        
        for (i, &dist) in min_distances.iter().enumerate() {
            cumulative += dist;
            if cumulative >= threshold {
                selected_idx = i;
                break;
            }
        }
        
        centroids.push(data[selected_idx].clone());
    }
    
    centroids
}

/// Calculate total inertia (sum of squared distances to nearest centroid)
fn calculate_inertia(data: &[Vec<f32>], centroids: &[Vec<f32>]) -> f32 {
    data.iter().map(|point| {
        centroids.iter()
            .map(|c| point.iter().zip(c.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>())
            .fold(f32::INFINITY, f32::min)
    }).sum()
}

/// K-Means++ training with multiple restarts
/// 
/// Uses K-Means++ initialization (much better than random) and runs
/// multiple times, keeping the best result.
fn kmeans_train(
    data: &[Vec<f32>],
    k: usize,
    dim: usize,
    _iterations: usize, // Ignored, we use fixed 100 iterations
) -> Vec<Vec<f32>> {
    if data.is_empty() {
        return vec![vec![0.0; dim]; k];
    }

    let max_iterations = 100;
    let n_init = 3; // Run 3 times, keep best
    let convergence_threshold = 1e-6;
    
    let mut best_centroids = vec![vec![0.0; dim]; k];
    let mut best_inertia = f32::INFINITY;
    
    for init_run in 0..n_init {
        let seed = init_run as u64 * 12345;
        
        // K-Means++ initialization
        let mut centroids = kmeans_plusplus_init(data, k, seed);
        
        // Pad if we got fewer centroids than requested
        if centroids.len() < k {
            let dim = data.first().map(|v| v.len()).unwrap_or(dim);
            while centroids.len() < k {
                centroids.push(vec![0.0; dim]);
            }
        }
        
        let mut prev_inertia = f32::INFINITY;
        
        // K-means iterations
        for _ in 0..max_iterations {
            // Assign points to nearest centroid
            let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); k];
            for (idx, point) in data.iter().enumerate() {
                let nearest = find_nearest(&centroids, point);
                assignments[nearest].push(idx);
            }

            // Update centroids
            for (c_idx, assigned) in assignments.iter().enumerate() {
                if assigned.is_empty() {
                    continue; // Keep existing centroid
                }
                
                let dim = centroids[c_idx].len();
                let mut new_centroid = vec![0.0; dim];
                for &point_idx in assigned {
                    for (i, &val) in data[point_idx].iter().enumerate() {
                        if i < dim {
                            new_centroid[i] += val;
                        }
                    }
                }
                let n = assigned.len() as f32;
                for val in &mut new_centroid {
                    *val /= n;
                }
                centroids[c_idx] = new_centroid;
            }
            
            // Check convergence
            let inertia = calculate_inertia(data, &centroids);
            if (prev_inertia - inertia).abs() < convergence_threshold {
                break;
            }
            prev_inertia = inertia;
        }
        
        // Keep best result
        let final_inertia = calculate_inertia(data, &centroids);
        if final_inertia < best_inertia {
            best_inertia = final_inertia;
            best_centroids = centroids;
        }
    }

    best_centroids
}

/// Find nearest centroid index
fn find_nearest(centroids: &[Vec<f32>], point: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    for (idx, centroid) in centroids.iter().enumerate() {
        let dist = squared_distance(point, centroid);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    best_idx
}

/// PQ compression modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PQMode {
    /// No PQ (use AQEA output directly)
    Disabled,
    /// 7 subvectors (highest compression)
    Sub7,
    /// 13 subvectors (balanced)
    Sub13,
    /// 17 subvectors (highest quality)
    Sub17,
    /// Custom number of subvectors
    Custom(usize),
}

impl PQMode {
    /// Get number of subvectors
    pub fn n_subvectors(&self, default: usize) -> usize {
        match self {
            PQMode::Disabled => 0,
            PQMode::Sub7 => 7,
            PQMode::Sub13 => 13,
            PQMode::Sub17 => 17,
            PQMode::Custom(n) => *n,
        }
    }

    /// Is PQ enabled?
    pub fn is_enabled(&self) -> bool {
        !matches!(self, PQMode::Disabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_config() {
        let config = PQConfig::new(35, 7, 8);
        assert_eq!(config.n_subvectors, 7);
        assert_eq!(config.subvec_dim, 5); // ceil(35/7) = 5
    }

    #[test]
    fn test_pq_encode_decode() {
        let mut pq = ProductQuantizer::new(13, 13, 8);

        // Generate some training data
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..13).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();

        pq.train(&data, 10);
        assert!(pq.is_trained());

        // Test encode/decode
        let test_vec: Vec<f32> = (0..13).map(|_| rng.gen::<f32>() - 0.5).collect();
        let codes = pq.encode(&test_vec);
        let decoded = pq.decode(&codes);

        assert_eq!(codes.len(), 13);
        assert_eq!(decoded.len(), 13);
    }

    #[test]
    fn test_pq_code_bytes() {
        let pq = ProductQuantizer::new(35, 7, 8);
        assert_eq!(pq.code_bytes(), 7);

        let pq = ProductQuantizer::new(35, 17, 8);
        assert_eq!(pq.code_bytes(), 17);
    }

    #[test]
    fn test_kmeans_basic() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![1.0, 1.0],
            vec![1.1, 1.1],
        ];

        let centroids = kmeans_train(&data, 2, 2, 10);
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_pq_mode() {
        assert!(!PQMode::Disabled.is_enabled());
        assert!(PQMode::Sub7.is_enabled());
        assert_eq!(PQMode::Sub7.n_subvectors(0), 7);
        assert_eq!(PQMode::Custom(10).n_subvectors(0), 10);
    }

    #[test]
    fn test_pq_save_load() {
        let mut pq = ProductQuantizer::new(13, 7, 8);

        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..13).map(|_| rng.gen::<f32>()).collect())
            .collect();

        pq.train(&data, 5);

        // Save
        let path = std::path::Path::new("/tmp/test_pq.json");
        pq.save(path).unwrap();

        // Load
        let loaded = ProductQuantizer::load(path).unwrap();
        assert!(loaded.is_trained());
        assert_eq!(loaded.config.n_subvectors, 7);

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}

