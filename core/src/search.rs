//! Search Module - AQEA Precision Boost Search
//!
//! Hybrid search that combines fast compressed search with precision reranking.
//!
//! # Architecture
//!
//! ```text
//! Query → Compressed Index Scan (Top-N) → Decompress → Precision Rerank → Top-K
//! ```
//!
//! # Search Modes
//!
//! - `Fast`: Pure compressed search (~70% recall, minimal latency)
//! - `Balanced`: Top-100 candidates → Rerank (~96% recall)
//! - `Precision`: Top-300 candidates → Rerank (~99% recall)
//! - `Custom`: Configure candidates and rerank level
//!
//! # Decompression Stages (Modular)
//!
//! Stage 1: Storage → AQEA (e.g., PQ 64B → 128D)
//! Stage 2: AQEA → Rerank space (e.g., 128D → 2048D)
//!
//! CONSTRAINT: rerank_dim <= original_dim (never larger!)

use crate::compression::{OctonionCompressor, Compressor};
use crate::decompression::{
    ProjectionDecompressor, LinearDecompressor, PQDecompressor,
    Decompressor,
};
use crate::pq::ProductQuantizer;
use crate::metrics::cosine_similarity;
use serde::{Serialize, Deserialize};
use std::time::Instant;

// ============================================================================
// MODULAR DECOMPRESSION STAGES
// ============================================================================

/// Decompression stage types for modular Precision Boost
///
/// These can be combined in sequence:
/// - Stage 1: Storage format → AQEA space (e.g., PQ → 128D)
/// - Stage 2: AQEA space → Rerank space (e.g., 128D → 2048D)
#[derive(Clone, Debug)]
pub enum DecompressionStageType {
    /// PQ Codebook Lookup (non-linear, for PQ codes → AQEA)
    /// Use this for Stage 1 when storage is PQ codes
    PQ,
    
    /// Linear Projection (W^T @ y, fast approximation)
    /// Use this for Stage 2 to go from AQEA → ~Original
    Projection,
    
    /// Simple Linear Replication (baseline, low quality)
    Linear,
    
    /// No decompression (stay in current space)
    None,
}

/// Configuration for a decompression stage
#[derive(Clone, Debug)]
pub struct DecompressionStageConfig {
    /// Type of decompression
    pub stage_type: DecompressionStageType,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension (MUST be <= original_dim!)
    pub output_dim: usize,
}

impl DecompressionStageConfig {
    /// Create a new stage config with validation
    ///
    /// # Panics
    /// Panics if output_dim > original_dim (constraint violation)
    pub fn new(stage_type: DecompressionStageType, input_dim: usize, output_dim: usize, original_dim: usize) -> Self {
        assert!(
            output_dim <= original_dim,
            "Rerank dimension ({}) cannot exceed original dimension ({})!",
            output_dim, original_dim
        );
        Self { stage_type, input_dim, output_dim }
    }
}

/// Modular Precision Boost configuration
#[derive(Clone, Debug)]
pub struct ModularBoostConfig {
    /// Original dimension (before any compression)
    pub original_dim: usize,
    
    /// Storage dimension (compressed, e.g., PQ codes = n_subvectors)
    pub storage_dim: usize,
    
    /// Stage 1: Storage → AQEA (optional, e.g., PQ decode)
    pub stage1: Option<DecompressionStageConfig>,
    
    /// Stage 2: AQEA → Rerank space (optional, e.g., Projection)
    pub stage2: Option<DecompressionStageConfig>,
    
    /// Final rerank dimension (MUST be <= original_dim!)
    pub rerank_dim: usize,
    
    /// Number of candidates to fetch for reranking
    pub candidates: usize,
}

impl ModularBoostConfig {
    /// Validate configuration constraints
    pub fn validate(&self) -> Result<(), String> {
        if self.rerank_dim > self.original_dim {
            return Err(format!(
                "Rerank dimension ({}) cannot exceed original dimension ({})!",
                self.rerank_dim, self.original_dim
            ));
        }
        
        if let Some(ref stage1) = self.stage1 {
            if stage1.output_dim > self.original_dim {
                return Err(format!(
                    "Stage 1 output ({}) cannot exceed original dimension ({})!",
                    stage1.output_dim, self.original_dim
                ));
            }
        }
        
        if let Some(ref stage2) = self.stage2 {
            if stage2.output_dim > self.original_dim {
                return Err(format!(
                    "Stage 2 output ({}) cannot exceed original dimension ({})!",
                    stage2.output_dim, self.original_dim
                ));
            }
        }
        
        Ok(())
    }
    
    /// Create a PQ Boost config (PQ → AQEA, rerank in AQEA space)
    pub fn pq_boost(original_dim: usize, aqea_dim: usize, pq_subvectors: usize, candidates: usize) -> Self {
        Self {
            original_dim,
            storage_dim: pq_subvectors,
            stage1: Some(DecompressionStageConfig {
                stage_type: DecompressionStageType::PQ,
                input_dim: pq_subvectors,
                output_dim: aqea_dim,
            }),
            stage2: None,
            rerank_dim: aqea_dim,
            candidates,
        }
    }
    
    /// Create a Turbo Boost config (PQ → AQEA → ~Original)
    pub fn turbo_boost(original_dim: usize, aqea_dim: usize, pq_subvectors: usize, rerank_dim: usize, candidates: usize) -> Self {
        assert!(rerank_dim <= original_dim, "Rerank dim must be <= original dim!");
        Self {
            original_dim,
            storage_dim: pq_subvectors,
            stage1: Some(DecompressionStageConfig {
                stage_type: DecompressionStageType::PQ,
                input_dim: pq_subvectors,
                output_dim: aqea_dim,
            }),
            stage2: Some(DecompressionStageConfig {
                stage_type: DecompressionStageType::Projection,
                input_dim: aqea_dim,
                output_dim: rerank_dim,
            }),
            rerank_dim,
            candidates,
        }
    }
}

// ============================================================================
// SEARCH MODES (Original API)
// ============================================================================

/// Search modes for AQEA similarity search
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum SearchMode {
    /// Fast: Pure compressed search (lowest latency, ~70% recall at 50D)
    Fast,

    /// Balanced: Top-100 candidates → Rerank in higher dim (~96% recall)
    Balanced,

    /// Precision: Top-300 candidates → Rerank (~99% recall)
    Precision,

    /// Custom configuration
    Custom {
        /// Number of candidates to fetch from compressed index
        candidates: usize,
        /// Rerank level: "original", "aqea_128d", "aqea_70d"
        rerank_level: String,
    },
}

impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::Balanced
    }
}

impl SearchMode {
    /// Get the number of candidates to scan for this mode
    pub fn candidates(&self) -> usize {
        match self {
            SearchMode::Fast => 10, // Just return top-K directly
            SearchMode::Balanced => 100,
            SearchMode::Precision => 300,
            SearchMode::Custom { candidates, .. } => *candidates,
        }
    }

    /// Get the rerank level for this mode
    pub fn rerank_level(&self) -> &str {
        match self {
            SearchMode::Fast => "none",
            SearchMode::Balanced => "aqea_128d",
            SearchMode::Precision => "aqea_128d",
            SearchMode::Custom { rerank_level, .. } => rerank_level,
        }
    }
}

/// A single search result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Index in the database
    pub index: usize,
    /// Optional ID (if provided)
    pub id: Option<String>,
    /// Similarity score (cosine similarity)
    pub similarity: f32,
    /// Original rank in compressed search (before rerank)
    pub original_rank: Option<usize>,
}

/// Search response with metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Top-K results
    pub results: Vec<SearchResult>,
    /// Total search latency in milliseconds
    pub latency_ms: f32,
    /// Search mode used
    pub mode: SearchMode,
    /// Number of candidates scanned in compressed space
    pub candidates_scanned: usize,
    /// Number of candidates reranked (if applicable)
    pub candidates_reranked: usize,
}

/// Configuration for the searcher
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Compressed dimension for fast search (e.g., 50D for 41× compression)
    pub compressed_dim: usize,
    /// Rerank dimension (e.g., 128D for 16× compression)
    pub rerank_dim: usize,
    /// Original dimension
    pub original_dim: usize,
}

/// Precision Boost Searcher - Hybrid compressed + rerank search
///
/// Stores vectors at high compression (e.g., 50D) for fast scanning,
/// then decompresses candidates and reranks in higher-precision space.
pub struct PrecisionBoostSearcher {
    /// Configuration
    config: SearchConfig,
    /// Compressor for fast search (high compression, e.g., 50D)
    fast_compressor: OctonionCompressor,
    /// Compressor for reranking (higher precision, e.g., 128D)
    rerank_compressor: OctonionCompressor,
    /// Decompressor from fast to rerank space
    decompressor: ProjectionDecompressor,
    /// Compressed database (fast search space)
    compressed_db: Vec<Vec<f32>>,
    /// IDs (optional)
    ids: Vec<Option<String>>,
}

impl PrecisionBoostSearcher {
    /// Create a new searcher with the given compressors
    ///
    /// # Arguments
    /// * `fast_compressor` - Compressor for database storage (high compression)
    /// * `rerank_compressor` - Compressor for reranking (higher precision)
    pub fn new(
        fast_compressor: OctonionCompressor,
        rerank_compressor: OctonionCompressor,
    ) -> Self {
        let config = SearchConfig {
            compressed_dim: fast_compressor.compressed_dim(),
            rerank_dim: rerank_compressor.compressed_dim(),
            original_dim: fast_compressor.original_dim(),
        };

        // Create decompressor from fast to rerank space
        // This uses the transpose of the fast compression weights
        let decompressor = ProjectionDecompressor::from_compression_weights(
            fast_compressor.weights()
        );

        Self {
            config,
            fast_compressor,
            rerank_compressor,
            decompressor,
            compressed_db: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// Build index from original vectors
    ///
    /// # Arguments
    /// * `vectors` - Original high-dimensional vectors
    /// * `ids` - Optional IDs for each vector
    pub fn build_index(&mut self, vectors: &[Vec<f32>], ids: Option<&[String]>) {
        // Compress all vectors to fast search space
        self.compressed_db = vectors
            .iter()
            .map(|v| {
                let compressed = self.fast_compressor.compress(v);
                normalize(&compressed)
            })
            .collect();

        // Store IDs
        self.ids = if let Some(id_list) = ids {
            id_list.iter().map(|s| Some(s.clone())).collect()
        } else {
            vec![None; vectors.len()]
        };
    }

    /// Search for similar vectors
    ///
    /// # Arguments
    /// * `query` - Query vector (original dimension)
    /// * `k` - Number of results to return
    /// * `mode` - Search mode
    pub fn search(&self, query: &[f32], k: usize, mode: SearchMode) -> SearchResponse {
        let start = Instant::now();

        // Step 1: Compress query to fast search space
        let query_compressed = normalize(&self.fast_compressor.compress(query));

        // Step 2: Compute similarities in compressed space and get top candidates
        let num_candidates = mode.candidates().max(k);
        let mut scores: Vec<(usize, f32)> = self.compressed_db
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, cosine_similarity(&query_compressed, v)))
            .collect();

        // Sort by similarity (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get top candidates
        let candidates: Vec<(usize, f32)> = scores.into_iter().take(num_candidates).collect();
        let candidates_scanned = candidates.len();

        // Step 3: Rerank if not Fast mode
        let results = if mode == SearchMode::Fast {
            // No reranking - return compressed search results directly
            candidates
                .into_iter()
                .take(k)
                .enumerate()
                .map(|(rank, (idx, sim))| SearchResult {
                    index: idx,
                    id: self.ids.get(idx).cloned().flatten(),
                    similarity: sim,
                    original_rank: Some(rank),
                })
                .collect()
        } else {
            // Rerank in higher-precision space
            let query_rerank = normalize(&self.rerank_compressor.compress(query));

            let mut reranked: Vec<(usize, f32, usize)> = candidates
                .into_iter()
                .enumerate()
                .map(|(original_rank, (idx, _compressed_sim))| {
                    // Decompress from fast space and recompress to rerank space
                    let decompressed = self.decompressor.decompress(&self.compressed_db[idx]);
                    let rerank_vec = normalize(&self.rerank_compressor.compress(&decompressed));
                    let rerank_sim = cosine_similarity(&query_rerank, &rerank_vec);
                    (idx, rerank_sim, original_rank)
                })
                .collect();

            // Sort by reranked similarity
            reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            reranked
                .into_iter()
                .take(k)
                .map(|(idx, sim, original_rank)| SearchResult {
                    index: idx,
                    id: self.ids.get(idx).cloned().flatten(),
                    similarity: sim,
                    original_rank: Some(original_rank),
                })
                .collect()
        };

        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;

        SearchResponse {
            results,
            latency_ms,
            mode: mode.clone(),
            candidates_scanned,
            candidates_reranked: if mode == SearchMode::Fast { 0 } else { candidates_scanned },
        }
    }

    /// Get database size
    pub fn len(&self) -> usize {
        self.compressed_db.len()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.compressed_db.is_empty()
    }

    /// Get configuration
    pub fn config(&self) -> &SearchConfig {
        &self.config
    }
}

/// Normalize a vector to unit length
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Compute recall with reranking - for benchmarking
///
/// Given ground truth top-K, compute recall after reranking.
///
/// # Arguments
/// * `compressed_rankings` - Rankings from compressed search
/// * `ground_truth` - True top-K from original space
/// * `k` - Number of results to evaluate
/// * `candidates` - Number of candidates to rerank
pub fn compute_recall_with_rerank(
    compressed_rankings: &[usize],
    ground_truth: &[usize],
    k: usize,
    candidates: usize,
) -> f32 {
    // Get top candidates from compressed search
    let candidate_set: std::collections::HashSet<_> = compressed_rankings
        .iter()
        .take(candidates)
        .cloned()
        .collect();

    // Count how many ground truth items are in candidate set
    let hits = ground_truth
        .iter()
        .take(k)
        .filter(|gt| candidate_set.contains(*gt))
        .count();

    hits as f32 / k as f32
}

// ============================================================================
// MODULAR BOOST SEARCHER (with configurable decompression stages)
// ============================================================================

/// Storage format for the index
#[derive(Clone, Debug)]
pub enum StorageFormat {
    /// AQEA Float vectors (e.g., 128D × f32)
    AqeaFloat(Vec<Vec<f32>>),
    /// PQ codes (e.g., 64 × u8)
    PQCodes(Vec<Vec<u8>>),
}

/// Modular Precision Boost Searcher
///
/// Supports configurable decompression stages for benchmarking different
/// combinations of PQ decode, Projection, and other decompression methods.
pub struct ModularBoostSearcher {
    /// Configuration
    config: ModularBoostConfig,
    
    /// Storage (either AQEA float or PQ codes)
    storage: StorageFormat,
    
    /// PQ decompressor (for Stage 1 PQ decode)
    pq_decompressor: Option<PQDecompressor>,
    
    /// Projection decompressor (for Stage 2)
    projection_decompressor: Option<ProjectionDecompressor>,
    
    /// Compressor for query (to AQEA space)
    query_compressor: Option<OctonionCompressor>,
    
    /// IDs (optional)
    ids: Vec<Option<String>>,
}

impl ModularBoostSearcher {
    /// Create a new modular searcher for PQ-based storage
    ///
    /// # Arguments
    /// * `config` - Modular boost configuration
    /// * `pq` - Trained ProductQuantizer (for PQ decode)
    /// * `aqea_compressor` - AQEA compressor (for query compression)
    /// * `projection_weights` - Optional weights for Stage 2 projection
    pub fn new_pq(
        config: ModularBoostConfig,
        pq: &ProductQuantizer,
        aqea_compressor: OctonionCompressor,
        projection_weights: Option<&[Vec<f32>]>,
    ) -> Result<Self, String> {
        config.validate()?;
        
        let pq_decompressor = Some(PQDecompressor::from_pq(pq));
        
        let projection_decompressor = projection_weights.map(|weights| {
            ProjectionDecompressor::from_compression_weights(weights)
        });
        
        Ok(Self {
            config,
            storage: StorageFormat::PQCodes(Vec::new()),
            pq_decompressor,
            projection_decompressor,
            query_compressor: Some(aqea_compressor),
            ids: Vec::new(),
        })
    }
    
    /// Build index from original vectors
    ///
    /// Compresses vectors to AQEA, then to PQ codes for storage.
    pub fn build_index(
        &mut self,
        vectors: &[Vec<f32>],
        pq: &ProductQuantizer,
        aqea_compressor: &OctonionCompressor,
        ids: Option<&[String]>,
    ) {
        // Compress to AQEA, then encode to PQ
        let pq_codes: Vec<Vec<u8>> = vectors
            .iter()
            .map(|v| {
                let aqea = normalize(&aqea_compressor.compress(v));
                pq.encode(&aqea)
            })
            .collect();
        
        self.storage = StorageFormat::PQCodes(pq_codes);
        
        // Store IDs
        self.ids = if let Some(id_list) = ids {
            id_list.iter().map(|s| Some(s.clone())).collect()
        } else {
            vec![None; vectors.len()]
        };
    }
    
    /// Decompress a stored vector through configured stages
    fn decompress_to_rerank(&self, idx: usize) -> Vec<f32> {
        match &self.storage {
            StorageFormat::PQCodes(codes) => {
                // Stage 1: PQ decode → AQEA
                let aqea = if let Some(ref pq_dec) = self.pq_decompressor {
                    pq_dec.decompress_codes(&codes[idx])
                } else {
                    // Fallback: convert codes to f32
                    codes[idx].iter().map(|&c| c as f32).collect()
                };
                
                // Stage 2: AQEA → Rerank space (if configured)
                if let Some(ref proj_dec) = self.projection_decompressor {
                    if self.config.stage2.is_some() {
                        return proj_dec.decompress(&aqea);
                    }
                }
                
                aqea
            },
            StorageFormat::AqeaFloat(vecs) => {
                // Already in AQEA space, apply Stage 2 if configured
                if let Some(ref proj_dec) = self.projection_decompressor {
                    if self.config.stage2.is_some() {
                        return proj_dec.decompress(&vecs[idx]);
                    }
                }
                vecs[idx].clone()
            },
        }
    }
    
    /// Search with modular decompression stages
    ///
    /// # Arguments
    /// * `query` - Query vector (original dimension)
    /// * `k` - Number of results to return
    pub fn search(&self, query: &[f32], k: usize) -> SearchResponse {
        let start = Instant::now();
        
        // Compress query to AQEA space
        let query_aqea = if let Some(ref compressor) = self.query_compressor {
            normalize(&compressor.compress(query))
        } else {
            normalize(query)
        };
        
        // Fast search: compare query AQEA with PQ-decoded AQEA
        let db_size = match &self.storage {
            StorageFormat::PQCodes(codes) => codes.len(),
            StorageFormat::AqeaFloat(vecs) => vecs.len(),
        };
        
        // Get top candidates from fast search (in AQEA space)
        let mut scores: Vec<(usize, f32)> = (0..db_size)
            .map(|idx| {
                // For fast search, decode PQ to AQEA
                let db_aqea = match &self.storage {
                    StorageFormat::PQCodes(codes) => {
                        if let Some(ref pq_dec) = self.pq_decompressor {
                            normalize(&pq_dec.decompress_codes(&codes[idx]))
                        } else {
                            vec![0.0; self.config.rerank_dim]
                        }
                    },
                    StorageFormat::AqeaFloat(vecs) => normalize(&vecs[idx]),
                };
                (idx, cosine_similarity(&query_aqea, &db_aqea))
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let candidates: Vec<(usize, f32)> = scores.into_iter()
            .take(self.config.candidates)
            .collect();
        let candidates_scanned = candidates.len();
        
        // Rerank in configured space
        let query_rerank = if self.config.stage2.is_some() {
            // Decompress query to rerank space
            if let Some(ref proj_dec) = self.projection_decompressor {
                normalize(&proj_dec.decompress(&query_aqea))
            } else {
                query_aqea.clone()
            }
        } else {
            query_aqea
        };
        
        let mut reranked: Vec<(usize, f32, usize)> = candidates
            .into_iter()
            .enumerate()
            .map(|(original_rank, (idx, _fast_sim))| {
                let rerank_vec = normalize(&self.decompress_to_rerank(idx));
                let sim = cosine_similarity(&query_rerank, &rerank_vec);
                (idx, sim, original_rank)
            })
            .collect();
        
        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let results: Vec<SearchResult> = reranked
            .into_iter()
            .take(k)
            .map(|(idx, sim, original_rank)| SearchResult {
                index: idx,
                id: self.ids.get(idx).cloned().flatten(),
                similarity: sim,
                original_rank: Some(original_rank),
            })
            .collect();
        
        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
        
        SearchResponse {
            results,
            latency_ms,
            mode: SearchMode::Custom {
                candidates: self.config.candidates,
                rerank_level: format!("modular_{}d", self.config.rerank_dim),
            },
            candidates_scanned,
            candidates_reranked: candidates_scanned,
        }
    }
    
    /// Get database size
    pub fn len(&self) -> usize {
        match &self.storage {
            StorageFormat::PQCodes(codes) => codes.len(),
            StorageFormat::AqeaFloat(vecs) => vecs.len(),
        }
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get configuration
    pub fn config(&self) -> &ModularBoostConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_mode_candidates() {
        assert_eq!(SearchMode::Fast.candidates(), 10);
        assert_eq!(SearchMode::Balanced.candidates(), 100);
        assert_eq!(SearchMode::Precision.candidates(), 300);

        let custom = SearchMode::Custom {
            candidates: 50,
            rerank_level: "aqea_128d".to_string(),
        };
        assert_eq!(custom.candidates(), 50);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_compute_recall_with_rerank() {
        // Ground truth: [0, 1, 2, 3, 4]
        // Compressed search: [0, 2, 4, 6, 8, 1, 3, ...]
        // With candidates=5, we get [0, 2, 4, 6, 8]
        // For K=5 ground truth, we find 3 of 5 = 60% recall
        let compressed = vec![0, 2, 4, 6, 8, 1, 3, 5, 7, 9];
        let ground_truth = vec![0, 1, 2, 3, 4];

        let recall = compute_recall_with_rerank(&compressed, &ground_truth, 5, 5);
        assert!((recall - 0.6).abs() < 1e-6);

        // With more candidates, recall improves
        let recall = compute_recall_with_rerank(&compressed, &ground_truth, 5, 10);
        assert!((recall - 1.0).abs() < 1e-6); // All found
    }
}
