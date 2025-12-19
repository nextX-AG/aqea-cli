//! AQEA Core - Compression Engine
//!
//! This crate provides the core compression and decompression algorithms
//! for AQEA Compression™ technology.
//!
//! # Features
//!
//! - **3-Stage Pipeline**: 8-bit → AQEA → PQ (up to 1200x compression)
//! - **OctonionCompressor**: Block-based aggregation with golden ratio weighting
//! - **PreQuantifier**: Rotation preprocessing for improved similarity preservation
//! - **ProductQuantizer**: Subvector quantization for extreme compression
//! - **BinaryWeights**: Efficient .aqwt format for weight storage
//! - **Metrics**: Cosine similarity, Spearman correlation, MSE, etc.
//!
//! # Pipeline Modes
//!
//! ```rust,ignore
//! use aqea_core::pipeline::{CompressionPipeline, PipelineMode};
//!
//! // AQEA only (29x compression, 95%+ quality)
//! let pipeline = CompressionPipeline::aqea_only(compressor);
//!
//! // 8-bit + AQEA (29x, same quality)
//! let pipeline = CompressionPipeline::with_8bit(compressor);
//!
//! // Full 3-stage (150-1200x, requires trained PQ)
//! let pipeline = CompressionPipeline::full(compressor, pq);
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use aqea_core::{OctonionCompressor, Compressor, BinaryWeights};
//!
//! // Load trained weights
//! let weights = BinaryWeights::load("weights/aqea_mistral_1024d.aqwt")?;
//! let compressor = OctonionCompressor::with_trained_weights(
//!     weights.original_dim as usize,
//!     weights.compressed_dim as usize,
//!     &weights.weights,
//! );
//!
//! // Compress embedding
//! let embedding: Vec<f32> = vec![0.1; 1024];
//! let compressed = compressor.compress(&embedding);
//! assert_eq!(compressed.len(), 35);
//! ```

// Core modules
pub mod compression;
pub mod decompression;
pub mod prequantify;
pub mod metrics;
pub mod energy;
pub mod constants;
pub mod weights_binary;
pub mod cmaes;

// 3-Stage Pipeline modules
pub mod quantization;
pub mod pq;
pub mod pipeline;

// Search module (Precision Boost)
pub mod search;

// Re-exports for convenience
pub use compression::{OctonionCompressor, Compressor};
pub use decompression::{ProjectionDecompressor, PQDecompressor, Decompressor};
pub use prequantify::{PreQuantifier, PreQuantifyConfig};
pub use weights_binary::{BinaryWeights, ModelType, WeightsError};
pub use metrics::{cosine_similarity, spearman_correlation, ValidationMetrics};
pub use cmaes::CMAES;

// 3-Stage Pipeline re-exports
pub use quantization::{quantize_8bit, dequantize_8bit, EightBitQuantizer, QuantizedEmbedding};
pub use pq::{ProductQuantizer, PQConfig, PQMode};
pub use pipeline::{CompressionPipeline, PipelineMode, CompressedOutput, CompressedData};

// Search re-exports (Precision Boost)
pub use search::{SearchMode, SearchResult, SearchResponse, SearchConfig, PrecisionBoostSearcher, compute_recall_with_rerank};



