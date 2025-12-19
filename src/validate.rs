//! User Self-Validation Module
//!
//! Allows users to validate AQEA compression on their OWN data.
//! This is the key for trust: users don't need our training data.
//!
//! # Usage
//! ```bash
//! # User provides their own embeddings and similarity scores
//! aqea validate \
//!   --embeddings my_embeddings.json \
//!   --model text-mpnet
//!
//! # Or with custom similarity pairs
//! aqea validate \
//!   --pairs my_pairs.json \
//!   --model text-mpnet
//! ```
//!
//! # Formats
//!
//! ## Embeddings Format (JSON)
//! ```json
//! {
//!   "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
//! }
//! ```
//!
//! ## Pairs Format (JSON) - with optional human scores
//! ```json
//! {
//!   "pairs": [
//!     {"embedding1": [...], "embedding2": [...], "score": 0.85},
//!     ...
//!   ]
//! }
//! ```

use aqea_core::{
    OctonionCompressor, Compressor, BinaryWeights,
    cosine_similarity, spearman_correlation,
    CompressionPipeline, ProductQuantizer,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// User-provided embeddings format
#[derive(Debug, Deserialize)]
pub struct UserEmbeddings {
    pub embeddings: Vec<Vec<f32>>,
}

/// User-provided pairs format
#[derive(Debug, Deserialize)]
pub struct UserPairs {
    pub pairs: Option<Vec<EmbeddingPair>>,
    // Alternative flat format
    pub embeddings1: Option<Vec<Vec<f32>>>,
    pub embeddings2: Option<Vec<Vec<f32>>>,
    pub scores: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingPair {
    pub embedding1: Vec<f32>,
    pub embedding2: Vec<f32>,
    pub score: Option<f32>,
}

/// Validation result
#[derive(Debug, Serialize)]
pub struct ValidationResult {
    pub status: String,
    pub n_samples: usize,
    pub original_dim: usize,
    pub compressed_dim: usize,
    pub compression_ratio: f32,
    
    // Quality metrics
    pub similarity_preservation: f32,  // Spearman(orig_sim, comp_sim)
    pub task_preservation: Option<f32>, // If user provided scores
    
    // Detailed results per mode
    pub modes: Vec<ModeResult>,
}

#[derive(Debug, Serialize)]
pub struct ModeResult {
    pub mode: String,
    pub compression_ratio: f32,
    pub bytes_per_embedding: usize,
    pub similarity_preservation: f32,
    pub task_preservation: Option<f32>,
}

/// Run validation on user's data
pub fn run_validation(
    weights_path: &Path,
    data_path: &Path,
    pq_subvectors: Option<usize>,
    verbose: bool,
) -> Result<ValidationResult, String> {
    // Load weights
    if verbose {
        println!("📦 Loading weights: {}", weights_path.display());
    }
    let weights = BinaryWeights::load(weights_path)
        .map_err(|e| format!("Failed to load weights: {}", e))?;
    
    let original_dim = weights.original_dim as usize;
    let compressed_dim = weights.compressed_dim as usize;
    
    if verbose {
        println!("  Compression: {}D → {}D", original_dim, compressed_dim);
    }
    
    // Load user data
    if verbose {
        println!("📊 Loading your data: {}", data_path.display());
    }
    let (emb1, emb2, user_scores) = load_user_data(data_path)?;
    
    let n_pairs = emb1.len();
    if n_pairs == 0 {
        return Err("No embedding pairs found".to_string());
    }
    
    // Verify dimensions
    let data_dim = emb1.first().map(|e| e.len()).unwrap_or(0);
    if data_dim != original_dim {
        return Err(format!(
            "Dimension mismatch: Your data is {}D, but model expects {}D",
            data_dim, original_dim
        ));
    }
    
    let has_scores = !user_scores.is_empty();
    if verbose {
        println!("  Pairs: {}", n_pairs);
        println!("  Human scores: {}", if has_scores { "✅ Provided" } else { "❌ Not provided" });
    }
    
    // Create compressor
    let compressor = Arc::new(OctonionCompressor::with_trained_weights(
        original_dim,
        compressed_dim,
        &weights.weights,
    ));
    
    // Compute original similarities
    let original_sims: Vec<f32> = emb1.iter()
        .zip(emb2.iter())
        .map(|(e1, e2)| cosine_similarity(e1, e2))
        .collect();
    
    // Original task performance (if scores provided)
    let original_task = if has_scores {
        Some(spearman_correlation(&original_sims, &user_scores))
    } else {
        None
    };
    
    if verbose {
        println!();
        println!("🔬 Running validation...");
    }
    
    let original_bytes = original_dim * 4;
    let mut modes = Vec::new();
    
    // Test AQEA Only
    let aqea_pipeline = CompressionPipeline::aqea_only(compressor.clone());
    let aqea_result = test_mode(&aqea_pipeline, &emb1, &emb2, &original_sims, 
                                 &user_scores, original_task, "AQEA Only", original_bytes, verbose);
    modes.push(aqea_result);
    
    // Test Full Pipeline if PQ subvectors specified
    if let Some(pq_subs) = pq_subvectors {
        // Train PQ on user's data
        if verbose {
            println!("  Training PQ ({} subvectors) on your data...", pq_subs);
        }
        let training_samples = 1000.min(n_pairs);
        let aqea_outputs: Vec<Vec<f32>> = emb1.iter()
            .take(training_samples)
            .map(|e| compressor.compress(e))
            .collect();
        
        let mut pq = ProductQuantizer::new(compressed_dim, pq_subs, 8);
        pq.train(&aqea_outputs, 20);
        
        let full_pipeline = CompressionPipeline::full(compressor.clone(), pq);
        let full_result = test_mode(&full_pipeline, &emb1, &emb2, &original_sims,
                                     &user_scores, original_task, "Full 3-Stage", original_bytes, verbose);
        modes.push(full_result);
    }
    
    // Best result
    let best = modes.iter().max_by(|a, b| 
        a.similarity_preservation.partial_cmp(&b.similarity_preservation).unwrap()
    ).unwrap();
    
    Ok(ValidationResult {
        status: "success".to_string(),
        n_samples: n_pairs,
        original_dim,
        compressed_dim,
        compression_ratio: best.compression_ratio,
        similarity_preservation: best.similarity_preservation,
        task_preservation: best.task_preservation,
        modes,
    })
}

fn test_mode(
    pipeline: &CompressionPipeline,
    emb1: &[Vec<f32>],
    emb2: &[Vec<f32>],
    original_sims: &[f32],
    user_scores: &[f32],
    original_task: Option<f32>,
    mode_name: &str,
    original_bytes: usize,
    verbose: bool,
) -> ModeResult {
    // Compress
    let comp1: Vec<Vec<f32>> = emb1.iter()
        .map(|e| pipeline.get_aqea_output(&pipeline.compress(e)))
        .collect();
    let comp2: Vec<Vec<f32>> = emb2.iter()
        .map(|e| pipeline.get_aqea_output(&pipeline.compress(e)))
        .collect();
    
    // Compute compressed similarities
    let comp_sims: Vec<f32> = comp1.iter()
        .zip(comp2.iter())
        .map(|(e1, e2)| cosine_similarity(e1, e2))
        .collect();
    
    // Calculate metrics
    let sim_preservation = spearman_correlation(original_sims, &comp_sims);
    
    let task_preservation = if !user_scores.is_empty() {
        let comp_task = spearman_correlation(&comp_sims, user_scores);
        original_task.map(|ot| comp_task / ot)
    } else {
        None
    };
    
    // Estimate bytes
    let compressed_output = pipeline.compress(&emb1[0]);
    let bytes = compressed_output.data.byte_size();
    let ratio = original_bytes as f32 / bytes as f32;
    
    if verbose {
        let task_str = task_preservation
            .map(|t| format!("{:.1}%", t * 100.0))
            .unwrap_or_else(|| "N/A".to_string());
        println!("  {} │ {:.0}x │ {}B │ SimPres: {:.4} │ TaskPres: {}",
                 mode_name, ratio, bytes, sim_preservation, task_str);
    }
    
    ModeResult {
        mode: mode_name.to_string(),
        compression_ratio: ratio,
        bytes_per_embedding: bytes,
        similarity_preservation: sim_preservation,
        task_preservation,
    }
}

fn load_user_data(path: &Path) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    
    // Try pairs format first
    if let Ok(pairs_data) = serde_json::from_str::<UserPairs>(&content) {
        // Check for flat format
        if let (Some(e1), Some(e2)) = (&pairs_data.embeddings1, &pairs_data.embeddings2) {
            let scores = pairs_data.scores.unwrap_or_default();
            return Ok((e1.clone(), e2.clone(), scores));
        }
        
        // Check for nested pairs format
        if let Some(pairs) = pairs_data.pairs {
            let mut emb1 = Vec::new();
            let mut emb2 = Vec::new();
            let mut scores = Vec::new();
            
            for pair in pairs {
                emb1.push(pair.embedding1);
                emb2.push(pair.embedding2);
                if let Some(score) = pair.score {
                    scores.push(score);
                }
            }
            
            return Ok((emb1, emb2, scores));
        }
    }
    
    // Try embeddings-only format
    if let Ok(emb_data) = serde_json::from_str::<UserEmbeddings>(&content) {
        // Create random pairs for similarity testing
        let n = emb_data.embeddings.len();
        if n < 2 {
            return Err("Need at least 2 embeddings for validation".to_string());
        }
        
        let mut emb1 = Vec::new();
        let mut emb2 = Vec::new();
        
        // Create consecutive pairs
        for i in 0..(n - 1) {
            emb1.push(emb_data.embeddings[i].clone());
            emb2.push(emb_data.embeddings[i + 1].clone());
        }
        // Add wrap-around
        emb1.push(emb_data.embeddings[n - 1].clone());
        emb2.push(emb_data.embeddings[0].clone());
        
        return Ok((emb1, emb2, Vec::new()));
    }
    
    Err("Could not parse data. Expected JSON with 'embeddings1'/'embeddings2' or 'embeddings' array".to_string())
}

/// Print validation summary in a nice format
pub fn print_summary(result: &ValidationResult) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          VALIDATION RESULTS (ON YOUR DATA)                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Samples tested:      {}", result.n_samples);
    println!("  Compression:         {}D → {}D ({:.0}x)", 
             result.original_dim, result.compressed_dim, result.compression_ratio);
    println!();
    println!("  ┌───────────────────┬─────────┬────────┬───────────┬──────────┐");
    println!("  │ Mode              │ Ratio   │ Bytes  │ Sim Pres  │ Task Pres│");
    println!("  ├───────────────────┼─────────┼────────┼───────────┼──────────┤");
    
    for mode in &result.modes {
        let task_str = mode.task_preservation
            .map(|t| format!("{:.1}%", t * 100.0))
            .unwrap_or_else(|| "N/A".to_string());
        
        println!("  │ {:17} │ {:>5.0}x  │ {:>4}B  │ {:.4}    │ {:>8} │",
                 mode.mode, mode.compression_ratio, mode.bytes_per_embedding,
                 mode.similarity_preservation, task_str);
    }
    
    println!("  └───────────────────┴─────────┴────────┴───────────┴──────────┘");
    println!();
    
    // Quality assessment
    let best_sim = result.modes.iter()
        .map(|m| m.similarity_preservation)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    
    let quality = if best_sim >= 0.95 {
        "✅ EXCELLENT - Your embeddings compress very well!"
    } else if best_sim >= 0.90 {
        "✅ GOOD - Quality is well preserved"
    } else if best_sim >= 0.85 {
        "⚠️  ACCEPTABLE - Some quality loss, but usable"
    } else {
        "❌ POOR - Consider using a higher-dimensional model"
    };
    
    println!("  Quality: {}", quality);
    println!();
    println!("  💡 This test ran on YOUR data - you can trust these results!");
}

