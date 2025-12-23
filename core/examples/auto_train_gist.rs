//! Auto-Training Test auf GIST1M
//!
//! Testet das neue Auto-Training Modul auf dem GIST1M Dataset.
//!
//! Run with: cargo run --example auto_train_gist --release

use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;

use aqea_core::training::{AutoTrainer, AutoTrainerConfig};

#[derive(Deserialize)]
struct PairsData {
    embeddings1: Vec<Vec<f32>>,
    embeddings2: Vec<Vec<f32>>,
    scores: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     AUTO-TRAINING TEST - GIST1M (960D)                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load GIST1M training pairs
    let data_path = "proof_data/gist_train_pairs.json";
    println!("📂 Loading data from: {}", data_path);
    
    let file = File::open(data_path)?;
    let reader = BufReader::new(file);
    let data: PairsData = serde_json::from_reader(reader)?;
    
    println!("   Loaded {} pairs", data.embeddings1.len());
    println!("   Dimension: {}D", data.embeddings1[0].len());
    println!();

    // Configure Auto-Training
    let config = AutoTrainerConfig::default()
        .with_compressed_dim(33)  // 960D → 33D (~29x)
        .with_patience(3);
    
    println!("⚙️  Configuration:");
    println!("   Compressed dim: {}D", config.compressed_dim);
    println!("   Patience: {}", config.patience);
    println!();

    // Run Auto-Training
    println!("🚀 Starting Auto-Training...");
    println!();
    
    let trainer = AutoTrainer::new(config);
    let result = trainer.train(&data.embeddings1, &data.embeddings2)?;
    
    // Print Results
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      RESULTS                                      ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Optimal Sample Size: {:>6}                                      ║", result.optimal_sample_size);
    println!("║  Final Train Score:   {:>5.1}%                                      ║", result.final_train_score * 100.0);
    println!("║  Final Val Score:     {:>5.1}%                                      ║", result.final_val_score * 100.0);
    println!("║  Gap:                 {:>5.1}%                                      ║", (result.final_train_score - result.final_val_score) * 100.0);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Print Report
    println!("📊 Training Report:");
    println!("   Stop reason: {:?}", result.report.stopped_reason);
    if !result.report.recommendations.is_empty() {
        println!("   Recommendations:");
        for rec in &result.report.recommendations {
            println!("     - {:?}", rec);
        }
    }

    Ok(())
}

