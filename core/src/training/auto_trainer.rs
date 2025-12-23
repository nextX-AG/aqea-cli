//! Auto Trainer - Main Training Logic
//!
//! Orchestrates the entire auto-training process:
//! 1. Split data into train/validation
//! 2. Progressive sample size increase
//! 3. Early stopping on stagnation
//! 4. Overfitting detection
//! 5. Report generation
//!
//! Uses Python pycma for fast CMA-ES optimization (10x faster than Rust)

use std::time::Instant;
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use crate::compression::{OctonionCompressor, Compressor};
use crate::metrics::{cosine_similarity, spearman_correlation};
use crate::training::{
    AutoTrainerConfig,
    DataSplitter,
    EarlyStopping,
    OverfitDetector,
    ProgressiveTrainer,
    TrainingReport,
    StopReason,
    sampling::create_sampler,
};

/// Result of auto-training
#[derive(Debug, Clone)]
pub struct AutoTrainResult {
    /// Trained weights
    pub weights: Vec<f32>,
    /// Original dimension
    pub original_dim: usize,
    /// Compressed dimension
    pub compressed_dim: usize,
    /// Optimal sample size found
    pub optimal_sample_size: usize,
    /// Final training score (Spearman)
    pub final_train_score: f64,
    /// Final validation score (Spearman)
    pub final_val_score: f64,
    /// Training report
    pub report: TrainingReport,
}

impl AutoTrainResult {
    /// Create a compressor from the trained weights
    pub fn to_compressor(&self) -> OctonionCompressor {
        OctonionCompressor::with_trained_weights(
            self.original_dim,
            self.compressed_dim,
            &self.weights,
        )
    }
}

/// Auto Trainer
///
/// Automatically trains AQEA weights with optimal sample size detection.
pub struct AutoTrainer {
    config: AutoTrainerConfig,
}

impl AutoTrainer {
    /// Create new AutoTrainer with given configuration
    pub fn new(config: AutoTrainerConfig) -> Self {
        Self { config }
    }

    /// Create AutoTrainer with default configuration
    pub fn default_config() -> Self {
        Self::new(AutoTrainerConfig::default())
    }

    /// Train on paired embedding data
    ///
    /// # Arguments
    /// * `embeddings1` - First set of embeddings (e.g., queries)
    /// * `embeddings2` - Second set of embeddings (e.g., documents)
    ///
    /// # Returns
    /// AutoTrainResult containing trained weights and report
    pub fn train(
        &self,
        embeddings1: &[Vec<f32>],
        embeddings2: &[Vec<f32>],
    ) -> Result<AutoTrainResult, String> {
        let start_time = Instant::now();

        // Validate input
        if embeddings1.is_empty() || embeddings2.is_empty() {
            return Err("Empty embeddings provided".to_string());
        }
        if embeddings1.len() != embeddings2.len() {
            return Err("Embedding counts don't match".to_string());
        }

        let original_dim = embeddings1[0].len();
        let compressed_dim = self.config.compressed_dim;
        let n_total = embeddings1.len();

        if self.config.verbose {
            println!("╔══════════════════════════════════════════════════════════════════════════════╗");
            println!("║                     AQEA AUTO-TRAINING                                       ║");
            println!("╠══════════════════════════════════════════════════════════════════════════════╣");
            println!("║ Dimensions: {}D -> {}D                                                       ║", original_dim, compressed_dim);
            println!("║ Total pairs: {}                                                           ║", n_total);
            println!("╚══════════════════════════════════════════════════════════════════════════════╝");
            println!();
        }

        // Step 1: Split data
        let splitter = DataSplitter::new()
            .with_validation_fraction(self.config.validation_fraction)
            .with_max_validation_size(self.config.max_validation_size)
            .with_seed(self.config.seed.unwrap_or(42));

        let (split1, split2) = splitter.split_pairs(embeddings1, embeddings2);

        if self.config.verbose {
            println!("Data split: {} train, {} validation",
                split1.train_size(), split1.validation_size());
            println!();
        }

        // Compute original similarities for train and validation sets
        let train_original_sims: Vec<f32> = split1.train.iter()
            .zip(split2.train.iter())
            .map(|(e1, e2)| cosine_similarity(e1, e2))
            .collect();

        let val_original_sims: Vec<f32> = split1.validation.iter()
            .zip(split2.validation.iter())
            .map(|(e1, e2)| cosine_similarity(e1, e2))
            .collect();

        // Initialize components
        let mut report = TrainingReport::new(original_dim, compressed_dim, n_total);
        let mut early_stopping = EarlyStopping::new(self.config.patience, self.config.min_improvement);
        let overfit_detector = OverfitDetector::new(
            self.config.overfit_threshold,
            self.config.overfit_stop_threshold,
        );

        let mut progressive = ProgressiveTrainer::new(
            self.config.progression.clone(),
            split1.train_size(),
            self.config.seed.unwrap_or(42),
        );

        // Create sampler based on configuration
        let sampler = create_sampler(
            self.config.sampling_strategy,
            self.config.seed,
        );

        // Initialize weights
        let init_compressor = OctonionCompressor::with_dims(original_dim, compressed_dim);
        let init_weights = init_compressor.get_flat_weights();
        let mut best_weights = init_weights.clone();
        let mut stop_reason = StopReason::Completed;

        if self.config.verbose {
            println!("Starting progressive training...");
            println!("Sampling strategy: {}", self.config.sampling_strategy);
            println!("Sample sizes: {:?}", progressive.sample_sizes());
            println!();
        }

        // Step 2: Progressive training loop
        while let Some(n_samples) = progressive.next_sample_size() {
            let step_start = Instant::now();

            if self.config.verbose {
                println!("┌─────────────────────────────────────────────────────────────────────────────┐");
                println!("│ Training with {} samples (step {}/{})                                      │",
                    n_samples, progressive.current_step(), progressive.total_steps());
                println!("└─────────────────────────────────────────────────────────────────────────────┘");
            }

            // Sample training data using configured sampling strategy
            let indices = sampler.sample(&split1.train, n_samples);
            let sampled_e1: Vec<Vec<f32>> = indices.iter().map(|&i| split1.train[i].clone()).collect();
            let sampled_e2: Vec<Vec<f32>> = indices.iter().map(|&i| split2.train[i].clone()).collect();

            let sampled_original_sims: Vec<f32> = sampled_e1.iter()
                .zip(sampled_e2.iter())
                .map(|(e1, e2)| cosine_similarity(e1, e2))
                .collect();

            // Train with CMA-ES
            let weights = self.train_cmaes(
                &sampled_e1,
                &sampled_e2,
                &sampled_original_sims,
                &best_weights,
                original_dim,
                compressed_dim,
            );

            // Evaluate on train and validation
            let train_score = self.evaluate(
                &split1.train,
                &split2.train,
                &train_original_sims,
                &weights,
                original_dim,
                compressed_dim,
            );

            let val_score = self.evaluate(
                &split1.validation,
                &split2.validation,
                &val_original_sims,
                &weights,
                original_dim,
                compressed_dim,
            );

            let step_time = step_start.elapsed().as_secs_f64();

            // Check overfitting
            let overfit_status = overfit_detector.check(train_score, val_score);

            if self.config.verbose {
                let status_icon = match overfit_status {
                    crate::training::OverfitStatus::Ok => "✓",
                    crate::training::OverfitStatus::Warning => "⚠",
                    crate::training::OverfitStatus::Overfitting => "✗",
                };
                println!("  {} Train: {:.2}%, Val: {:.2}%, Gap: {:.2}%",
                    status_icon,
                    train_score * 100.0,
                    val_score * 100.0,
                    (train_score - val_score) * 100.0
                );
                println!("  Time: {:.1}s", step_time);
                println!();
            }

            // Add to report
            report.add_step(n_samples, train_score, val_score, overfit_status, step_time);

            // Check for overfitting stop
            if overfit_status == crate::training::OverfitStatus::Overfitting {
                if self.config.verbose {
                    println!("⚠️  Overfitting detected! Stopping training.");
                }
                stop_reason = StopReason::OverfittingDetected;
                break;
            }

            // Early stopping check
            if early_stopping.step(val_score, &weights, n_samples) {
                if self.config.verbose {
                    println!("✓ Early stopping triggered (patience exhausted)");
                }
                stop_reason = StopReason::ValidationStagnated;
                break;
            }

            // Update best weights
            best_weights = weights;
        }

        // Get best results from early stopping
        let (final_weights, optimal_samples) = if let Some(weights) = early_stopping.best_weights() {
            (weights.to_vec(), early_stopping.best_sample_size())
        } else {
            (best_weights.clone(), progressive.sample_sizes().last().copied().unwrap_or(0))
        };

        // Final evaluation
        let final_train_score = self.evaluate(
            &split1.train,
            &split2.train,
            &train_original_sims,
            &final_weights,
            original_dim,
            compressed_dim,
        );

        let final_val_score = self.evaluate(
            &split1.validation,
            &split2.validation,
            &val_original_sims,
            &final_weights,
            original_dim,
            compressed_dim,
        );

        let total_time = start_time.elapsed().as_secs_f64();

        // Finalize report
        report.finalize(
            optimal_samples,
            final_train_score,
            final_val_score,
            stop_reason,
            total_time,
        );

        if self.config.verbose {
            println!();
            report.print();
        }

        Ok(AutoTrainResult {
            weights: final_weights,
            original_dim,
            compressed_dim,
            optimal_sample_size: optimal_samples,
            final_train_score,
            final_val_score,
            report,
        })
    }

    /// Train weights using Python pycma (10x faster than Rust CMA-ES!)
    fn train_cmaes(
        &self,
        e1: &[Vec<f32>],
        _e2: &[Vec<f32>],
        _original_sims: &[f32],
        init_weights: &[f32],
        original_dim: usize,
        compressed_dim: usize,
    ) -> Vec<f32> {
        // Try Python training first (much faster)
        match self.train_with_python(e1, original_dim, compressed_dim) {
            Ok(weights) => weights,
            Err(e) => {
                if self.config.verbose {
                    println!("    ⚠️ Python training failed: {}", e);
                    println!("    Falling back to Rust CMA-ES (slower)...");
                }
                self.train_cmaes_rust(e1, original_dim, compressed_dim, init_weights)
            }
        }
    }
    
    /// Train using Python pycma - FAST!
    fn train_with_python(
        &self,
        embeddings: &[Vec<f32>],
        original_dim: usize,
        compressed_dim: usize,
    ) -> Result<Vec<f32>, String> {
        // Find Python script
        let script_paths = vec![
            PathBuf::from("cli/scripts/train_aqea.py"),
            PathBuf::from("/home/aqea/aqea-compress/cli/scripts/train_aqea.py"),
        ];
        
        let script_path = script_paths.iter()
            .find(|p| p.exists())
            .ok_or_else(|| "Python training script not found".to_string())?;
        
        // Find Python with pycma (prefer venv with pycma installed)
        let python_paths = vec![
            "/home/aqea/aqea-compress/benchmark/venv/bin/python",
            "/usr/bin/python3",
            "python3",
        ];
        
        let python = python_paths.iter()
            .find(|p| std::path::Path::new(p).exists())
            .ok_or_else(|| "Python not found".to_string())?;
        
        // Save embeddings to temp file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("aqea_train_{}.json", std::process::id()));
        
        let json_data = serde_json::to_string(embeddings)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;
        std::fs::write(&temp_file, json_data)
            .map_err(|e| format!("Failed to write temp file: {}", e))?;
        
        // Spawn Python process
        let mut child = Command::new(python)
            .arg(script_path)
            .arg(&temp_file)
            .arg(compressed_dim.to_string())
            .arg(self.config.cmaes_generations.to_string())
            .arg("0.97")  // Early stop threshold
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to spawn Python: {}", e))?;
        
        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);
        
        let mut best_weights: Vec<f32> = Vec::new();
        let mut final_spearman = 0.0f32;
        
        // Parse JSON output line by line
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                match json.get("type").and_then(|t| t.as_str()) {
                    Some("progress") => {
                        if self.config.verbose {
                            let gen = json.get("generation").and_then(|v| v.as_i64()).unwrap_or(0);
                            let best = json.get("best_spearman").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let time_ms = json.get("gen_time_ms").and_then(|v| v.as_i64()).unwrap_or(0);
                            println!("    Gen {:3}: Spearman={:.4} ({}ms)", gen, best, time_ms);
                        }
                    }
                    Some("result") => {
                        if let Some(weights) = json.get("weights").and_then(|w| w.as_array()) {
                            best_weights = weights.iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect();
                        }
                        final_spearman = json.get("final_spearman")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                    }
                    Some("error") => {
                        let msg = json.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown error");
                        return Err(format!("Python error: {}", msg));
                    }
                    _ => {}
                }
            }
        }
        
        // Wait for child to finish
        let _ = child.wait();
        
        // Clean up temp file
        let _ = std::fs::remove_file(&temp_file);
        
        if best_weights.is_empty() {
            return Err("No weights returned from Python".to_string());
        }
        
        if self.config.verbose {
            println!("    ✅ Python training complete: Spearman={:.4}", final_spearman);
        }
        
        Ok(best_weights)
    }
    
    /// Fallback Rust CMA-ES (slower but works without Python)
    fn train_cmaes_rust(
        &self,
        embeddings: &[Vec<f32>],
        original_dim: usize,
        compressed_dim: usize,
        init_weights: &[f32],
    ) -> Vec<f32> {
        use crate::cmaes::CMAES;
        use nalgebra::DVector;
        
        let n_weights = original_dim * compressed_dim;
        
        // Create pairs for evaluation
        let pairs: Vec<(usize, usize, f32)> = {
            let n = embeddings.len().min(50);
            let mut pairs = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
                    pairs.push((i, j, sim));
                    if pairs.len() >= 200 { break; }
                }
                if pairs.len() >= 200 { break; }
            }
            pairs
        };
        
        let init_mean = DVector::from_iterator(
            n_weights,
            init_weights.iter().map(|&x| x as f64)
        );

        let mut cmaes = CMAES::new(
            n_weights,
            Some(init_mean),
            0.3,
            self.config.cmaes_population,
        );

        let mut best_weights = init_weights.to_vec();
        let mut best_fitness = f64::INFINITY;

        for gen in 0..self.config.cmaes_generations {
            let solutions = cmaes.ask();

            let fitnesses: Vec<f64> = solutions.iter()
                .map(|sol| {
                    let weights: Vec<f32> = sol.iter().map(|&x| x as f32).collect();
                    let compressor = OctonionCompressor::with_trained_weights(
                        original_dim, compressed_dim, &weights
                    );
                    
                    let compressed_sims: Vec<f32> = pairs.iter()
                        .map(|(i, j, _)| {
                            let c1 = compressor.compress(&embeddings[*i]);
                            let c2 = compressor.compress(&embeddings[*j]);
                            cosine_similarity(&c1, &c2)
                        })
                        .collect();
                    
                    let original_sims: Vec<f32> = pairs.iter().map(|(_, _, s)| *s).collect();
                    let score = spearman_correlation(&compressed_sims, &original_sims);
                    -score as f64
                })
                .collect();

            for (i, &fitness) in fitnesses.iter().enumerate() {
                if fitness < best_fitness {
                    best_fitness = fitness;
                    best_weights = solutions[i].iter().map(|&x| x as f32).collect();
                }
            }

            cmaes.tell(&solutions, &fitnesses);

            if self.config.verbose && (gen + 1) % 5 == 0 {
                println!("    Gen {:3}: Spearman={:.4}", gen + 1, -best_fitness);
            }
        }

        best_weights
    }

    /// Evaluate weights on a dataset
    fn evaluate(
        &self,
        e1: &[Vec<f32>],
        e2: &[Vec<f32>],
        original_sims: &[f32],
        weights: &[f32],
        original_dim: usize,
        compressed_dim: usize,
    ) -> f64 {
        let compressor = OctonionCompressor::with_trained_weights(
            original_dim,
            compressed_dim,
            weights,
        );

        let compressed_sims: Vec<f32> = e1.iter()
            .zip(e2.iter())
            .map(|(e1, e2)| {
                let c1 = compressor.compress(e1);
                let c2 = compressor.compress(e2);
                cosine_similarity(&c1, &c2)
            })
            .collect();

        spearman_correlation(&compressed_sims, original_sims) as f64
    }
}

impl Default for AutoTrainer {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn generate_test_embeddings(n: usize, dim: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = StdRng::seed_from_u64(seed);

        let e1: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();

        // e2 is a noisy version of e1 to simulate related embeddings
        let e2: Vec<Vec<f32>> = e1.iter()
            .map(|v| v.iter().map(|&x| x + rng.gen::<f32>() * 0.2 - 0.1).collect())
            .collect();

        (e1, e2)
    }

    #[test]
    fn test_auto_trainer_creation() {
        let config = AutoTrainerConfig::default();
        let _trainer = AutoTrainer::new(config);
    }

    #[test]
    fn test_auto_trainer_quick() {
        // Quick test with minimal settings
        let config = AutoTrainerConfig::quick()
            .with_compressed_dim(4)
            .quiet();

        let trainer = AutoTrainer::new(config);

        let (e1, e2) = generate_test_embeddings(100, 16, 42);
        let result = trainer.train(&e1, &e2);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.original_dim, 16);
        assert_eq!(result.compressed_dim, 4);
        assert!(!result.weights.is_empty());
    }

    #[test]
    fn test_evaluate() {
        let config = AutoTrainerConfig::default().quiet();
        let trainer = AutoTrainer::new(config);

        let (e1, e2) = generate_test_embeddings(50, 32, 42);
        let original_sims: Vec<f32> = e1.iter()
            .zip(e2.iter())
            .map(|(a, b)| cosine_similarity(a, b))
            .collect();

        // Random weights
        let weights: Vec<f32> = (0..(32 * 8)).map(|i| (i as f32 * 0.01).sin()).collect();

        let score = trainer.evaluate(&e1, &e2, &original_sims, &weights, 32, 8);

        // Score should be between -1 and 1
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_empty_input() {
        let trainer = AutoTrainer::default();
        let result = trainer.train(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_input() {
        let trainer = AutoTrainer::default();
        let e1 = vec![vec![1.0, 2.0, 3.0]];
        let e2 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = trainer.train(&e1, &e2);
        assert!(result.is_err());
    }
}
