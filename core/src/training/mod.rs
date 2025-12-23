//! Auto-Training Module for AQEA
//!
//! Provides intelligent training with:
//! - Automatic 80/20 train/validation split
//! - Progressive sample size increase
//! - Early stopping on validation stagnation
//! - Overfitting detection
//! - Training reports with recommendations
//! - Stratified sampling for uniform space coverage
//!
//! # Example
//!
//! ```rust,ignore
//! use aqea_core::training::{AutoTrainer, AutoTrainerConfig};
//!
//! let config = AutoTrainerConfig::default();
//! let trainer = AutoTrainer::new(config);
//!
//! let result = trainer.train(&embeddings1, &embeddings2)?;
//! println!("Optimal samples: {}", result.optimal_sample_size);
//! println!("Final validation score: {:.2}%", result.final_val_score * 100.0);
//! ```

mod config;
mod splitter;
mod early_stopping;
mod progressive;
mod report;
mod auto_trainer;
pub mod sampling;
pub mod loader;

// Re-exports
pub use config::{AutoTrainerConfig, SampleProgression};
pub use splitter::{DataSplitter, SplitData};
pub use early_stopping::{EarlyStopping, EarlyStopReason, OverfitDetector, OverfitStatus};
pub use progressive::{ProgressiveTrainer, ProgressStep};
pub use report::{TrainingReport, ProgressionEntry, StopReason, Recommendation};
pub use auto_trainer::{AutoTrainer, AutoTrainResult};

// Sampling re-exports
pub use sampling::{Sampler, SamplingStrategy, RandomSampler, KMeansSampler, TsneGridSampler, create_sampler};
