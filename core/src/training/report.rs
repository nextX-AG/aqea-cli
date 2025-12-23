//! Training Report Generation
//!
//! Generates comprehensive reports about the auto-training process.

use serde::{Deserialize, Serialize};
use crate::training::early_stopping::{EarlyStopReason, OverfitStatus};

/// Reason training stopped
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Validation score stopped improving
    ValidationStagnated,
    /// Overfitting detected
    OverfittingDetected,
    /// Maximum samples reached
    MaxSamplesReached,
    /// Training completed all steps
    Completed,
}

impl From<EarlyStopReason> for StopReason {
    fn from(reason: EarlyStopReason) -> Self {
        match reason {
            EarlyStopReason::ValidationStagnated => StopReason::ValidationStagnated,
            EarlyStopReason::PatienceExhausted => StopReason::ValidationStagnated,
            EarlyStopReason::Overfitting => StopReason::OverfittingDetected,
            EarlyStopReason::MaxSamplesReached => StopReason::MaxSamplesReached,
            EarlyStopReason::Completed => StopReason::Completed,
        }
    }
}

/// Single entry in the training progression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressionEntry {
    /// Number of samples used
    pub samples: usize,
    /// Training score
    pub train_score: f64,
    /// Validation score
    pub val_score: f64,
    /// Train-validation gap
    pub gap: f64,
    /// Overfitting status at this step
    pub overfit_status: String,
    /// Training time for this step (seconds)
    pub time_secs: f64,
}

/// Training recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation type (info, warning, error)
    pub level: String,
    /// Recommendation message
    pub message: String,
}

impl Recommendation {
    pub fn info(message: impl Into<String>) -> Self {
        Self { level: "info".to_string(), message: message.into() }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self { level: "warning".to_string(), message: message.into() }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self { level: "error".to_string(), message: message.into() }
    }
}

/// Complete training report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    /// Training status (success, warning, failed)
    pub status: String,

    /// Original embedding dimension
    pub original_dim: usize,

    /// Compressed embedding dimension
    pub compressed_dim: usize,

    /// Total number of samples available
    pub total_samples: usize,

    /// Optimal number of training samples found
    pub optimal_samples: usize,

    /// Final training score
    pub final_train_score: f64,

    /// Final validation score
    pub final_val_score: f64,

    /// Final train-val gap
    pub final_gap: f64,

    /// Reason training stopped
    pub stopped_reason: StopReason,

    /// Training progression history
    pub progression: Vec<ProgressionEntry>,

    /// Total training time (seconds)
    pub total_time_secs: f64,

    /// Recommendations and observations
    pub recommendations: Vec<Recommendation>,
}

impl TrainingReport {
    /// Create a new empty report
    pub fn new(original_dim: usize, compressed_dim: usize, total_samples: usize) -> Self {
        Self {
            status: "in_progress".to_string(),
            original_dim,
            compressed_dim,
            total_samples,
            optimal_samples: 0,
            final_train_score: 0.0,
            final_val_score: 0.0,
            final_gap: 0.0,
            stopped_reason: StopReason::Completed,
            progression: Vec::new(),
            total_time_secs: 0.0,
            recommendations: Vec::new(),
        }
    }

    /// Add a progression entry
    pub fn add_step(
        &mut self,
        samples: usize,
        train_score: f64,
        val_score: f64,
        overfit_status: OverfitStatus,
        time_secs: f64,
    ) {
        let gap = train_score - val_score;
        self.progression.push(ProgressionEntry {
            samples,
            train_score,
            val_score,
            gap,
            overfit_status: overfit_status.to_string(),
            time_secs,
        });
    }

    /// Finalize the report
    pub fn finalize(
        &mut self,
        optimal_samples: usize,
        final_train_score: f64,
        final_val_score: f64,
        stopped_reason: StopReason,
        total_time_secs: f64,
    ) {
        self.optimal_samples = optimal_samples;
        self.final_train_score = final_train_score;
        self.final_val_score = final_val_score;
        self.final_gap = final_train_score - final_val_score;
        self.stopped_reason = stopped_reason;
        self.total_time_secs = total_time_secs;

        // Determine status
        if self.final_gap > 0.10 {
            self.status = "failed".to_string();
        } else if self.final_gap > 0.05 {
            self.status = "warning".to_string();
        } else {
            self.status = "success".to_string();
        }

        // Generate recommendations
        self.generate_recommendations();
    }

    /// Generate recommendations based on training results
    fn generate_recommendations(&mut self) {
        self.recommendations.clear();

        // Check final gap
        if self.final_gap < 0.02 {
            self.recommendations.push(Recommendation::info(
                "Training converged well with minimal overfitting"
            ));
        } else if self.final_gap < 0.05 {
            self.recommendations.push(Recommendation::warning(
                format!("Moderate train-val gap ({:.1}%). Consider using fewer samples.", self.final_gap * 100.0)
            ));
        } else {
            self.recommendations.push(Recommendation::error(
                format!("High train-val gap ({:.1}%). Overfitting detected.", self.final_gap * 100.0)
            ));
        }

        // Check final score
        if self.final_val_score > 0.95 {
            self.recommendations.push(Recommendation::info(
                format!("Excellent validation score: {:.1}%", self.final_val_score * 100.0)
            ));
        } else if self.final_val_score > 0.90 {
            self.recommendations.push(Recommendation::info(
                format!("Good validation score: {:.1}%", self.final_val_score * 100.0)
            ));
        } else if self.final_val_score > 0.80 {
            self.recommendations.push(Recommendation::warning(
                format!("Validation score {:.1}% may be improved with more data or different parameters",
                    self.final_val_score * 100.0)
            ));
        } else {
            self.recommendations.push(Recommendation::error(
                format!("Low validation score: {:.1}%. Check data quality.", self.final_val_score * 100.0)
            ));
        }

        // Check stopped reason
        match self.stopped_reason {
            StopReason::OverfittingDetected => {
                self.recommendations.push(Recommendation::warning(
                    format!("Stopped due to overfitting at {} samples. Using fewer samples.", self.optimal_samples)
                ));
            }
            StopReason::ValidationStagnated => {
                self.recommendations.push(Recommendation::info(
                    format!("Optimal training size found: {} samples", self.optimal_samples)
                ));
            }
            StopReason::MaxSamplesReached => {
                self.recommendations.push(Recommendation::info(
                    "Reached maximum sample size without overfitting"
                ));
            }
            StopReason::Completed => {
                self.recommendations.push(Recommendation::info(
                    "Training completed successfully"
                ));
            }
        }

        // Check if more samples might help
        if !matches!(self.stopped_reason, StopReason::OverfittingDetected) {
            if let Some(last) = self.progression.last() {
                if last.samples < self.total_samples / 2 {
                    // Stopped early, more samples might help
                    self.recommendations.push(Recommendation::info(
                        "Early stopping triggered. Current sample size is optimal for this data."
                    ));
                }
            }
        }
    }

    /// Get summary line
    pub fn summary(&self) -> String {
        format!(
            "{}: {} samples, val={:.1}%, gap={:.1}%",
            self.status.to_uppercase(),
            self.optimal_samples,
            self.final_val_score * 100.0,
            self.final_gap * 100.0
        )
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Convert to compact JSON string
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Print human-readable report
    pub fn print(&self) {
        println!("╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                        AUTO-TRAINING REPORT                                  ║");
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        println!("║ Status: {:<68} ║", self.status.to_uppercase());
        println!("║ Dimensions: {}D -> {}D{:>54} ║",
            self.original_dim, self.compressed_dim, "");
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        println!("║ Optimal Samples: {:>59} ║", self.optimal_samples);
        println!("║ Final Train Score: {:>56.2}% ║", self.final_train_score * 100.0);
        println!("║ Final Validation Score: {:>51.2}% ║", self.final_val_score * 100.0);
        println!("║ Train-Val Gap: {:>60.2}% ║", self.final_gap * 100.0);
        println!("║ Total Time: {:>61.1}s ║", self.total_time_secs);
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        println!("║ Progression:                                                                 ║");
        for entry in &self.progression {
            println!("║   {:>6} samples: train={:.2}%, val={:.2}%, gap={:.2}%{:>24} ║",
                entry.samples,
                entry.train_score * 100.0,
                entry.val_score * 100.0,
                entry.gap * 100.0,
                ""
            );
        }
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        println!("║ Recommendations:                                                             ║");
        for rec in &self.recommendations {
            let icon = match rec.level.as_str() {
                "info" => "✓",
                "warning" => "⚠",
                "error" => "✗",
                _ => "•",
            };
            // Truncate message if too long
            let msg = if rec.message.len() > 70 {
                format!("{}...", &rec.message[..67])
            } else {
                rec.message.clone()
            };
            println!("║   {} {:<72} ║", icon, msg);
        }
        println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_creation() {
        let report = TrainingReport::new(1024, 13, 10000);
        assert_eq!(report.original_dim, 1024);
        assert_eq!(report.compressed_dim, 13);
        assert_eq!(report.total_samples, 10000);
    }

    #[test]
    fn test_add_step() {
        let mut report = TrainingReport::new(1024, 13, 10000);
        report.add_step(1000, 0.85, 0.84, OverfitStatus::Ok, 10.5);

        assert_eq!(report.progression.len(), 1);
        assert_eq!(report.progression[0].samples, 1000);
        assert!((report.progression[0].gap - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_finalize_success() {
        let mut report = TrainingReport::new(1024, 13, 10000);
        report.add_step(1000, 0.90, 0.89, OverfitStatus::Ok, 10.0);
        report.add_step(2000, 0.92, 0.91, OverfitStatus::Ok, 15.0);

        report.finalize(2000, 0.92, 0.91, StopReason::ValidationStagnated, 25.0);

        assert_eq!(report.status, "success");
        assert_eq!(report.optimal_samples, 2000);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_finalize_warning() {
        let mut report = TrainingReport::new(1024, 13, 10000);
        report.finalize(5000, 0.95, 0.88, StopReason::OverfittingDetected, 30.0);

        assert_eq!(report.status, "warning");
        assert!((report.final_gap - 0.07).abs() < 1e-9);
    }

    #[test]
    fn test_finalize_failed() {
        let mut report = TrainingReport::new(1024, 13, 10000);
        report.finalize(5000, 0.98, 0.80, StopReason::OverfittingDetected, 30.0);

        assert_eq!(report.status, "failed");
    }

    #[test]
    fn test_to_json() {
        let report = TrainingReport::new(1024, 13, 10000);
        let json = report.to_json().unwrap();

        assert!(json.contains("\"original_dim\": 1024"));
        assert!(json.contains("\"compressed_dim\": 13"));
    }

    #[test]
    fn test_summary() {
        let mut report = TrainingReport::new(1024, 13, 10000);
        report.finalize(5000, 0.92, 0.90, StopReason::Completed, 30.0);

        let summary = report.summary();
        assert!(summary.contains("SUCCESS"));
        assert!(summary.contains("5000"));
    }
}
