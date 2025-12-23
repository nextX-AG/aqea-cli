//! Early Stopping and Overfitting Detection
//!
//! Provides mechanisms to:
//! - Stop training when validation score stagnates
//! - Detect overfitting based on train-validation gap

use serde::{Deserialize, Serialize};

/// Reason for early stopping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EarlyStopReason {
    /// Validation score stopped improving
    ValidationStagnated,
    /// Maximum patience reached
    PatienceExhausted,
    /// Overfitting detected (train >> val)
    Overfitting,
    /// Maximum samples reached
    MaxSamplesReached,
    /// Training completed normally
    Completed,
}

impl std::fmt::Display for EarlyStopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EarlyStopReason::ValidationStagnated => write!(f, "validation_stagnated"),
            EarlyStopReason::PatienceExhausted => write!(f, "patience_exhausted"),
            EarlyStopReason::Overfitting => write!(f, "overfitting_detected"),
            EarlyStopReason::MaxSamplesReached => write!(f, "max_samples_reached"),
            EarlyStopReason::Completed => write!(f, "completed"),
        }
    }
}

/// Early stopping controller
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Best validation score seen so far
    best_score: f64,
    /// Weights at best score
    best_weights: Option<Vec<f32>>,
    /// Sample size at best score
    best_sample_size: usize,
    /// Number of steps without improvement
    steps_without_improvement: usize,
    /// Maximum patience
    patience: usize,
    /// Minimum improvement to count as "improving"
    min_improvement: f64,
    /// Whether to maximize or minimize score (true = maximize)
    maximize: bool,
}

impl EarlyStopping {
    /// Create new early stopping controller
    ///
    /// # Arguments
    /// * `patience` - Number of steps without improvement before stopping
    /// * `min_improvement` - Minimum improvement to count as "improving"
    pub fn new(patience: usize, min_improvement: f64) -> Self {
        Self {
            best_score: f64::NEG_INFINITY,
            best_weights: None,
            best_sample_size: 0,
            steps_without_improvement: 0,
            patience,
            min_improvement,
            maximize: true,
        }
    }

    /// Set to minimize instead of maximize
    pub fn minimize(mut self) -> Self {
        self.maximize = false;
        self.best_score = f64::INFINITY;
        self
    }

    /// Record a new validation score
    ///
    /// Returns true if training should stop
    pub fn step(&mut self, score: f64, weights: &[f32], sample_size: usize) -> bool {
        let improved = if self.maximize {
            score > self.best_score + self.min_improvement
        } else {
            score < self.best_score - self.min_improvement
        };

        if improved {
            self.best_score = score;
            self.best_weights = Some(weights.to_vec());
            self.best_sample_size = sample_size;
            self.steps_without_improvement = 0;
        } else {
            self.steps_without_improvement += 1;
        }

        self.steps_without_improvement >= self.patience
    }

    /// Check if should stop (without recording new score)
    pub fn should_stop(&self) -> bool {
        self.steps_without_improvement >= self.patience
    }

    /// Get the reason for stopping
    pub fn stop_reason(&self) -> EarlyStopReason {
        if self.steps_without_improvement >= self.patience {
            EarlyStopReason::PatienceExhausted
        } else {
            EarlyStopReason::Completed
        }
    }

    /// Get best score seen
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    /// Get weights at best score
    pub fn best_weights(&self) -> Option<&[f32]> {
        self.best_weights.as_deref()
    }

    /// Get sample size at best score
    pub fn best_sample_size(&self) -> usize {
        self.best_sample_size
    }

    /// Get current patience state
    pub fn remaining_patience(&self) -> usize {
        self.patience.saturating_sub(self.steps_without_improvement)
    }

    /// Reset the controller
    pub fn reset(&mut self) {
        self.best_score = if self.maximize {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        self.best_weights = None;
        self.best_sample_size = 0;
        self.steps_without_improvement = 0;
    }
}

/// Overfitting status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverfitStatus {
    /// No overfitting detected
    Ok,
    /// Warning: train-val gap is concerning
    Warning,
    /// Overfitting: train >> val
    Overfitting,
}

impl std::fmt::Display for OverfitStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OverfitStatus::Ok => write!(f, "ok"),
            OverfitStatus::Warning => write!(f, "warning"),
            OverfitStatus::Overfitting => write!(f, "overfitting"),
        }
    }
}

/// Overfitting detector
#[derive(Debug, Clone)]
pub struct OverfitDetector {
    /// Threshold for warning (train - val)
    warning_threshold: f64,
    /// Threshold for overfitting (train - val)
    overfit_threshold: f64,
}

impl OverfitDetector {
    /// Create new overfitting detector
    ///
    /// # Arguments
    /// * `warning_threshold` - Gap for warning (e.g., 0.02 = 2%)
    /// * `overfit_threshold` - Gap for overfitting (e.g., 0.05 = 5%)
    pub fn new(warning_threshold: f64, overfit_threshold: f64) -> Self {
        Self {
            warning_threshold,
            overfit_threshold,
        }
    }

    /// Check for overfitting
    ///
    /// # Arguments
    /// * `train_score` - Training set score
    /// * `val_score` - Validation set score
    pub fn check(&self, train_score: f64, val_score: f64) -> OverfitStatus {
        let gap = train_score - val_score;

        if gap > self.overfit_threshold {
            OverfitStatus::Overfitting
        } else if gap > self.warning_threshold {
            OverfitStatus::Warning
        } else {
            OverfitStatus::Ok
        }
    }

    /// Get the train-val gap
    pub fn gap(train_score: f64, val_score: f64) -> f64 {
        train_score - val_score
    }
}

impl Default for OverfitDetector {
    fn default() -> Self {
        Self::new(0.02, 0.05) // 2% warning, 5% overfit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_improvement() {
        let mut es = EarlyStopping::new(3, 0.001);

        // Improving scores
        assert!(!es.step(0.80, &[1.0], 1000));
        assert_eq!(es.best_score(), 0.80);

        assert!(!es.step(0.85, &[2.0], 2000));
        assert_eq!(es.best_score(), 0.85);

        assert!(!es.step(0.90, &[3.0], 5000));
        assert_eq!(es.best_score(), 0.90);
        assert_eq!(es.best_sample_size(), 5000);
    }

    #[test]
    fn test_early_stopping_stagnation() {
        let mut es = EarlyStopping::new(3, 0.001);

        es.step(0.90, &[1.0], 1000);

        // Non-improving scores
        assert!(!es.step(0.90, &[2.0], 2000)); // 1 step
        assert_eq!(es.remaining_patience(), 2);

        assert!(!es.step(0.89, &[3.0], 5000)); // 2 steps
        assert_eq!(es.remaining_patience(), 1);

        assert!(es.step(0.90, &[4.0], 10000)); // 3 steps -> stop
        assert_eq!(es.remaining_patience(), 0);

        // Best weights should be from first step
        assert_eq!(es.best_weights(), Some(&[1.0][..]));
    }

    #[test]
    fn test_early_stopping_minimize() {
        let mut es = EarlyStopping::new(2, 0.001).minimize();

        es.step(0.50, &[1.0], 1000);
        assert_eq!(es.best_score(), 0.50);

        es.step(0.40, &[2.0], 2000); // Improved (lower)
        assert_eq!(es.best_score(), 0.40);
    }

    #[test]
    fn test_overfit_detector_ok() {
        let detector = OverfitDetector::new(0.02, 0.05);

        assert_eq!(detector.check(0.90, 0.89), OverfitStatus::Ok);
        assert_eq!(detector.check(0.90, 0.90), OverfitStatus::Ok);
    }

    #[test]
    fn test_overfit_detector_warning() {
        let detector = OverfitDetector::new(0.02, 0.05);

        // Gap = 0.03 > 0.02 warning threshold
        assert_eq!(detector.check(0.93, 0.90), OverfitStatus::Warning);
    }

    #[test]
    fn test_overfit_detector_overfitting() {
        let detector = OverfitDetector::new(0.02, 0.05);

        // Gap = 0.10 > 0.05 overfit threshold
        assert_eq!(detector.check(0.95, 0.85), OverfitStatus::Overfitting);
    }

    #[test]
    fn test_gap_calculation() {
        assert!((OverfitDetector::gap(0.95, 0.90) - 0.05).abs() < 1e-9);
        assert!((OverfitDetector::gap(0.80, 0.85) - (-0.05)).abs() < 1e-9);
    }
}
