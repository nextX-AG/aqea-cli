//! Quality Metrics for Compression Evaluation
//!
//! Provides metrics for evaluating compression quality:
//! - Cosine Similarity: Information preservation
//! - Mean Squared Error (MSE): Reconstruction accuracy
//! - Spearman Correlation: Ranking preservation (most important for AQEA)

/// Compute cosine similarity between two vectors
///
/// Returns value in range [-1, 1] where 1 is identical direction.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-9 || norm_b < 1e-9 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Mean Squared Error (MSE)
///
/// MSE = (1/n) * Σ(original_i - decoded_i)²
/// Lower is better.
pub fn mean_squared_error(original: &[f32], decoded: &[f32]) -> f32 {
    debug_assert_eq!(original.len(), decoded.len());
    
    if original.is_empty() {
        return 0.0;
    }

    let n = original.len() as f32;
    original.iter()
        .zip(decoded.iter())
        .map(|(o, d)| (o - d).powi(2))
        .sum::<f32>() / n
}

/// Mean Absolute Error (MAE)
///
/// MAE = (1/n) * Σ|original_i - decoded_i|
/// More robust to outliers than MSE.
pub fn mean_absolute_error(original: &[f32], decoded: &[f32]) -> f32 {
    debug_assert_eq!(original.len(), decoded.len());
    
    if original.is_empty() {
        return 0.0;
    }

    let n = original.len() as f32;
    original.iter()
        .zip(decoded.iter())
        .map(|(o, d)| (o - d).abs())
        .sum::<f32>() / n
}

/// Compute Spearman rank correlation coefficient
///
/// Measures how well relative rankings are preserved.
/// This is the most important metric for AQEA as it measures
/// whether nearest neighbors are preserved after compression.
///
/// Returns value in range [-1, 1] where 1 is perfect rank preservation.
pub fn spearman_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    if n < 2 || n != y.len() {
        return 0.0;
    }

    // Create ranks for x
    let mut x_indexed: Vec<(f32, usize)> = x.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    x_indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut x_ranks = vec![0.0f32; n];
    for (rank, (_, idx)) in x_indexed.iter().enumerate() {
        x_ranks[*idx] = rank as f32;
    }

    // Create ranks for y
    let mut y_indexed: Vec<(f32, usize)> = y.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    y_indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_ranks = vec![0.0f32; n];
    for (rank, (_, idx)) in y_indexed.iter().enumerate() {
        y_ranks[*idx] = rank as f32;
    }

    // Pearson correlation of ranks
    let mean_x: f32 = x_ranks.iter().sum::<f32>() / n as f32;
    let mean_y: f32 = y_ranks.iter().sum::<f32>() / n as f32;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x_ranks[i] - mean_x;
        let dy = y_ranks[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-9 || var_y < 1e-9 {
        0.0
    } else {
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

/// Compute pairwise cosine similarities for a set of embeddings
pub fn pairwise_similarities(embeddings: &[Vec<f32>]) -> Vec<f32> {
    let n = embeddings.len();
    let mut sims = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            sims.push(cosine_similarity(&embeddings[i], &embeddings[j]));
        }
    }

    sims
}

/// Euclidean distance between two vectors
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Signal-to-Noise Ratio (SNR) in dB
///
/// Higher is better. Typical ranges:
/// - >40dB: Excellent
/// - 20-40dB: Good
/// - <20dB: Poor
pub fn signal_to_noise_ratio(original: &[f32], decoded: &[f32]) -> f32 {
    let signal_power: f32 = original.iter().map(|x| x * x).sum();
    let noise_power: f32 = original.iter()
        .zip(decoded.iter())
        .map(|(o, d)| (o - d).powi(2))
        .sum();

    if noise_power < 1e-12 {
        return f32::INFINITY;  // Perfect reconstruction
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Validation metrics for API responses
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationMetrics {
    /// Checksum match (true if original == recalculated)
    #[serde(rename = "checksumMatch")]
    pub checksum_match: bool,

    /// Cosine similarity between original and reconstructed
    #[serde(rename = "cosineSimilarity")]
    pub cosine_similarity: f32,

    /// Mean squared error
    pub mse: f32,

    /// Signal-to-noise ratio in dB (optional)
    #[serde(rename = "snrDb", skip_serializing_if = "Option::is_none")]
    pub snr_db: Option<f32>,
}

impl ValidationMetrics {
    /// Compute all validation metrics
    pub fn compute(original: &[f32], reconstructed: &[f32], checksum_match: bool) -> Self {
        let cosine = cosine_similarity(original, reconstructed);
        let mse = mean_squared_error(original, reconstructed);
        let snr = signal_to_noise_ratio(original, reconstructed);

        Self {
            checksum_match,
            cosine_similarity: cosine,
            mse,
            snr_db: if snr.is_finite() { Some(snr) } else { None },
        }
    }

    /// Create a placeholder when original is not available
    pub fn placeholder() -> Self {
        Self {
            checksum_match: false,
            cosine_similarity: 0.0,
            mse: 0.0,
            snr_db: None,
        }
    }
}

/// Comprehensive quality statistics
#[derive(Debug, Clone)]
pub struct QualityStats {
    pub mean: f32,
    pub min: f32,
    pub max: f32,
    pub std: f32,
}

impl QualityStats {
    pub fn from_values(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self { mean: 0.0, min: 0.0, max: 0.0, std: 0.0 };
        }

        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        Self { mean, min, max, std }
    }
}

/// Batch evaluation for multiple embeddings
pub struct BatchMetrics {
    pub cosine: QualityStats,
    pub mse: QualityStats,
    pub n_samples: usize,
}

impl BatchMetrics {
    /// Compute metrics over a batch of original/reconstructed pairs
    pub fn compute(originals: &[Vec<f32>], reconstructed: &[Vec<f32>]) -> Self {
        let cosines: Vec<f32> = originals.iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| cosine_similarity(o, r))
            .collect();

        let mses: Vec<f32> = originals.iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| mean_squared_error(o, r))
            .collect();

        Self {
            cosine: QualityStats::from_values(&cosines),
            mse: QualityStats::from_values(&mses),
            n_samples: originals.len(),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_mse_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(mean_squared_error(&a, &b) < 1e-9);
    }

    #[test]
    fn test_mse_known_value() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let dec = vec![1.1, 2.1, 3.1, 4.1];
        // MSE = (0.01 + 0.01 + 0.01 + 0.01) / 4 = 0.01
        assert!((mean_squared_error(&orig, &dec) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_spearman_perfect_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((spearman_correlation(&x, &y) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spearman_inverse_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((spearman_correlation(&x, &z) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_snr_perfect() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let dec = vec![1.0, 2.0, 3.0, 4.0];
        assert!(signal_to_noise_ratio(&orig, &dec).is_infinite());
    }

    #[test]
    fn test_snr_with_noise() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let dec = vec![1.1, 2.1, 3.1, 4.1];
        let snr = signal_to_noise_ratio(&orig, &dec);
        assert!(snr > 10.0);  // Should still be decent
        assert!(snr.is_finite());
    }

    #[test]
    fn test_quality_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = QualityStats::from_values(&values);
        
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_validation_metrics() {
        let orig = vec![0.1, 0.2, 0.3, 0.4];
        let recon = vec![0.1, 0.2, 0.3, 0.4];
        
        let metrics = ValidationMetrics::compute(&orig, &recon, true);
        assert!(metrics.checksum_match);
        assert!((metrics.cosine_similarity - 1.0).abs() < 1e-6);
        assert!(metrics.mse < 1e-9);
    }
}
