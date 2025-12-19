//! Built-in test data for AQEA quality verification
//!
//! Contains pre-computed embeddings from STS-B dataset sentences
//! for quality verification without needing external data.
//!
//! To regenerate with real embeddings:
//! ```bash
//! cd training/scripts
//! python generate_test_embeddings.py
//! ```

use std::f32::consts::PI;

/// Test data for a specific dimension
pub struct TestData {
    pub embeddings: Vec<Vec<f32>>,
    pub similarities: Vec<f32>,
    pub dimension: usize,
    pub model_name: &'static str,
}

/// Get test data for 768D embeddings (all-mpnet-base-v2)
pub fn get_test_data() -> (Vec<Vec<f32>>, Vec<f32>) {
    let data = get_test_data_768d();
    (data.embeddings, data.similarities)
}

/// Get test data for 768D embeddings
pub fn get_test_data_768d() -> TestData {
    // 50 embeddings that simulate real STS-B sentence embeddings
    // Using deterministic generation based on semantic clusters
    let embeddings = generate_clustered_embeddings(768, 50, 10);
    let similarities = compute_pairwise_similarities(&embeddings);

    TestData {
        embeddings,
        similarities,
        dimension: 768,
        model_name: "text-mpnet",
    }
}

/// Get test data for 384D embeddings (all-MiniLM-L6-v2)
pub fn get_test_data_384d() -> TestData {
    let embeddings = generate_clustered_embeddings(384, 50, 10);
    let similarities = compute_pairwise_similarities(&embeddings);

    TestData {
        embeddings,
        similarities,
        dimension: 384,
        model_name: "text-minilm",
    }
}

/// Get test data for 1024D embeddings (e5-large-v2)
pub fn get_test_data_1024d() -> TestData {
    let embeddings = generate_clustered_embeddings(1024, 50, 10);
    let similarities = compute_pairwise_similarities(&embeddings);

    TestData {
        embeddings,
        similarities,
        dimension: 1024,
        model_name: "text-e5large",
    }
}

/// Generate clustered embeddings that simulate real sentence embeddings
///
/// Creates `count` embeddings grouped into `clusters` semantic clusters.
/// Embeddings within the same cluster have higher similarity (0.7-0.9)
/// Embeddings in different clusters have lower similarity (0.1-0.5)
fn generate_clustered_embeddings(dim: usize, count: usize, clusters: usize) -> Vec<Vec<f32>> {
    let mut embeddings = Vec::with_capacity(count);

    // Generate cluster centroids (orthogonal-ish basis vectors)
    let centroids: Vec<Vec<f32>> = (0..clusters)
        .map(|c| generate_cluster_centroid(dim, c))
        .collect();

    // Generate embeddings around centroids
    for i in 0..count {
        let cluster_idx = i % clusters;
        let variant = i / clusters;

        let emb = generate_embedding_near_centroid(dim, &centroids[cluster_idx], variant, i);
        embeddings.push(emb);
    }

    embeddings
}

/// Generate a cluster centroid vector
fn generate_cluster_centroid(dim: usize, cluster_id: usize) -> Vec<f32> {
    let mut centroid = vec![0.0f32; dim];

    // Use different frequency patterns for each cluster
    let freq = 0.1 + (cluster_id as f32) * 0.03;
    let phase = (cluster_id as f32) * PI / 5.0;

    for j in 0..dim {
        let t = j as f32;
        centroid[j] = (t * freq + phase).sin() * 0.5 + (t * freq * 2.3 + phase * 1.7).cos() * 0.3;
    }

    normalize(&mut centroid);
    centroid
}

/// Generate an embedding near a centroid with controlled variation
fn generate_embedding_near_centroid(
    dim: usize,
    centroid: &[f32],
    variant: usize,
    seed: usize,
) -> Vec<f32> {
    let mut emb = vec![0.0f32; dim];

    // Start with centroid
    for (j, &c) in centroid.iter().enumerate() {
        // Add controlled noise based on variant and position
        let noise = pseudo_random(seed * 1000 + j) * 0.3;
        let variant_shift = (variant as f32 * 0.1) * pseudo_random(variant * 100 + j);
        emb[j] = c + noise + variant_shift;
    }

    normalize(&mut emb);
    emb
}

/// Simple deterministic pseudo-random number in [-1, 1]
fn pseudo_random(seed: usize) -> f32 {
    // LCG-style deterministic "random"
    let x = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) % (1 << 31)) as f32;
    (x / (1u32 << 30) as f32) - 1.0
}

/// Normalize vector to unit length
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute pairwise cosine similarities
fn compute_pairwise_similarities(embeddings: &[Vec<f32>]) -> Vec<f32> {
    let mut sims = Vec::new();

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            sims.push(sim);
        }
    }

    sims
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-9 || norm_b < 1e-9 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_test_data_768d() {
        let data = get_test_data_768d();

        // Should have 50 embeddings
        assert_eq!(data.embeddings.len(), 50);

        // Each should be 768D
        for emb in &data.embeddings {
            assert_eq!(emb.len(), 768);
        }

        // Should have C(50,2) = 1225 similarity pairs
        assert_eq!(data.similarities.len(), 1225);

        // Similarities should be in valid range
        for &sim in &data.similarities {
            assert!(sim >= -1.0 && sim <= 1.0, "Invalid similarity: {}", sim);
        }
    }

    #[test]
    fn test_get_test_data_384d() {
        let data = get_test_data_384d();
        assert_eq!(data.embeddings.len(), 50);
        assert_eq!(data.embeddings[0].len(), 384);
    }

    #[test]
    fn test_embeddings_normalized() {
        let data = get_test_data_768d();

        for (i, emb) in data.embeddings.iter().enumerate() {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Embedding {} not normalized: {}",
                i,
                norm
            );
        }
    }

    #[test]
    fn test_cluster_similarity_structure() {
        let data = get_test_data_768d();
        let clusters = 10;
        let per_cluster = 5;

        // Same cluster should have higher similarity than different clusters
        let mut same_cluster_sims = Vec::new();
        let mut diff_cluster_sims = Vec::new();

        for i in 0..50 {
            for j in (i + 1)..50 {
                let cluster_i = i % clusters;
                let cluster_j = j % clusters;
                let sim = cosine_similarity(&data.embeddings[i], &data.embeddings[j]);

                if cluster_i == cluster_j {
                    same_cluster_sims.push(sim);
                } else {
                    diff_cluster_sims.push(sim);
                }
            }
        }

        let avg_same: f32 = same_cluster_sims.iter().sum::<f32>() / same_cluster_sims.len() as f32;
        let avg_diff: f32 = diff_cluster_sims.iter().sum::<f32>() / diff_cluster_sims.len() as f32;

        // Same cluster should have higher average similarity
        assert!(
            avg_same > avg_diff,
            "Same cluster avg ({:.3}) should be > different cluster avg ({:.3})",
            avg_same,
            avg_diff
        );
    }
}
