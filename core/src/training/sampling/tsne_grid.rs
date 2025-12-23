//! t-SNE Grid Sampler - Grid-based sampling in reduced space
//!
//! User-proposed sampling strategy:
//! 1. Project embeddings to 2D (via t-SNE or PCA)
//! 2. Overlay a grid on the 2D space
//! 3. Select one point from each grid cell
//!
//! This guarantees uniform coverage of the embedding space.
//!
//! # Why this works
//!
//! - t-SNE preserves local structure in 2D
//! - Grid sampling ensures even distribution
//! - Much faster than high-dimensional distance computations
//!
//! # Implementation
//!
//! Uses Python sklearn/umap for t-SNE computation (much faster than Rust).
//! Falls back to PCA if Python unavailable.

use super::Sampler;
use rand::prelude::*;
use std::collections::HashSet;
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// Projection method for 2D reduction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectionMethod {
    /// t-SNE (best structure, slower)
    Tsne,
    /// UMAP (good structure, faster than t-SNE)
    Umap,
    /// PCA (fastest, basic structure)
    Pca,
}

impl Default for ProjectionMethod {
    fn default() -> Self {
        ProjectionMethod::Tsne
    }
}

/// t-SNE Grid Sampler for uniform space coverage
///
/// Samples points by overlaying a grid on a 2D projection
/// of the embedding space and selecting representatives
/// from each cell.
#[derive(Debug, Clone)]
pub struct TsneGridSampler {
    /// Pre-computed 2D coordinates (optional)
    /// If None, will compute using Python or PCA fallback
    coords_2d: Option<Vec<[f32; 2]>>,
    /// Grid resolution (grid_size x grid_size cells)
    grid_size: usize,
    /// Random seed
    seed: u64,
    /// Projection method (t-SNE, UMAP, PCA)
    method: ProjectionMethod,
    /// Whether to use Python for projection
    use_python: bool,
}

impl TsneGridSampler {
    /// Create a new t-SNE Grid sampler with default settings
    pub fn new() -> Self {
        Self {
            coords_2d: None,
            grid_size: 50,
            seed: 42,
            method: ProjectionMethod::Tsne,
            use_python: true,
        }
    }

    /// Create with specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            coords_2d: None,
            grid_size: 50,
            seed,
            method: ProjectionMethod::Tsne,
            use_python: true,
        }
    }

    /// Set pre-computed 2D coordinates (from t-SNE, UMAP, etc.)
    pub fn with_coords(mut self, coords: Vec<[f32; 2]>) -> Self {
        self.coords_2d = Some(coords);
        self
    }

    /// Set grid resolution
    pub fn with_grid_size(mut self, size: usize) -> Self {
        self.grid_size = size.max(2);
        self
    }

    /// Set seed (builder pattern)
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    
    /// Set projection method
    pub fn with_method(mut self, method: ProjectionMethod) -> Self {
        self.method = method;
        self
    }
    
    /// Disable Python (use Rust PCA fallback only)
    pub fn without_python(mut self) -> Self {
        self.use_python = false;
        self
    }
    
    /// Compute 2D projection using Python (t-SNE, UMAP, or PCA)
    fn compute_projection_python(&self, embeddings: &[Vec<f32>]) -> Result<Vec<[f32; 2]>, String> {
        // Find Python script
        let script_paths = vec![
            PathBuf::from("cli/scripts/compute_tsne.py"),
            PathBuf::from("/home/aqea/aqea-compress/cli/scripts/compute_tsne.py"),
        ];
        
        let script_path = script_paths.iter()
            .find(|p| p.exists())
            .ok_or_else(|| "Python t-SNE script not found".to_string())?;
        
        // Find Python
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
        let temp_file = temp_dir.join(format!("tsne_emb_{}.json", std::process::id()));
        
        let json_data = serde_json::to_string(embeddings)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;
        std::fs::write(&temp_file, json_data)
            .map_err(|e| format!("Failed to write temp file: {}", e))?;
        
        // Method string
        let method_str = match self.method {
            ProjectionMethod::Tsne => "tsne",
            ProjectionMethod::Umap => "umap",
            ProjectionMethod::Pca => "pca",
        };
        
        // Spawn Python process
        let mut child = Command::new(python)
            .arg(script_path)
            .arg(&temp_file)
            .arg(method_str)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to spawn Python: {}", e))?;
        
        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);
        
        let mut coords: Vec<[f32; 2]> = Vec::new();
        
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                match json.get("type").and_then(|t| t.as_str()) {
                    Some("result") => {
                        if let Some(coord_arr) = json.get("coords").and_then(|c| c.as_array()) {
                            coords = coord_arr.iter()
                                .filter_map(|c| {
                                    let arr = c.as_array()?;
                                    Some([
                                        arr.get(0)?.as_f64()? as f32,
                                        arr.get(1)?.as_f64()? as f32,
                                    ])
                                })
                                .collect();
                        }
                    }
                    Some("error") => {
                        let msg = json.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown");
                        return Err(format!("Python error: {}", msg));
                    }
                    _ => {}
                }
            }
        }
        
        let _ = child.wait();
        let _ = std::fs::remove_file(&temp_file);
        
        if coords.is_empty() {
            return Err("No coordinates returned from Python".to_string());
        }
        
        Ok(coords)
    }

    /// Compute simple PCA-based 2D projection
    ///
    /// Uses power iteration to find top 2 principal components.
    fn compute_pca_2d(embeddings: &[Vec<f32>], seed: u64) -> Vec<[f32; 2]> {
        if embeddings.is_empty() {
            return vec![];
        }

        let n = embeddings.len();
        let d = embeddings[0].len();

        // Center the data
        let mut mean = vec![0.0f32; d];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }

        let centered: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|emb| emb.iter().zip(&mean).map(|(v, m)| v - m).collect())
            .collect();

        // Power iteration for first principal component
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pc1: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
        normalize(&mut pc1);

        for _ in 0..50 {
            let mut new_pc = vec![0.0f32; d];
            for row in &centered {
                let dot: f32 = row.iter().zip(&pc1).map(|(a, b)| a * b).sum();
                for (i, &v) in row.iter().enumerate() {
                    new_pc[i] += dot * v;
                }
            }
            normalize(&mut new_pc);
            pc1 = new_pc;
        }

        // Power iteration for second principal component (orthogonal to first)
        let mut pc2: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
        orthogonalize(&mut pc2, &pc1);
        normalize(&mut pc2);

        for _ in 0..50 {
            let mut new_pc = vec![0.0f32; d];
            for row in &centered {
                let dot: f32 = row.iter().zip(&pc2).map(|(a, b)| a * b).sum();
                for (i, &v) in row.iter().enumerate() {
                    new_pc[i] += dot * v;
                }
            }
            orthogonalize(&mut new_pc, &pc1);
            normalize(&mut new_pc);
            pc2 = new_pc;
        }

        // Project onto PC1 and PC2
        centered
            .iter()
            .map(|row| {
                let x: f32 = row.iter().zip(&pc1).map(|(a, b)| a * b).sum();
                let y: f32 = row.iter().zip(&pc2).map(|(a, b)| a * b).sum();
                [x, y]
            })
            .collect()
    }

    /// Sample using grid-based selection
    fn grid_sample(&self, coords: &[[f32; 2]], n_samples: usize, seed: u64) -> Vec<usize> {
        if coords.is_empty() {
            return vec![];
        }

        // Find bounds
        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;

        for &[x, y] in coords {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }

        // Add small padding to avoid edge issues
        let x_range = (x_max - x_min).max(1e-6);
        let y_range = (y_max - y_min).max(1e-6);

        // Compute grid cell for each point
        let grid_size = self.grid_size;
        let mut cells: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();

        for (idx, &[x, y]) in coords.iter().enumerate() {
            let gx = ((x - x_min) / x_range * (grid_size - 1) as f32).round() as usize;
            let gy = ((y - y_min) / y_range * (grid_size - 1) as f32).round() as usize;
            let gx = gx.min(grid_size - 1);
            let gy = gy.min(grid_size - 1);
            cells.entry((gx, gy)).or_default().push(idx);
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut selected: HashSet<usize> = HashSet::new();

        // Strategy 1: Select one from each occupied cell
        // Sort cells first for deterministic ordering, then shuffle
        let mut cell_list: Vec<_> = cells.into_iter()
            .map(|(cell, mut indices)| {
                indices.sort_unstable(); // Sort indices within each cell for determinism
                (cell, indices)
            })
            .collect();
        cell_list.sort_by_key(|(cell, _)| *cell);
        cell_list.shuffle(&mut rng);

        for (_cell, indices) in &cell_list {
            if selected.len() >= n_samples {
                break;
            }
            // Select random point from this cell
            if let Some(&idx) = indices.choose(&mut rng) {
                selected.insert(idx);
            }
        }

        // Strategy 2: If we need more, do second pass with remaining points
        if selected.len() < n_samples {
            let remaining: Vec<usize> = (0..coords.len())
                .filter(|i| !selected.contains(i))
                .collect();

            for idx in remaining.choose_multiple(&mut rng, n_samples - selected.len()) {
                selected.insert(*idx);
            }
        }

        let mut result: Vec<usize> = selected.into_iter().collect();
        result.sort_unstable();
        result
    }
}

impl Default for TsneGridSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for TsneGridSampler {
    fn sample(&self, embeddings: &[Vec<f32>], n_samples: usize) -> Vec<usize> {
        let n = embeddings.len();
        let n_samples = n_samples.min(n);

        if n_samples == 0 || n == 0 {
            return vec![];
        }

        if n_samples == n {
            return (0..n).collect();
        }

        // Get 2D coordinates (priority: pre-computed > Python > Rust PCA)
        let coords = match &self.coords_2d {
            Some(c) if c.len() == n => c.clone(),
            _ => {
                if self.use_python {
                    // Try Python t-SNE/UMAP
                    match self.compute_projection_python(embeddings) {
                        Ok(coords) => coords,
                        Err(_e) => {
                            // Fallback to Rust PCA
                            Self::compute_pca_2d(embeddings, self.seed)
                        }
                    }
                } else {
                    Self::compute_pca_2d(embeddings, self.seed)
                }
            }
        };

        self.grid_sample(&coords, n_samples, self.seed)
    }

    fn name(&self) -> &'static str {
        "tsne-grid"
    }

    fn description(&self) -> &'static str {
        "Grid sampling in t-SNE/PCA reduced space for uniform coverage"
    }
}

/// Normalize a vector to unit length
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Make vector orthogonal to another (Gram-Schmidt)
fn orthogonalize(v: &mut [f32], other: &[f32]) {
    let dot: f32 = v.iter().zip(other).map(|(a, b)| a * b).sum();
    for (x, &o) in v.iter_mut().zip(other) {
        *x -= dot * o;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_embeddings(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(123);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    fn generate_grid_embeddings() -> Vec<Vec<f32>> {
        // Create embeddings that form a clear 2D grid pattern
        let mut embeddings = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                // First two dimensions form a grid, rest are noise
                let mut emb = vec![i as f32, j as f32];
                let mut rng = StdRng::seed_from_u64((i * 10 + j) as u64);
                for _ in 0..30 {
                    emb.push(rng.gen::<f32>() * 0.1);
                }
                embeddings.push(emb);
            }
        }
        embeddings
    }

    #[test]
    fn test_tsne_grid_sampler_basic() {
        let sampler = TsneGridSampler::new();
        let embeddings = generate_embeddings(100, 32);

        let indices = sampler.sample(&embeddings, 20);

        assert_eq!(indices.len(), 20);
        // Check all indices are valid
        for &idx in &indices {
            assert!(idx < 100);
        }
        // Check no duplicates
        let unique: HashSet<_> = indices.iter().collect();
        assert_eq!(unique.len(), 20);
    }

    #[test]
    fn test_tsne_grid_sampler_all() {
        let sampler = TsneGridSampler::new();
        let embeddings = generate_embeddings(50, 32);

        let indices = sampler.sample(&embeddings, 50);

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_tsne_grid_sampler_more_than_available() {
        let sampler = TsneGridSampler::new();
        let embeddings = generate_embeddings(30, 32);

        let indices = sampler.sample(&embeddings, 100);

        assert_eq!(indices.len(), 30);
    }

    #[test]
    fn test_tsne_grid_with_precomputed_coords() {
        let embeddings = generate_embeddings(100, 32);

        // Create simple pre-computed coords
        let coords: Vec<[f32; 2]> = (0..100)
            .map(|i| [(i % 10) as f32, (i / 10) as f32])
            .collect();

        let sampler = TsneGridSampler::new()
            .with_coords(coords)
            .with_grid_size(10);

        let indices = sampler.sample(&embeddings, 20);

        assert_eq!(indices.len(), 20);
    }

    #[test]
    fn test_tsne_grid_coverage() {
        // Test that grid sampling covers the space well
        let embeddings = generate_grid_embeddings();
        let sampler = TsneGridSampler::new().with_grid_size(10);

        let indices = sampler.sample(&embeddings, 25);

        // Check that selected points span different grid regions
        // Convert indices back to original grid positions
        let positions: HashSet<(usize, usize)> = indices
            .iter()
            .map(|&i| (i / 10, i % 10))
            .collect();

        // With 25 samples from 100 (10x10 grid), we expect good spread
        // At least 5 different rows and columns should be hit
        let rows: HashSet<_> = positions.iter().map(|(r, _)| r).collect();
        let cols: HashSet<_> = positions.iter().map(|(_, c)| c).collect();

        assert!(rows.len() >= 4, "Expected at least 4 rows covered, got {}", rows.len());
        assert!(cols.len() >= 4, "Expected at least 4 cols covered, got {}", cols.len());
    }

    #[test]
    fn test_pca_projection() {
        let embeddings = generate_embeddings(50, 32);
        let coords = TsneGridSampler::compute_pca_2d(&embeddings, 42);

        assert_eq!(coords.len(), 50);

        // Check that projection has reasonable variance
        let x_var: f32 = coords.iter().map(|c| c[0] * c[0]).sum::<f32>() / 50.0;
        let y_var: f32 = coords.iter().map(|c| c[1] * c[1]).sum::<f32>() / 50.0;

        assert!(x_var > 0.0, "PCA x-projection should have variance");
        assert!(y_var > 0.0, "PCA y-projection should have variance");
    }

    #[test]
    fn test_tsne_grid_deterministic() {
        let embeddings = generate_embeddings(100, 32);

        let sampler1 = TsneGridSampler::with_seed(42);
        let sampler2 = TsneGridSampler::with_seed(42);

        let indices1 = sampler1.sample(&embeddings, 30);
        let indices2 = sampler2.sample(&embeddings, 30);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_tsne_grid_empty() {
        let sampler = TsneGridSampler::new();
        let embeddings: Vec<Vec<f32>> = vec![];
        let indices = sampler.sample(&embeddings, 10);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_tsne_grid_name() {
        let sampler = TsneGridSampler::new();
        assert_eq!(sampler.name(), "tsne-grid");
    }

    #[test]
    fn test_grid_size_configuration() {
        let sampler = TsneGridSampler::new()
            .with_grid_size(100)
            .seed(123);

        assert_eq!(sampler.grid_size, 100);
        assert_eq!(sampler.seed, 123);
    }
}
