//! Minimal CMA-ES Implementation for AQEA Training
//!
//! Covariance Matrix Adaptation Evolution Strategy
//! Optimized for training compression weights
//!
//! Based on: Hansen, N. (2006). The CMA Evolution Strategy: A Tutorial

use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rayon::prelude::*;

/// CMA-ES Optimizer
pub struct CMAES {
    /// Problem dimension
    dim: usize,

    /// Population size (lambda)
    lambda: usize,

    /// Number of parents for recombination (mu)
    mu: usize,

    /// Weights for recombination
    weights: DVector<f64>,

    /// Effective mu
    mueff: f64,

    /// Learning rate for mean update
    cm: f64,

    /// Learning rate for step size (sigma)
    cs: f64,

    /// Damping for step size
    damps: f64,

    /// Learning rate for rank-one update
    cc: f64,

    /// Learning rate for rank-mu update
    c1: f64,

    /// Learning rate for rank-mu update
    cmu: f64,

    /// Expected value of ||N(0,I)||
    chiN: f64,

    /// Current mean
    mean: DVector<f64>,

    /// Current step size (sigma)
    sigma: f64,

    /// Covariance matrix
    C: DMatrix<f64>,

    /// Evolution path for sigma
    ps: DVector<f64>,

    /// Evolution path for C
    pc: DVector<f64>,

    /// Eigenvalues of C
    D: DVector<f64>,

    /// Eigenvectors of C (columns)
    B: DMatrix<f64>,

    /// sqrt(C)^-1 = B * D^-1 * B^T
    invsqrtC: DMatrix<f64>,

    /// Eigendecomposition counter
    eigeneval: usize,

    /// Generation counter
    generation: usize,
}

impl CMAES {
    /// Create new CMA-ES optimizer
    ///
    /// # Arguments
    /// * `dim` - Problem dimension
    /// * `initial_mean` - Starting point
    /// * `sigma` - Initial step size (typically 0.3-0.5)
    /// * `lambda` - Population size (if 0, auto-calculated)
    pub fn new(dim: usize, initial_mean: Option<DVector<f64>>, sigma: f64, lambda: usize) -> Self {
        // Population size
        let lambda = if lambda == 0 {
            4 + (3.0 * (dim as f64).ln()).floor() as usize
        } else {
            lambda
        };

        // Number of parents
        let mu = lambda / 2;

        // Recombination weights
        let mut weights = DVector::zeros(mu);
        for i in 0..mu {
            weights[i] = ((lambda as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln();
        }
        let sum_weights: f64 = weights.iter().sum();
        weights /= sum_weights;

        // Variance-effectiveness
        let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Strategy parameters
        let cc = (4.0 + mueff / dim as f64) / (dim as f64 + 4.0 + 2.0 * mueff / dim as f64);
        let cs = (mueff + 2.0) / (dim as f64 + mueff + 5.0);
        let c1 = 2.0 / ((dim as f64 + 1.3).powi(2) + mueff);
        let cmu = f64::min(
            1.0 - c1,
            2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim as f64 + 2.0).powi(2) + mueff),
        );
        let damps = 1.0 + 2.0 * f64::max(0.0, ((mueff - 1.0) / (dim as f64 + 1.0)).sqrt() - 1.0) + cs;
        let chiN = (dim as f64).sqrt() * (1.0 - 1.0 / (4.0 * dim as f64) + 1.0 / (21.0 * (dim as f64).powi(2)));

        // Initialize state
        let mean = initial_mean.unwrap_or_else(|| DVector::zeros(dim));
        let C = DMatrix::identity(dim, dim);
        let ps = DVector::zeros(dim);
        let pc = DVector::zeros(dim);
        let D = DVector::from_element(dim, 1.0);
        let B = DMatrix::identity(dim, dim);
        let invsqrtC = DMatrix::identity(dim, dim);

        Self {
            dim,
            lambda,
            mu,
            weights,
            mueff,
            cm: 1.0,
            cs,
            damps,
            cc,
            c1,
            cmu,
            chiN,
            mean,
            sigma,
            C,
            ps,
            pc,
            D,
            B,
            invsqrtC,
            eigeneval: 0,
            generation: 0,
        }
    }

    /// Sample new population
    pub fn ask(&self) -> Vec<DVector<f64>> {
        let mut rng = rand::thread_rng();

        (0..self.lambda)
            .map(|_| {
                // Sample from N(0, I)
                let z: DVector<f64> = DVector::from_fn(self.dim, |_, _| rng.sample(Standard));

                // Transform: mean + sigma * B * D * z
                let y = &self.B * self.D.component_mul(&z);
                &self.mean + self.sigma * y
            })
            .collect()
    }

    /// Update distribution based on fitness values
    ///
    /// # Arguments
    /// * `solutions` - Sampled solutions from `ask()`
    /// * `fitnesses` - Fitness values (lower is better - CMA-ES minimizes!)
    pub fn tell(&mut self, solutions: &[DVector<f64>], fitnesses: &[f64]) {
        assert_eq!(solutions.len(), self.lambda);
        assert_eq!(fitnesses.len(), self.lambda);

        // Sort by fitness (lower is better)
        let mut indices: Vec<usize> = (0..self.lambda).collect();
        indices.sort_by(|&a, &b| fitnesses[a].partial_cmp(&fitnesses[b]).unwrap());

        // Old mean for update
        let mean_old = self.mean.clone();

        // Update mean (weighted average of best mu solutions)
        self.mean = DVector::zeros(self.dim);
        for i in 0..self.mu {
            self.mean += self.weights[i] * &solutions[indices[i]];
        }

        // Update evolution paths
        let y = (&self.mean - &mean_old) / self.sigma;
        let z = &self.invsqrtC * &y;

        // Update sigma path
        self.ps = (1.0 - self.cs) * &self.ps + (self.cs * (2.0 - self.cs) * self.mueff).sqrt() * &z;

        // Heaviside function
        let ps_norm = self.ps.norm();
        let hsig = if ps_norm / (1.0 - (1.0 - self.cs).powi(2 * (self.generation as i32 + 1))).sqrt()
            < (1.4 + 2.0 / (self.dim as f64 + 1.0)) * self.chiN
        {
            1.0
        } else {
            0.0
        };

        // Update pc path
        self.pc = (1.0 - self.cc) * &self.pc + hsig * (self.cc * (2.0 - self.cc) * self.mueff).sqrt() * &y;

        // Adapt covariance matrix
        let artmp: Vec<DVector<f64>> = (0..self.mu)
            .map(|i| (&solutions[indices[i]] - &mean_old) / self.sigma)
            .collect();

        // Rank-one update
        let c1a = self.c1 * (1.0 - (1.0 - hsig * hsig) * self.cc * (2.0 - self.cc));
        let rank_one = &self.pc * self.pc.transpose();

        // Rank-mu update
        let mut rank_mu = DMatrix::zeros(self.dim, self.dim);
        for i in 0..self.mu {
            rank_mu += self.weights[i] * (&artmp[i] * artmp[i].transpose());
        }

        // Update C
        self.C = (1.0 - c1a - self.cmu) * &self.C + self.c1 * rank_one + self.cmu * rank_mu;

        // Adapt sigma
        self.sigma *= ((self.cs / self.damps) * (ps_norm / self.chiN - 1.0)).exp();

        // Eigendecomposition of C
        self.eigeneval += 1;
        if self.eigeneval > self.lambda as usize / (10 * self.dim) {
            self.update_eigensystem();
            self.eigeneval = 0;
        }

        self.generation += 1;
    }

    /// Update eigensystem of C
    fn update_eigensystem(&mut self) {
        // Enforce symmetry
        self.C = (&self.C + self.C.transpose()) / 2.0;

        // Eigendecomposition
        let eig = self.C.clone().symmetric_eigen();

        self.D = eig.eigenvalues.map(|v| v.max(1e-20).sqrt());
        self.B = eig.eigenvectors;

        // Compute invsqrtC = B * D^-1 * B^T
        let D_inv = DVector::from_fn(self.dim, |i, _| 1.0 / self.D[i]);
        let D_inv_diag = DMatrix::from_diagonal(&D_inv);
        self.invsqrtC = &self.B * D_inv_diag * self.B.transpose();
    }

    /// Get current best estimate (mean)
    pub fn best(&self) -> &DVector<f64> {
        &self.mean
    }

    /// Get current sigma
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Get current generation
    pub fn generation(&self) -> usize {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_optimization() {
        // Simple sphere function: f(x) = sum(x^2)
        let dim = 10;
        let mut cmaes = CMAES::new(dim, None, 0.5, 0);

        for _ in 0..50 {
            let solutions = cmaes.ask();
            let fitnesses: Vec<f64> = solutions.iter().map(|x| x.iter().map(|v| v * v).sum()).collect();
            cmaes.tell(&solutions, &fitnesses);
        }

        // Mean should be close to zero
        assert!(cmaes.best().norm() < 0.1);
    }
}

