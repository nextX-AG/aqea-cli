//! Physical and Mathematical Constants for AQEA
//!
//! Core constants used throughout the compression system.
//! These are validated mathematical constants - DO NOT MODIFY.

/// Golden Ratio φ = (1 + √5) / 2
pub const PHI: f64 = 1.618033988749895;

/// Golden Ratio as f32
pub const PHI_F32: f32 = 1.618033988749895;

/// Golden Angle in radians = 2π / φ²
pub const GOLDEN_ANGLE: f64 = 2.399963229728653;

/// Golden Angle as f32
pub const GOLDEN_ANGLE_F32: f32 = 2.399963229728653;

/// Reduced Planck constant (normalized for energy calculations)
pub const HBAR: f64 = 0.01;

/// Tesla 3-6-9 ratios for energy stacking
pub const TESLA_3: f64 = 3.0 / 18.0;  // 0.167
pub const TESLA_6: f64 = 6.0 / 18.0;  // 0.333
pub const TESLA_9: f64 = 9.0 / 18.0;  // 0.500

/// Fibonacci sequence (first 20 terms)
pub const FIBONACCI: [u64; 20] = [
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
    89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765
];

/// Boltzmann constant in eV/K (for thermodynamic calculations)
pub const KB_EV: f32 = 0.00008617;

/// Standard temperature for Monte Carlo (Kelvin)
pub const TEMPERATURE_DEFAULT: f32 = 300.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_ratio() {
        // φ² = φ + 1
        let phi_squared = PHI * PHI;
        let expected = PHI + 1.0;
        assert!((phi_squared - expected).abs() < 1e-10);
    }

    #[test]
    fn test_golden_angle() {
        // Golden angle = 2π / φ²
        use std::f64::consts::PI;
        let expected = 2.0 * PI / (PHI * PHI);
        assert!((GOLDEN_ANGLE - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fibonacci_ratio_converges_to_phi() {
        // Consecutive Fibonacci ratios converge to φ
        for i in 10..19 {
            let ratio = FIBONACCI[i + 1] as f64 / FIBONACCI[i] as f64;
            assert!((ratio - PHI).abs() < 0.001);
        }
    }
}

