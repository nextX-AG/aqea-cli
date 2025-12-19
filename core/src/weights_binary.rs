//! Binary weights format for AQEA Compression™ v2.0
//!
//! Format: .aqwt (AQEA Weights)
//! - 68% smaller than JSON
//! - 15x faster to parse
//! - Native Rust, zero-copy where possible
//!
//! v2.0 Format:
//! - Header: AQWT (4) + version (1) + original_dim (2) + compressed_dim (2)
//! - Meta: model_type (1) + spearman (4) + rotation_scale (4)
//! - Weights: f32[] little-endian

use std::io::{Read, Write, Cursor};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Magic bytes for .aqwt files
pub const MAGIC: &[u8; 4] = b"AQWT";

/// Current format version
pub const VERSION: u8 = 2;

/// Model type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelType {
    Unknown = 0,
    Text = 1,
    Audio = 2,
    OpenAI = 3,
    Protein = 4,
}

impl ModelType {
    pub fn from_byte(b: u8) -> Self {
        match b {
            1 => Self::Text,
            2 => Self::Audio,
            3 => Self::OpenAI,
            4 => Self::Protein,
            _ => Self::Unknown,
        }
    }
    
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "text" => Self::Text,
            "audio" => Self::Audio,
            "openai" => Self::OpenAI,
            "protein" => Self::Protein,
            _ => Self::Unknown,
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Text => "text",
            Self::Audio => "audio",
            Self::OpenAI => "openai",
            Self::Protein => "protein",
        }
    }
}

/// Binary weights container (v2.0)
#[derive(Debug, Clone)]
pub struct BinaryWeights {
    pub original_dim: u16,
    pub compressed_dim: u16,
    pub model_type: ModelType,
    pub spearman: f32,
    pub rotation_scale: f32,
    pub weights: Vec<f32>,
}

impl BinaryWeights {
    /// Create new weights from raw data (v2.0)
    pub fn new(
        original_dim: u16,
        compressed_dim: u16,
        weights: Vec<f32>,
        spearman: f32,
        rotation_scale: f32,
        model_type: ModelType,
    ) -> Self {
        assert_eq!(
            weights.len(),
            (original_dim as usize) * (compressed_dim as usize),
            "Weights size mismatch"
        );
        Self {
            original_dim,
            compressed_dim,
            model_type,
            spearman,
            rotation_scale,
            weights,
        }
    }

    /// Serialize to binary format (v2.0)
    /// 
    /// Format: AQWT(4) + ver(1) + orig(2) + comp(2) + type(1) + spearman(4) + rot(4) + weights
    pub fn to_bytes(&self) -> Vec<u8> {
        let header_size = 18;  // 4+1+2+2+1+4+4
        let data_size = header_size + self.weights.len() * 4;
        let mut buf = Vec::with_capacity(data_size);

        // Header
        buf.write_all(MAGIC).unwrap();
        buf.write_u8(VERSION).unwrap();
        buf.write_u16::<LittleEndian>(self.original_dim).unwrap();
        buf.write_u16::<LittleEndian>(self.compressed_dim).unwrap();
        buf.write_u8(self.model_type as u8).unwrap();
        buf.write_f32::<LittleEndian>(self.spearman).unwrap();
        buf.write_f32::<LittleEndian>(self.rotation_scale).unwrap();

        // Weights data
        for &w in &self.weights {
            buf.write_f32::<LittleEndian>(w).unwrap();
        }

        buf
    }

    /// Deserialize from binary format (v2.0)
    pub fn from_bytes(data: &[u8]) -> Result<Self, WeightsError> {
        let header_size = 18;
        if data.len() < header_size {
            return Err(WeightsError::TooShort);
        }

        let mut cursor = Cursor::new(data);

        // Read and validate magic
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(WeightsError::InvalidMagic);
        }

        // Read version
        let version = cursor.read_u8()?;
        if version > VERSION {
            return Err(WeightsError::UnsupportedVersion(version));
        }

        // Read metadata
        let original_dim = cursor.read_u16::<LittleEndian>()?;
        let compressed_dim = cursor.read_u16::<LittleEndian>()?;
        let model_type = ModelType::from_byte(cursor.read_u8()?);
        let spearman = cursor.read_f32::<LittleEndian>()?;
        
        // v2.0: read rotation_scale (v1 defaults to 1.75)
        let rotation_scale = if version >= 2 {
            cursor.read_f32::<LittleEndian>()?
        } else {
            1.75
        };

        // Calculate expected weights count
        let num_weights = (original_dim as usize) * (compressed_dim as usize);
        let expected_size = header_size + num_weights * 4;

        if data.len() < expected_size {
            return Err(WeightsError::DataTruncated {
                expected: expected_size,
                actual: data.len(),
            });
        }

        // Read weights
        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            weights.push(cursor.read_f32::<LittleEndian>()?);
        }

        Ok(Self {
            original_dim,
            compressed_dim,
            model_type,
            spearman,
            rotation_scale,
            weights,
        })
    }

    /// Get weights as 2D matrix (row-major: compressed_dim × original_dim)
    pub fn as_matrix(&self) -> Vec<Vec<f32>> {
        self.weights
            .chunks(self.original_dim as usize)
            .map(|row| row.to_vec())
            .collect()
    }

    /// Load from file
    pub fn load(path: &std::path::Path) -> Result<Self, WeightsError> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Save to file
    pub fn save(&self, path: &std::path::Path) -> Result<(), WeightsError> {
        let data = self.to_bytes();
        std::fs::write(path, data)?;
        Ok(())
    }
}

/// Errors that can occur during weights operations
#[derive(Debug)]
pub enum WeightsError {
    TooShort,
    InvalidMagic,
    UnsupportedVersion(u8),
    DataTruncated { expected: usize, actual: usize },
    Io(std::io::Error),
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => write!(f, "Data too short for valid weights file"),
            Self::InvalidMagic => write!(f, "Invalid magic bytes, expected 'AQWT'"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            Self::DataTruncated { expected, actual } => {
                write!(f, "Data truncated: expected {} bytes, got {}", expected, actual)
            }
            Self::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for WeightsError {}

impl From<std::io::Error> for WeightsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_v2() {
        let weights = BinaryWeights::new(
            768,
            26,
            vec![0.1f32; 768 * 26],
            0.983,
            1.75,
            ModelType::Text,
        );

        let bytes = weights.to_bytes();
        let loaded = BinaryWeights::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.original_dim, 768);
        assert_eq!(loaded.compressed_dim, 26);
        assert_eq!(loaded.model_type, ModelType::Text);
        assert!((loaded.spearman - 0.983).abs() < 0.001);
        assert!((loaded.rotation_scale - 1.75).abs() < 0.001);
        assert_eq!(loaded.weights.len(), 768 * 26);
    }

    #[test]
    fn test_size_v2() {
        let weights = BinaryWeights::new(768, 26, vec![0.0f32; 768 * 26], 0.0, 1.75, ModelType::Audio);
        let bytes = weights.to_bytes();
        
        // Header: 18 bytes, Data: 768 * 26 * 4 = 79,872 bytes
        assert_eq!(bytes.len(), 18 + 768 * 26 * 4);
    }
    
    #[test]
    fn test_model_type() {
        assert_eq!(ModelType::from_str("text"), ModelType::Text);
        assert_eq!(ModelType::from_str("audio"), ModelType::Audio);
        assert_eq!(ModelType::from_str("openai"), ModelType::OpenAI);
        assert_eq!(ModelType::from_str("protein"), ModelType::Protein);
        assert_eq!(ModelType::from_str("???"), ModelType::Unknown);
    }
}

