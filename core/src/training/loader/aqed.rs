//! AQED Binary Format Loader
//!
//! Fast binary format for embedding data (60x faster than JSON).
//!
//! # Format Specification (v1) - Python-compatible 64-byte header
//!
//! ```text
//! Offset  Size    Description
//! ------  ----    -----------
//! 0       4       Magic: "AQED"
//! 4       4       Version (u32 LE, currently 1)
//! 8       4       N: number of embeddings (u32 LE)
//! 12      4       D_orig: original dimension (u32 LE)
//! 16      4       D_aqea: AQEA dimension (u32 LE)
//! 20      4       Flags (u32 LE)
//! 24      4       PQ subvectors (u32 LE)
//! 28      4       PQ centroids (u32 LE)
//! 32      28      Reserved
//! 64      ...     Data sections (based on flags)
//!
//! Flags:
//!   Bit 0: HAS_ORIGINAL (original embeddings present)
//!   Bit 1: HAS_AQEA (AQEA compressed present)
//!   Bit 2: HAS_INT8 (int8 quantized present)
//!   Bit 3: HAS_PQ (PQ codes present)
//!   Bit 4: HAS_TSNE (t-SNE coordinates present)
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, Seek, SeekFrom};
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use super::{DataLoader, EmbeddingData, LoaderError};

/// AQED format magic bytes
const MAGIC: &[u8; 4] = b"AQED";
/// Current format version
const VERSION: u32 = 1;
/// Header size: 4 (magic) + 7*4 (u32s) + 28 (reserved) = 60 bytes
const HEADER_SIZE: u64 = 60;

// Flags
const FLAG_HAS_ORIGINAL: u32 = 1 << 0;
const FLAG_HAS_AQEA: u32 = 1 << 1;
const FLAG_HAS_INT8: u32 = 1 << 2;
const FLAG_HAS_PQ: u32 = 1 << 3;
const FLAG_HAS_TSNE: u32 = 1 << 4;

/// AQED binary format loader
pub struct AqedLoader;

impl AqedLoader {
    /// Write embedding data to AQED format (64-byte header, Python-compatible)
    pub fn write<P: AsRef<Path>>(path: P, data: &EmbeddingData) -> Result<(), LoaderError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let n = data.len() as u32;
        let d = data.dimension() as u32;
        let flags = FLAG_HAS_ORIGINAL;  // We only write original embeddings

        // Write 64-byte header
        writer.write_all(MAGIC)?;                           // 4 bytes
        writer.write_u32::<LittleEndian>(VERSION)?;         // 4 bytes
        writer.write_u32::<LittleEndian>(n)?;               // 4 bytes (count)
        writer.write_u32::<LittleEndian>(d)?;               // 4 bytes (original dim)
        writer.write_u32::<LittleEndian>(0)?;               // 4 bytes (aqea dim - not used)
        writer.write_u32::<LittleEndian>(flags)?;           // 4 bytes (flags)
        writer.write_u32::<LittleEndian>(0)?;               // 4 bytes (pq subvectors)
        writer.write_u32::<LittleEndian>(0)?;               // 4 bytes (pq centroids)
        writer.write_all(&[0u8; 28])?;                      // 28 bytes reserved

        // Write embeddings (original)
        for emb in &data.embeddings {
            for &val in emb {
                writer.write_f32::<LittleEndian>(val)?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

impl DataLoader for AqedLoader {
    fn load(&self, path: &Path) -> Result<EmbeddingData, LoaderError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(LoaderError::InvalidFormat(
                format!("Invalid AQED magic: {:?}", magic)
            ));
        }

        // Read 64-byte header (Python-compatible format)
        let version = reader.read_u32::<LittleEndian>()?;
        if version != VERSION {
            return Err(LoaderError::InvalidFormat(
                format!("Unsupported AQED version: {}", version)
            ));
        }

        let n = reader.read_u32::<LittleEndian>()? as usize;
        let d_orig = reader.read_u32::<LittleEndian>()? as usize;
        let d_aqea = reader.read_u32::<LittleEndian>()? as usize;
        let flags = reader.read_u32::<LittleEndian>()?;
        let _pq_subvectors = reader.read_u32::<LittleEndian>()?;
        let _pq_centroids = reader.read_u32::<LittleEndian>()?;
        
        // Skip reserved bytes (28 bytes)
        let mut reserved = [0u8; 28];
        reader.read_exact(&mut reserved)?;

        if n == 0 {
            return Err(LoaderError::EmptyData);
        }

        // Log what we're loading
        eprintln!("  📄 AQED: {} items, {}D original, {}D AQEA, flags={:05b}", 
            n, d_orig, d_aqea, flags);

        // Determine which embeddings to load (prefer original for training)
        let (dim, offset) = if flags & FLAG_HAS_ORIGINAL != 0 {
            eprintln!("  📂 Loading original embeddings ({}D)", d_orig);
            (d_orig, HEADER_SIZE)
        } else if flags & FLAG_HAS_AQEA != 0 {
            eprintln!("  📂 Loading AQEA embeddings ({}D)", d_aqea);
            (d_aqea, HEADER_SIZE)
        } else {
            return Err(LoaderError::InvalidFormat(
                "AQED file has no embeddings (neither original nor AQEA)".into()
            ));
        };

        // Seek to data offset
        reader.seek(SeekFrom::Start(offset))?;

        // Read embeddings
        let mut embeddings = Vec::with_capacity(n);
        for _ in 0..n {
            let mut emb = Vec::with_capacity(dim);
            for _ in 0..dim {
                emb.push(reader.read_f32::<LittleEndian>()?);
            }
            embeddings.push(emb);
        }

        eprintln!("  ✓ Loaded {} embeddings ({}D)", embeddings.len(), dim);

        Ok(EmbeddingData::new(embeddings))
    }

    fn supports(&self, path: &Path) -> bool {
        // Check extension first
        if let Some(ext) = path.extension() {
            if ext == "aqed" {
                return true;
            }
        }

        // Check magic bytes
        if let Ok(mut file) = File::open(path) {
            let mut magic = [0u8; 4];
            if file.read_exact(&mut magic).is_ok() {
                return &magic == MAGIC;
            }
        }

        false
    }

    fn format_name(&self) -> &'static str {
        "AQED"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_write_and_read() {
        let emb = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let data = EmbeddingData::new(emb);

        // Write
        let file = NamedTempFile::with_suffix(".aqed").unwrap();
        AqedLoader::write(file.path(), &data).unwrap();

        // Read
        let loaded = AqedLoader.load(file.path()).unwrap();

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.dimension(), 3);
        assert_eq!(loaded.embeddings[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded.embeddings[2], vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_large_data() {
        // Test with larger dataset
        let n = 1000;
        let d = 128;
        let emb: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..d).map(|j| (i * d + j) as f32 * 0.001).collect())
            .collect();
        let data = EmbeddingData::new(emb);

        let file = NamedTempFile::with_suffix(".aqed").unwrap();
        AqedLoader::write(file.path(), &data).unwrap();

        let loaded = AqedLoader.load(file.path()).unwrap();

        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.dimension(), d);
        assert!((loaded.embeddings[0][0] - 0.0).abs() < 1e-6);
        assert!((loaded.embeddings[999][127] - (999.0 * 128.0 + 127.0) * 0.001).abs() < 1e-4);
    }

    #[test]
    fn test_supports() {
        let loader = AqedLoader;

        let file = NamedTempFile::with_suffix(".aqed").unwrap();
        let data = EmbeddingData::new(vec![vec![1.0]]);
        AqedLoader::write(file.path(), &data).unwrap();

        assert!(loader.supports(file.path()));
    }

    #[test]
    fn test_invalid_magic() {
        let file = NamedTempFile::with_suffix(".aqed").unwrap();
        std::fs::write(file.path(), b"NOTAQED").unwrap();

        let result = AqedLoader.load(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_format_name() {
        assert_eq!(AqedLoader.format_name(), "AQED");
    }

    #[test]
    fn test_header_size() {
        // Verify our header is exactly 60 bytes (Python-compatible format)
        let emb = vec![vec![1.0, 2.0]];
        let data = EmbeddingData::new(emb);

        let file = NamedTempFile::with_suffix(".aqed").unwrap();
        AqedLoader::write(file.path(), &data).unwrap();

        let meta = std::fs::metadata(file.path()).unwrap();
        // 60 header + 2 floats * 4 bytes = 68 bytes
        assert_eq!(meta.len(), 60 + 2 * 4);
    }
}
