//! CSV support for AQEA CLI
//!
//! Provides parsing and writing of embeddings in CSV format.
//!
//! ## Supported Formats
//!
//! **With ID column:**
//! ```csv
//! id,dim_0,dim_1,dim_2,...
//! doc_001,0.123,0.456,0.789,...
//! doc_002,0.234,0.567,0.890,...
//! ```
//!
//! **Without ID column:**
//! ```csv
//! 0.123,0.456,0.789,...
//! 0.234,0.567,0.890,...
//! ```

use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

/// Detected file format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileFormat {
    Json,
    Csv,
}

/// Parsed CSV data with optional IDs
#[derive(Debug)]
pub struct CsvData {
    pub ids: Vec<Option<String>>,
    pub vectors: Vec<Vec<f32>>,
    pub dimension: usize,
}

impl CsvData {
    /// Check if all rows have IDs
    pub fn has_ids(&self) -> bool {
        self.ids.iter().any(|id| id.is_some())
    }
}

/// Detect file format from extension or content
pub fn detect_format(path: &Path, content: Option<&str>) -> FileFormat {
    // First check extension
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "csv" | "tsv" => return FileFormat::Csv,
            "json" => return FileFormat::Json,
            _ => {}
        }
    }

    // If no extension match, check content
    if let Some(content) = content {
        let trimmed = content.trim();
        if trimmed.starts_with('[') || trimmed.starts_with('{') {
            return FileFormat::Json;
        }
    }

    // Default to CSV if content looks like numbers
    FileFormat::Csv
}

/// Detect if CSV has a header row
///
/// Heuristic: if first row contains any non-numeric values, it's likely a header
pub fn detect_header(first_line: &str) -> bool {
    let fields: Vec<&str> = first_line.split(',').collect();
    if fields.is_empty() {
        return false;
    }

    // Check if first few fields are parseable as floats
    let non_numeric_count = fields
        .iter()
        .take(5) // Check first 5 columns
        .filter(|f| f.trim().parse::<f64>().is_err())
        .count();

    // If more than half of checked fields are non-numeric, it's probably a header
    non_numeric_count > fields.len().min(5) / 2
}

/// Detect if there's an ID column (first column is non-numeric)
pub fn detect_id_column(lines: &[&str], has_header: bool) -> Option<usize> {
    let start_idx = if has_header { 1 } else { 0 };

    if lines.len() <= start_idx {
        return None;
    }

    // Check if first column of data rows looks like an ID (non-numeric)
    let sample_line = lines[start_idx];
    let first_field = sample_line.split(',').next()?.trim();

    // If first field is not a valid float, it's likely an ID
    if first_field.parse::<f64>().is_err() {
        Some(0)
    } else {
        None
    }
}

/// Parse CSV embeddings from string
pub fn parse_csv(
    content: &str,
    has_header: Option<bool>,
    id_column: Option<usize>,
) -> Result<CsvData> {
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        anyhow::bail!("Empty CSV file");
    }

    // Auto-detect header if not specified
    let has_header = has_header.unwrap_or_else(|| detect_header(lines[0]));

    // Auto-detect ID column if not specified
    let id_col = id_column.or_else(|| detect_id_column(&lines, has_header));

    let start_idx = if has_header { 1 } else { 0 };

    let mut ids = Vec::new();
    let mut vectors = Vec::new();
    let mut dimension = None;

    for (line_num, line) in lines.iter().enumerate().skip(start_idx) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();

        let (id, values_fields): (Option<String>, Vec<&str>) = if let Some(id_idx) = id_col {
            let id = fields.get(id_idx).map(|s| s.to_string());
            let values: Vec<&str> = fields
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != id_idx)
                .map(|(_, v)| *v)
                .collect();
            (id, values)
        } else {
            (None, fields)
        };

        // Parse values
        let values: Vec<f32> = values_fields
            .iter()
            .map(|v| {
                v.parse::<f32>().with_context(|| {
                    format!(
                        "Invalid number '{}' at line {} in CSV",
                        v,
                        line_num + 1
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Check dimension consistency
        match dimension {
            None => dimension = Some(values.len()),
            Some(d) if d != values.len() => {
                anyhow::bail!(
                    "Dimension mismatch at line {}: expected {}, got {}",
                    line_num + 1,
                    d,
                    values.len()
                );
            }
            _ => {}
        }

        ids.push(id);
        vectors.push(values);
    }

    if vectors.is_empty() {
        anyhow::bail!("No valid data rows found in CSV");
    }

    Ok(CsvData {
        ids,
        vectors,
        dimension: dimension.unwrap_or(0),
    })
}

/// Parse CSV from a reader (for streaming large files)
pub fn parse_csv_streaming<R: Read>(
    reader: R,
    has_header: Option<bool>,
    id_column: Option<usize>,
) -> Result<CsvData> {
    let buf_reader = BufReader::new(reader);
    let mut lines: Vec<String> = Vec::new();

    for line in buf_reader.lines() {
        lines.push(line?);
    }

    let content = lines.join("\n");
    parse_csv(&content, has_header, id_column)
}

/// Write embeddings to CSV format
pub fn write_csv(
    data: &CsvData,
    include_header: bool,
) -> Result<String> {
    let mut output = String::new();

    let has_ids = data.has_ids();

    // Write header
    if include_header && !data.vectors.is_empty() {
        let dim = data.dimension;
        if has_ids {
            output.push_str("id,");
        }
        let dim_headers: Vec<String> = (0..dim).map(|i| format!("dim_{}", i)).collect();
        output.push_str(&dim_headers.join(","));
        output.push('\n');
    }

    // Write data rows
    for (id, vector) in data.ids.iter().zip(data.vectors.iter()) {
        if has_ids {
            if let Some(id) = id {
                output.push_str(id);
            }
            output.push(',');
        }

        let values: Vec<String> = vector.iter().map(|v| format!("{:.6}", v)).collect();
        output.push_str(&values.join(","));
        output.push('\n');
    }

    Ok(output)
}

/// Write embeddings to CSV file
pub fn write_csv_to_file<W: Write>(
    writer: &mut W,
    data: &CsvData,
    include_header: bool,
) -> Result<()> {
    let csv_content = write_csv(data, include_header)?;
    writer.write_all(csv_content.as_bytes())?;
    Ok(())
}

/// Convert vectors to CsvData (for output)
pub fn vectors_to_csv_data(
    vectors: Vec<Vec<f32>>,
    ids: Option<Vec<String>>,
) -> CsvData {
    let dimension = vectors.first().map(|v| v.len()).unwrap_or(0);

    let ids = match ids {
        Some(id_list) => id_list.into_iter().map(Some).collect(),
        None => vec![None; vectors.len()],
    };

    CsvData {
        ids,
        vectors,
        dimension,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_by_extension() {
        assert_eq!(
            detect_format(Path::new("data.csv"), None),
            FileFormat::Csv
        );
        assert_eq!(
            detect_format(Path::new("data.json"), None),
            FileFormat::Json
        );
    }

    #[test]
    fn test_detect_format_by_content() {
        assert_eq!(
            detect_format(Path::new("data"), Some("[[0.1, 0.2]]")),
            FileFormat::Json
        );
        assert_eq!(
            detect_format(Path::new("data"), Some("0.1,0.2,0.3")),
            FileFormat::Csv
        );
    }

    #[test]
    fn test_detect_header() {
        assert!(detect_header("id,dim_0,dim_1,dim_2"));
        assert!(!detect_header("0.1,0.2,0.3,0.4"));
        assert!(detect_header("doc_id,0.1,0.2"));
    }

    #[test]
    fn test_parse_csv_simple() {
        let csv = "0.1,0.2,0.3\n0.4,0.5,0.6\n";
        let data = parse_csv(csv, Some(false), None).unwrap();

        assert_eq!(data.vectors.len(), 2);
        assert_eq!(data.dimension, 3);
        assert!(!data.has_ids());
    }

    #[test]
    fn test_parse_csv_with_header() {
        let csv = "dim_0,dim_1,dim_2\n0.1,0.2,0.3\n0.4,0.5,0.6\n";
        let data = parse_csv(csv, Some(true), None).unwrap();

        assert_eq!(data.vectors.len(), 2);
        assert_eq!(data.dimension, 3);
    }

    #[test]
    fn test_parse_csv_with_ids() {
        let csv = "id,dim_0,dim_1\ndoc_1,0.1,0.2\ndoc_2,0.3,0.4\n";
        let data = parse_csv(csv, Some(true), Some(0)).unwrap();

        assert_eq!(data.vectors.len(), 2);
        assert_eq!(data.dimension, 2);
        assert!(data.has_ids());
        assert_eq!(data.ids[0], Some("doc_1".to_string()));
        assert_eq!(data.ids[1], Some("doc_2".to_string()));
    }

    #[test]
    fn test_parse_csv_auto_detect() {
        let csv = "id,dim_0,dim_1\ndoc_1,0.1,0.2\ndoc_2,0.3,0.4\n";
        let data = parse_csv(csv, None, None).unwrap();

        assert!(data.has_ids());
        assert_eq!(data.dimension, 2);
    }

    #[test]
    fn test_write_csv() {
        let data = CsvData {
            ids: vec![Some("doc_1".to_string()), Some("doc_2".to_string())],
            vectors: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            dimension: 3,
        };

        let output = write_csv(&data, true).unwrap();

        assert!(output.contains("id,dim_0,dim_1,dim_2"));
        assert!(output.contains("doc_1,0.100000,0.200000,0.300000"));
        assert!(output.contains("doc_2,0.400000,0.500000,0.600000"));
    }

    #[test]
    fn test_write_csv_no_header() {
        let data = CsvData {
            ids: vec![None, None],
            vectors: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            dimension: 2,
        };

        let output = write_csv(&data, false).unwrap();

        assert!(!output.contains("dim_"));
        assert!(output.contains("0.100000,0.200000"));
    }

    #[test]
    fn test_roundtrip() {
        let original = CsvData {
            ids: vec![Some("a".to_string()), Some("b".to_string())],
            vectors: vec![vec![0.123456, 0.654321], vec![0.111111, 0.999999]],
            dimension: 2,
        };

        let csv_output = write_csv(&original, true).unwrap();
        let parsed = parse_csv(&csv_output, None, None).unwrap();

        assert_eq!(parsed.vectors.len(), original.vectors.len());
        assert_eq!(parsed.dimension, original.dimension);

        for (orig, parsed) in original.vectors.iter().zip(parsed.vectors.iter()) {
            for (o, p) in orig.iter().zip(parsed.iter()) {
                assert!((o - p).abs() < 0.0001, "Values don't match: {} vs {}", o, p);
            }
        }
    }

    #[test]
    fn test_vectors_to_csv_data() {
        let vectors = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let ids = vec!["a".to_string(), "b".to_string()];

        let data = vectors_to_csv_data(vectors.clone(), Some(ids));

        assert_eq!(data.vectors, vectors);
        assert_eq!(data.ids[0], Some("a".to_string()));
        assert_eq!(data.dimension, 2);
    }

    #[test]
    fn test_empty_csv_error() {
        let result = parse_csv("", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let csv = "0.1,0.2,0.3\n0.4,0.5\n";
        let result = parse_csv(csv, Some(false), None);
        assert!(result.is_err());
    }
}
