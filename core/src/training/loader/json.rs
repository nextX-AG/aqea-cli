//! JSON Format Loader
//!
//! Supports multiple JSON formats for embedding data:
//!
//! # Format 1: Simple array of arrays
//! ```json
//! [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
//! ```
//!
//! # Format 2: Object with embeddings field
//! ```json
//! {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}
//! ```
//!
//! # Format 3: Array of objects with vector field
//! ```json
//! [{"vector": [1.0, 2.0], "label": "SEEN"}, ...]
//! ```
//!
//! # Format 4: Full EmbeddingData format
//! ```json
//! {"embeddings": [...], "labels": [...], "metadata": {...}}
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{DataLoader, EmbeddingData, LoaderError};

/// JSON format loader
pub struct JsonLoader;

/// Format 3: Object with vector field
#[derive(Deserialize)]
struct VectorObject {
    vector: Vec<f32>,
    #[serde(default)]
    label: Option<String>,
}

/// Format 4: Full embedding data
#[derive(Serialize, Deserialize)]
struct JsonEmbeddingData {
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    labels: Option<Vec<String>>,
    #[serde(default)]
    metadata: Option<HashMap<String, Value>>,
}

impl JsonLoader {
    /// Write embedding data to JSON format
    pub fn write<P: AsRef<Path>>(path: P, data: &EmbeddingData) -> Result<(), LoaderError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        let json_data = JsonEmbeddingData {
            embeddings: data.embeddings.clone(),
            labels: data.labels.clone(),
            metadata: if data.metadata.is_empty() {
                None
            } else {
                Some(data.metadata.clone())
            },
        };

        serde_json::to_writer_pretty(writer, &json_data)?;
        Ok(())
    }

    /// Write just the embeddings as a simple array
    pub fn write_simple<P: AsRef<Path>>(path: P, embeddings: &[Vec<f32>]) -> Result<(), LoaderError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        serde_json::to_writer(&mut writer, embeddings)?;
        writer.flush()?;
        Ok(())
    }
}

impl DataLoader for JsonLoader {
    fn load(&self, path: &Path) -> Result<EmbeddingData, LoaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        // Parse as generic JSON first
        let value: Value = serde_json::from_reader(reader)?;

        // Try to parse in order of preference
        match &value {
            // Format 1: Simple array of arrays [[f32]]
            Value::Array(arr) if !arr.is_empty() => {
                // Check first element
                match &arr[0] {
                    // Format 1: [[1.0, 2.0], ...]
                    Value::Array(_) => {
                        let embeddings: Vec<Vec<f32>> = serde_json::from_value(value)?;
                        if embeddings.is_empty() {
                            return Err(LoaderError::EmptyData);
                        }
                        Ok(EmbeddingData::new(embeddings))
                    }
                    // Format 3: [{"vector": [...], "label": "..."}]
                    Value::Object(_) => {
                        let objects: Vec<VectorObject> = serde_json::from_value(value)?;
                        if objects.is_empty() {
                            return Err(LoaderError::EmptyData);
                        }

                        let embeddings: Vec<Vec<f32>> = objects.iter()
                            .map(|o| o.vector.clone())
                            .collect();

                        let labels: Vec<String> = objects.iter()
                            .map(|o| o.label.clone().unwrap_or_default())
                            .collect();

                        // Only include labels if at least one is non-empty
                        let has_labels = labels.iter().any(|l| !l.is_empty());

                        if has_labels {
                            Ok(EmbeddingData::with_labels(embeddings, labels))
                        } else {
                            Ok(EmbeddingData::new(embeddings))
                        }
                    }
                    _ => Err(LoaderError::InvalidFormat(
                        "Expected array of arrays or objects".into()
                    )),
                }
            }
            // Format 2/4: Object with embeddings field
            Value::Object(obj) => {
                if let Some(emb_value) = obj.get("embeddings") {
                    let embeddings: Vec<Vec<f32>> = serde_json::from_value(emb_value.clone())?;
                    if embeddings.is_empty() {
                        return Err(LoaderError::EmptyData);
                    }

                    let labels = obj.get("labels")
                        .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok());

                    let metadata = obj.get("metadata")
                        .and_then(|v| serde_json::from_value::<HashMap<String, Value>>(v.clone()).ok())
                        .unwrap_or_default();

                    let mut data = EmbeddingData::new(embeddings);
                    data.labels = labels;
                    data.metadata = metadata;

                    Ok(data)
                } else if let Some(vec_value) = obj.get("vectors") {
                    // Alternative: "vectors" instead of "embeddings"
                    let embeddings: Vec<Vec<f32>> = serde_json::from_value(vec_value.clone())?;
                    if embeddings.is_empty() {
                        return Err(LoaderError::EmptyData);
                    }
                    Ok(EmbeddingData::new(embeddings))
                } else {
                    Err(LoaderError::InvalidFormat(
                        "Object must have 'embeddings' or 'vectors' field".into()
                    ))
                }
            }
            _ => Err(LoaderError::InvalidFormat(
                "Expected array or object".into()
            )),
        }
    }

    fn supports(&self, path: &Path) -> bool {
        // Check extension
        if let Some(ext) = path.extension() {
            if ext == "json" {
                return true;
            }
        }

        // Check if starts with [ or {
        if let Ok(mut file) = File::open(path) {
            use std::io::Read;
            let mut byte = [0u8; 1];
            if file.read_exact(&mut byte).is_ok() {
                return byte[0] == b'[' || byte[0] == b'{';
            }
        }

        false
    }

    fn format_name(&self) -> &'static str {
        "JSON"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_format1_simple_array() {
        let file = NamedTempFile::with_suffix(".json").unwrap();
        std::fs::write(file.path(), r#"[[1.0, 2.0], [3.0, 4.0]]"#).unwrap();

        let data = JsonLoader.load(file.path()).unwrap();

        assert_eq!(data.len(), 2);
        assert_eq!(data.dimension(), 2);
        assert_eq!(data.embeddings[0], vec![1.0, 2.0]);
    }

    #[test]
    fn test_format2_object_with_embeddings() {
        let file = NamedTempFile::with_suffix(".json").unwrap();
        std::fs::write(file.path(), r#"{"embeddings": [[1.0], [2.0], [3.0]]}"#).unwrap();

        let data = JsonLoader.load(file.path()).unwrap();

        assert_eq!(data.len(), 3);
        assert_eq!(data.dimension(), 1);
    }

    #[test]
    fn test_format3_vector_objects() {
        let file = NamedTempFile::with_suffix(".json").unwrap();
        std::fs::write(file.path(), r#"[
            {"vector": [1.0, 2.0], "label": "SEEN"},
            {"vector": [3.0, 4.0], "label": "UNSEEN"}
        ]"#).unwrap();

        let data = JsonLoader.load(file.path()).unwrap();

        assert_eq!(data.len(), 2);
        assert!(data.labels.is_some());
        let labels = data.labels.unwrap();
        assert_eq!(labels[0], "SEEN");
        assert_eq!(labels[1], "UNSEEN");
    }

    #[test]
    fn test_format4_full() {
        let file = NamedTempFile::with_suffix(".json").unwrap();
        std::fs::write(file.path(), r#"{
            "embeddings": [[1.0], [2.0]],
            "labels": ["A", "B"],
            "metadata": {"source": "test"}
        }"#).unwrap();

        let data = JsonLoader.load(file.path()).unwrap();

        assert_eq!(data.len(), 2);
        assert!(data.labels.is_some());
        assert!(data.metadata.contains_key("source"));
    }

    #[test]
    fn test_write_and_read() {
        let emb = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let labels = vec!["A".into(), "B".into()];
        let data = EmbeddingData::with_labels(emb, labels);

        let file = NamedTempFile::with_suffix(".json").unwrap();
        JsonLoader::write(file.path(), &data).unwrap();

        let loaded = JsonLoader.load(file.path()).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.dimension(), 3);
        assert!(loaded.labels.is_some());
    }

    #[test]
    fn test_write_simple() {
        let emb = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];

        let file = NamedTempFile::with_suffix(".json").unwrap();
        JsonLoader::write_simple(file.path(), &emb).unwrap();

        let loaded = JsonLoader.load(file.path()).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_vectors_field() {
        let file = NamedTempFile::with_suffix(".json").unwrap();
        std::fs::write(file.path(), r#"{"vectors": [[1.0], [2.0]]}"#).unwrap();

        let data = JsonLoader.load(file.path()).unwrap();
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_empty_data() {
        let file = NamedTempFile::with_suffix(".json").unwrap();
        std::fs::write(file.path(), r#"[]"#).unwrap();

        let result = JsonLoader.load(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_supports() {
        let file = NamedTempFile::with_suffix(".json").unwrap();
        std::fs::write(file.path(), r#"[[1.0]]"#).unwrap();

        assert!(JsonLoader.supports(file.path()));
    }

    #[test]
    fn test_format_name() {
        assert_eq!(JsonLoader.format_name(), "JSON");
    }
}
