//! Authentication utilities for AQEA CLI
//!
//! Handles API key validation and storage.

use anyhow::Result;
use crate::config::Config;

/// Validate an API key format
pub fn validate_key_format(key: &str) -> bool {
    // Expected format: aqea_live_xxxx or aqea_test_xxxx
    key.starts_with("aqea_") && key.len() >= 20
}

/// Check if we have valid credentials
pub fn is_authenticated() -> Result<bool> {
    let config = Config::load()?;
    Ok(config.api_key.is_some())
}

/// Get the current API key (if authenticated)
pub fn get_api_key() -> Result<Option<String>> {
    let config = Config::load()?;
    Ok(config.api_key)
}

/// Mask an API key for display (show first 8 and last 4 chars)
pub fn mask_key(key: &str) -> String {
    if key.len() <= 12 {
        return "*".repeat(key.len());
    }
    format!("{}...{}", &key[..8], &key[key.len()-4..])
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_key_format() {
        assert!(validate_key_format("aqea_live_sk_1234567890abcdef"));
        assert!(validate_key_format("aqea_test_sk_1234567890abcdef"));
        assert!(!validate_key_format("sk_1234567890"));
        assert!(!validate_key_format("aqea_"));
    }
    
    #[test]
    fn test_mask_key() {
        assert_eq!(mask_key("aqea_live_sk_1234567890abcde"), "aqea_liv...bcde");
    }
}

