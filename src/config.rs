//! Configuration management for AQEA CLI
//!
//! Stores credentials and settings in ~/.aqea/config.toml

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;

// ============================================================================
// FEATURE FLAGS
// ============================================================================

/// Feature flags for enabling/disabling CLI functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Option A: /benchmark with pretrained weights (Standard)
    #[serde(default = "default_true")]
    pub benchmark_pretrained: bool,
    
    /// Option B: /benchmark --train with fresh training (Standard)
    #[serde(default = "default_true")]
    pub benchmark_with_train: bool,
    
    /// Option C: /train standalone command (Internal Only)
    #[serde(default = "default_false")]
    pub standalone_training: bool,
}

fn default_true() -> bool { true }
fn default_false() -> bool { false }

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            benchmark_pretrained: true,   // A: Standard
            benchmark_with_train: true,   // B: Standard
            standalone_training: false,   // C: Internal only
        }
    }
}

// ============================================================================
// MAIN CONFIG
// ============================================================================

/// CLI Configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// API key for authentication
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    
    /// Default model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    
    /// API base URL (for enterprise/self-hosted)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_url: Option<String>,
    
    /// Feature flags for advanced features
    #[serde(default)]
    pub features: FeatureFlags,
}

impl Config {
    /// Get the config directory path (~/.aqea/)
    pub fn dir() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
        Ok(home.join(".aqea"))
    }
    
    /// Get the config file path (~/.aqe/config.toml)
    pub fn path() -> Result<PathBuf> {
        Ok(Self::dir()?.join("config.toml"))
    }
    
    /// Load config from file (creates default if not exists)
    pub fn load() -> Result<Self> {
        let path = Self::path()?;
        
        if !path.exists() {
            return Ok(Self::default());
        }
        
        let contents = fs::read_to_string(&path)?;
        let config: Config = toml::from_str(&contents)?;
        
        Ok(config)
    }
    
    /// Save config to file
    pub fn save(&self) -> Result<()> {
        let dir = Self::dir()?;
        let path = Self::path()?;
        
        // Create directory if needed
        if !dir.exists() {
            fs::create_dir_all(&dir)?;
        }
        
        // Serialize and save
        let contents = toml::to_string_pretty(self)?;
        fs::write(&path, contents)?;
        
        // Set restrictive permissions on config file (contains API key)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&path)?.permissions();
            perms.set_mode(0o600);
            fs::set_permissions(&path, perms)?;
        }
        
        Ok(())
    }
    
    /// Get API URL with default fallback
    pub fn api_url(&self) -> &str {
        self.api_url.as_deref().unwrap_or("https://api.aqea.ai")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert!(config.api_key.is_none());
        assert!(config.default_model.is_none());
    }
    
    #[test]
    fn test_api_url_default() {
        let config = Config::default();
        assert_eq!(config.api_url(), "https://api.aqea.ai");
    }
}

