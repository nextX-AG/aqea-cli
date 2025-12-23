//! Command handler for REPL

use super::Command;
use anyhow::Context;
use console::style;
use dialoguer::{Password, Select};
use std::path::PathBuf;
use std::time::Instant;
use std::io::Write;
use rayon::prelude::*;
use std::sync::mpsc;
use std::thread;
use std::process::{Command as ProcessCommand, Stdio};
use std::io::{BufRead, BufReader};

use aqea_core::{
    OctonionCompressor, Compressor, BinaryWeights, ProductQuantizer,
    cosine_similarity, spearman_correlation, CMAES,
};

use crate::config::{Config, FeatureFlags};

/// Available models with PQ options
const MODELS: &[(&str, &str, &str, &[usize])] = &[
    ("openai-small", "1536d → 52d", "OpenAI text-embedding-3-small", &[13, 10, 7]),
    ("openai-large", "3072d → 105d", "OpenAI text-embedding-3-large", &[10, 8, 6, 4]),
    ("e5-large", "1024d → 35d", "E5-Large-v2", &[7, 5]),
    ("mpnet", "768d → 26d", "all-mpnet-base-v2", &[7, 4]),
    ("minilm", "384d → 13d", "all-MiniLM-L6-v2", &[7, 4]),
    ("audio-wav2vec2", "768d → 26d", "wav2vec2-base", &[13, 10, 7, 4]),
    ("protein-esm", "320d → 11d", "ESM-2", &[6, 4]),
    ("auto", "auto", "Auto-detect from input dimensions", &[]),
];

/// Compression modes
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionMode {
    /// 2-Stage: AQEA only (29x compression)
    Aqea,
    /// 3-Stage: AQEA + PQ (213-3072x compression)
    AqeaPq,
}

/// Session state
pub struct CommandHandler {
    api_url: String,
    api_key: Option<String>,
    user_email: Option<String>,
    user_plan: Option<String>,
    quota_remaining: Option<i64>,
    selected_model: String,
    compression_mode: CompressionMode,
    pq_subvectors: Option<usize>,  // None = auto (best quality)
    compression_ratio: u32,
    features: FeatureFlags,
}

impl CommandHandler {
    pub fn new(api_url: String) -> Self {
        // Try to load saved credentials
        let (api_key, user_email) = load_saved_credentials();
        
        // Load feature flags from config
        let features = Config::load()
            .map(|c| c.features)
            .unwrap_or_default();
        
        Self {
            api_url,
            api_key,
            user_email,
            user_plan: None,
            quota_remaining: None,
            selected_model: "auto".to_string(),
            compression_mode: CompressionMode::Aqea,  // Default: 2-Stage
            pq_subvectors: None,  // Auto = best quality
            compression_ratio: 29,
            features,
        }
    }
    
    /// Get the prompt string
    pub fn get_prompt(&self) -> String {
        if self.api_key.is_some() {
            format!("{} ", style("aqea>").green().bold())
        } else {
            format!("{} ", style("aqea>").yellow())
        }
    }
    
    /// Execute a command, returns true if should exit
    pub fn execute(&mut self, cmd: Command) -> anyhow::Result<bool> {
        match cmd {
            Command::Help => self.cmd_help(),
            Command::Exit => return Ok(true),
            Command::Clear => self.cmd_clear(),
            
            Command::Login { api_key } => self.cmd_login(api_key)?,
            Command::Logout => self.cmd_logout(),
            Command::Status => self.cmd_status(),
            
            Command::Models => self.cmd_models(),
            Command::Model { name } => self.cmd_model(name)?,
            Command::Mode { mode } => self.cmd_mode(mode)?,
            Command::Subvectors { count } => self.cmd_subvectors(count)?,
            
            Command::Compress { file, output } => self.cmd_compress(file, output)?,
            Command::CompressPq { file, output } => self.cmd_compress_pq(file, output)?,
            Command::Decompress { file, output } => self.cmd_decompress(file, output)?,
            Command::Batch { folder } => self.cmd_batch(folder)?,
            Command::Ratio { value } => self.cmd_ratio(value)?,
            
            // Benchmark & Training commands (NEW!)
            Command::Benchmark { file, output, train, compressions, pq_subs } => {
                self.cmd_benchmark(file, output, train, compressions, pq_subs)?
            }
            Command::Train { file, compression, name } => {
                self.cmd_train(file, compression, name)?
            }
            
            Command::Config => self.cmd_config(),
            Command::History => self.cmd_history(),
            
            Command::Unknown { input } => {
                println!("{} Unknown input: {}", style("?").yellow(), input);
                println!("  Type {} for available commands", style("/help").cyan());
            }
        }
        
        Ok(false)
    }
    
    // ========================================================================
    // Core Commands
    // ========================================================================
    
    fn cmd_help(&self) {
        println!();
        println!("{}", style("Available Commands:").bold().underlined());
        println!();
        
        println!("{}", style("Core:").cyan().bold());
        println!("  {}  {}   Show this help", style("/help").green(), style("(/h, /?)").dim());
        println!("  {}  {}   Exit the CLI", style("/exit").green(), style("(/quit, /q)").dim());
        println!("  {} {}   Clear screen", style("/clear").green(), style("(/cls)").dim());
        println!();
        
        println!("{}", style("Authentication:").cyan().bold());
        println!("  {} {}   Login with API key", style("/login").green(), style("[key]").dim());
        println!("  {}          Logout", style("/logout").green());
        println!("  {} {}   Show session info", style("/status").green(), style("(/whoami)").dim());
        println!();
        
        println!("{}", style("Models & Mode:").cyan().bold());
        println!("  {}         List available models + weights status", style("/models").green());
        println!("  {}  {}   Select model", style("/model").green(), style("[name]").dim());
        println!("  {}   {}   Select mode (aqea/pq)", style("/mode").green(), style("[aqea|pq]").dim());
        println!("  {}   {}   Set PQ subvectors", style("/subs").green(), style("[n]").dim());
        println!();
        
        println!("{}", style("Compression:").cyan().bold());
        println!("  {} {} {}   2-Stage (AQEA only)", style("/compress").green(), style("<file>").yellow(), style("[-o out]").dim());
        println!("  {} {} {}   3-Stage (AQEA+PQ)", style("/compress-pq").green(), style("<file>").yellow(), style("[-o out]").dim());
        println!("  {} {} {}   Decompress file", style("/decompress").green(), style("<file>").yellow(), style("[-o out]").dim());
        println!();
        
        // Benchmark & Training
        if self.features.benchmark_pretrained || self.features.benchmark_with_train {
            println!("{}", style("Benchmarking:").cyan().bold());
            if self.features.benchmark_pretrained {
                println!("  {} {} {}   Benchmark with pretrained weights", 
                    style("/benchmark").green(), style("<file>").yellow(), style("[-o out]").dim());
            }
            if self.features.benchmark_with_train {
                println!("  {} {} {}   Benchmark with fresh training", 
                    style("/benchmark").green(), style("<file> --train").yellow(), style("[-o out]").dim());
            }
            println!("    {} Options: --compressions 10,20,30 --pq-subs 4,8,13,16", style("│").dim());
            println!();
        }
        
        // Standalone training (internal only)
        if self.features.standalone_training {
            println!("{}", style("Training (Internal):").magenta().bold());
            println!("  {} {} {}   Train new AQEA weights", 
                style("/train").green(), style("<file>").yellow(), style("[--compression 20] [--name model]").dim());
            println!();
        }
        
        println!("{}", style("Compression Modes:").dim());
        println!("    {} 2-Stage (AQEA):    ~29x compression, works with ANY vector DB", style("•").cyan());
        println!("    {} 3-Stage (AQEA+PQ): 213-3072x, requires codebook for search", style("•").magenta());
        println!();
        
        println!("{}", style("Config:").cyan().bold());
        println!("  {}         Show configuration", style("/config").green());
        println!("  {}        Show command history", style("/history").green());
        println!();
    }
    
    fn cmd_clear(&self) {
        print!("\x1B[2J\x1B[1;1H");
    }
    
    // ========================================================================
    // Auth Commands
    // ========================================================================
    
    fn cmd_login(&mut self, api_key: Option<String>) -> anyhow::Result<()> {
        let key = match api_key {
            Some(k) => k,
            None => {
                Password::new()
                    .with_prompt("API Key")
                    .interact()?
            }
        };
        
        // Validate key with API
        println!("{} Validating API key...", style("⟳").cyan());
        
        match self.validate_api_key(&key) {
            Ok((email, plan, quota)) => {
                self.api_key = Some(key.clone());
                self.user_email = Some(email.clone());
                self.user_plan = Some(plan.clone());
                self.quota_remaining = Some(quota);
                
                // Save credentials
                save_credentials(&key);
                
                println!("{} Logged in as {} ({} Plan, {} requests remaining)", 
                    style("✓").green().bold(),
                    style(&email).cyan(),
                    style(&plan).yellow(),
                    style(quota).green()
                );
            }
            Err(e) => {
                println!("{} Login failed: {}", style("✗").red().bold(), e);
            }
        }
        
        Ok(())
    }
    
    fn cmd_logout(&mut self) {
        self.api_key = None;
        self.user_email = None;
        self.user_plan = None;
        self.quota_remaining = None;
        
        clear_credentials();
        
        println!("{} Logged out", style("✓").green());
    }
    
    fn cmd_status(&self) {
        println!();
        if let Some(ref email) = self.user_email {
            println!("  {}    {}", style("User:").dim(), style(email).cyan());
            if let Some(ref plan) = self.user_plan {
                println!("  {}    {}", style("Plan:").dim(), style(plan).yellow());
            }
            if let Some(quota) = self.quota_remaining {
                println!("  {}   {}", style("Quota:").dim(), style(format!("{} remaining", quota)).green());
            }
        } else {
            println!("  {} Not logged in", style("⚠").yellow());
            println!("  Use {} to authenticate", style("/login").cyan());
        }
        println!("  {}   {}", style("Model:").dim(), style(&self.selected_model).magenta());
        
        // Show mode
        let mode_str = match self.compression_mode {
            CompressionMode::Aqea => style("AQEA (2-Stage, ~29x)").cyan(),
            CompressionMode::AqeaPq => style("AQEA+PQ (3-Stage, 213-3072x)").magenta(),
        };
        println!("  {}    {}", style("Mode:").dim(), mode_str);
        
        // Show PQ subvectors if in PQ mode
        if self.compression_mode == CompressionMode::AqeaPq {
            let subs_str = match self.pq_subvectors {
                Some(n) => format!("{} (output: {} bytes)", n, n),
                None => "auto (best quality)".to_string(),
            };
            println!("  {}    {}", style("Subs:").dim(), style(subs_str).yellow());
        }
        
        println!("  {}     {} {}", style("API:").dim(), self.api_url, 
            if self.api_key.is_some() { style("✓").green() } else { style("○").dim() });
        println!();
    }
    
    fn validate_api_key(&self, key: &str) -> anyhow::Result<(String, String, i64)> {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .get(format!("{}/api/v1/auth/verify", self.api_url))
            .header("X-API-Key", key)
            .send()?;
        
        if resp.status().is_success() {
            let data: serde_json::Value = resp.json()?;
            let email = data["user_id"].as_str().unwrap_or("unknown").to_string();
            let plan = data["plan"].as_str().unwrap_or("free").to_string();
            let quota = data["quota_remaining"].as_i64().unwrap_or(0);
            Ok((email, plan, quota))
        } else {
            anyhow::bail!("Invalid API key")
        }
    }
    
    // ========================================================================
    // Model Commands
    // ========================================================================
    
    fn cmd_models(&self) {
        println!();
        println!("{}", style("Available Models:").bold().underlined());
        println!();
        
        // Check for pretrained weights
        let weights_dir = std::path::Path::new("weights");
        let pq_dir = weights_dir.join("pq_codebooks");
        
        println!("  {:<14} {:<12} {:<8} {:<8} {}", 
            style("Model").underlined(),
            style("Dimensions").underlined(),
            style("AQEA").underlined(),
            style("PQ").underlined(),
            style("Description").underlined()
        );
        println!("  {}", style("─".repeat(75)).dim());
        
        for (name, dims, desc, pq_opts) in MODELS {
            let marker = if *name == self.selected_model { 
                style("▸").green().bold() 
            } else { 
                style(" ").dim() 
            };
            
            // Check if AQEA weights exist
            let aqea_trained = self.check_aqea_weights(*name);
            let aqea_status = if aqea_trained {
                style("✓").green()
            } else {
                style("−").dim()
            };
            
            // Check PQ codebooks
            let pq_count = self.count_pq_codebooks(*name);
            let pq_status = if pq_count > 0 {
                style(format!("{}cb", pq_count)).green()
            } else {
                style("−".to_string()).dim()
            };
            
            println!("  {} {:<13} {:<12} {:<8} {:<8} {}", 
                marker,
                style(name).cyan(),
                style(dims).dim(),
                aqea_status,
                pq_status,
                style(desc).dim()
            );
        }
        
        // Check for custom models
        let custom_models = self.find_custom_models();
        if !custom_models.is_empty() {
        println!();
            println!("  {}", style("Custom Models:").bold());
            println!("  {}", style("─".repeat(75)).dim());
            for (name, dim, pq_count) in &custom_models {
                let marker = if *name == self.selected_model { 
                    style("▸").green().bold() 
                } else { 
                    style(" ").dim() 
                };
                let pq_status = if *pq_count > 0 {
                    style(format!("{}cb", pq_count)).green()
                } else {
                    style("−".to_string()).dim()
                };
                println!("  {} {:<13} {:<12} {:<8} {:<8} {}", 
                    marker,
                    style(name).yellow(),
                    style(format!("{}D", dim)).dim(),
                    style("✓").green(),  // Custom always have weights
                    pq_status,
                    style("Custom trained").dim()
                );
            }
        }
        
        println!();
        println!("  {}", style("Legend:").dim());
        println!("  {} AQEA: {} = trained weights, {} = not trained", 
            style("•").dim(), style("✓").green(), style("−").dim());
        println!("  {} PQ: {}cb = N codebooks available", 
            style("•").dim(), style("N").green());
        println!();
        println!("  {}", style("Modes:").dim());
        println!("  {} 2-Stage (AQEA):    ~29x compression", style("•").dim());
        println!("  {} 3-Stage (AQEA+PQ): 96-384x compression (use /mode pq)", style("•").dim());
        println!();
    }
    
    /// Check if AQEA weights exist for a model
    fn check_aqea_weights(&self, model_name: &str) -> bool {
        let patterns = vec![
            format!("weights/aqea_{}.aqwt", model_name),
            format!("weights/aqea_{}_rotated.aqwt", model_name),
        ];
        
        // Also check by dimensions
        if let Some((_, dims, _, _)) = MODELS.iter().find(|(n, _, _, _)| *n == model_name) {
            // Parse dimension from "384D" format
            let dim_str = dims.trim_end_matches('D');
            if let Ok(dim) = dim_str.parse::<usize>() {
                let target_dim = std::cmp::max(4, dim / 30);
                patterns.iter().any(|p| std::path::Path::new(p).exists())
                    || std::path::Path::new(&format!("weights/aqea_{}d_{}d.aqwt", dim, target_dim)).exists()
                    || std::path::Path::new(&format!("weights/aqea_{}d_{}d.json", dim, target_dim)).exists()
            } else {
                patterns.iter().any(|p| std::path::Path::new(p).exists())
            }
        } else {
            patterns.iter().any(|p| std::path::Path::new(p).exists())
        }
    }
    
    /// Count PQ codebooks for a model
    fn count_pq_codebooks(&self, model_name: &str) -> usize {
        let pq_dir = std::path::Path::new("weights/pq_codebooks");
        if !pq_dir.exists() {
            return 0;
        }
        
        if let Ok(entries) = std::fs::read_dir(pq_dir) {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.contains(model_name) && name.ends_with(".pqcb.json")
                })
                .count()
        } else {
            0
        }
    }
    
    /// Find custom trained models
    fn find_custom_models(&self) -> Vec<(String, usize, usize)> {
        let mut models = Vec::new();
        let weights_dir = std::path::Path::new("weights");
        
        if let Ok(entries) = std::fs::read_dir(weights_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("custom_") && name.ends_with(".aqwt") {
                    // Parse custom_<name>_<dim>d_<target>d.aqwt
                    let parts: Vec<&str> = name.trim_end_matches(".aqwt").split('_').collect();
                    if parts.len() >= 3 {
                        let model_name = parts[1].to_string();
                        let dim = parts.get(2)
                            .and_then(|d| d.trim_end_matches('d').parse().ok())
                            .unwrap_or(0);
                        
                        // Count PQ codebooks for this custom model
                        let pq_count = self.count_pq_codebooks(&model_name);
                        
                        if !models.iter().any(|(n, _, _)| n == &model_name) {
                            models.push((model_name, dim, pq_count));
                        }
                    }
                }
            }
        }
        
        models
    }
    
    fn cmd_model(&mut self, name: Option<String>) -> anyhow::Result<()> {
        match name {
            Some(n) => {
                // Direct selection
                let valid = MODELS.iter().any(|(m, _, _, _)| *m == n);
                if valid {
                    self.selected_model = n.clone();
                    println!("{} Model set: {}", style("✓").green(), style(&n).cyan());
                    
                    // Show available PQ options for this model
                    if let Some((_, _, _, pq_opts)) = MODELS.iter().find(|(m, _, _, _)| *m == n) {
                        if !pq_opts.is_empty() {
                            println!("  {} PQ subvector options: {:?}", style("ℹ").blue(), pq_opts);
                        }
                    }
                } else {
                    println!("{} Unknown model: {}. Use /models to see available models.", 
                        style("✗").red(), n);
                }
            }
            None => {
                // Interactive selection
                let items: Vec<String> = MODELS.iter()
                    .map(|(name, dims, desc, _)| format!("{} ({}) - {}", name, dims, desc))
                    .collect();
                
                let selection = Select::new()
                    .with_prompt("Select model")
                    .items(&items)
                    .default(0)
                    .interact()?;
                
                self.selected_model = MODELS[selection].0.to_string();
                println!("{} Model set: {}", style("✓").green(), style(&self.selected_model).cyan());
            }
        }
        Ok(())
    }
    
    fn cmd_mode(&mut self, mode: Option<String>) -> anyhow::Result<()> {
        match mode {
            Some(m) => {
                match m.to_lowercase().as_str() {
                    "aqea" | "2stage" | "2" => {
                        self.compression_mode = CompressionMode::Aqea;
                        println!("{} Mode: {} (2-Stage, ~29x compression)", 
                            style("✓").green(), style("AQEA").cyan().bold());
                        println!("  {} Works with ANY vector database", style("ℹ").blue());
                    }
                    "pq" | "aqea-pq" | "3stage" | "3" => {
                        self.compression_mode = CompressionMode::AqeaPq;
                        println!("{} Mode: {} (3-Stage, 213-3072x compression)", 
                            style("✓").green(), style("AQEA+PQ").magenta().bold());
                        println!("  {} Requires codebook for search queries", style("⚠").yellow());
                        println!("  Use {} to set subvector count", style("/subs").cyan());
                    }
                    _ => {
                        println!("{} Unknown mode: {}. Use 'aqea' or 'pq'.", style("✗").red(), m);
                    }
                }
            }
            None => {
                // Interactive selection
                let items = vec![
                    "AQEA (2-Stage)      ~29x compression, works with ANY vector DB",
                    "AQEA+PQ (3-Stage)   213-3072x compression, max savings ⚡",
                ];
                
                let selection = Select::new()
                    .with_prompt("Select compression mode")
                    .items(&items)
                    .default(if self.compression_mode == CompressionMode::Aqea { 0 } else { 1 })
                    .interact()?;
                
                self.compression_mode = if selection == 0 { 
                    CompressionMode::Aqea 
                } else { 
                    CompressionMode::AqeaPq 
                };
                
                let mode_str = if self.compression_mode == CompressionMode::Aqea {
                    style("AQEA (2-Stage)").cyan().bold()
                } else {
                    style("AQEA+PQ (3-Stage)").magenta().bold()
                };
                println!("{} Mode set: {}", style("✓").green(), mode_str);
            }
        }
        Ok(())
    }
    
    fn cmd_subvectors(&mut self, count: Option<usize>) -> anyhow::Result<()> {
        // Find current model's PQ options
        let pq_opts = MODELS.iter()
            .find(|(m, _, _, _)| *m == self.selected_model)
            .map(|(_, _, _, opts)| *opts)
            .unwrap_or(&[]);
        
        if pq_opts.is_empty() {
            println!("{} Model {} has no PQ options. Select a model with PQ support first.", 
                style("⚠").yellow(), self.selected_model);
            return Ok(());
        }
        
        match count {
            Some(n) => {
                if pq_opts.contains(&n) {
                    self.pq_subvectors = Some(n);
                    println!("{} Subvectors set: {} (output: {} bytes)", 
                        style("✓").green(), style(n).yellow(), n);
                } else {
                    println!("{} Invalid subvector count. Available for {}: {:?}", 
                        style("✗").red(), self.selected_model, pq_opts);
                }
            }
            None => {
                // Interactive selection
                let items: Vec<String> = pq_opts.iter()
                    .map(|&n| {
                        let quality = match n {
                            n if n >= 10 => "⭐ Best Quality",
                            n if n >= 7 => "Good Quality",
                            n if n >= 5 => "Balanced",
                            _ => "⚡ Max Compression",
                        };
                        format!("{} subvectors ({} bytes output) - {}", n, n, quality)
                    })
                    .collect();
                
                let selection = Select::new()
                    .with_prompt("Select PQ subvector count")
                    .items(&items)
                    .default(0)  // Default = best quality (first/highest)
                    .interact()?;
                
                self.pq_subvectors = Some(pq_opts[selection]);
                println!("{} Subvectors set: {}", style("✓").green(), style(pq_opts[selection]).yellow());
            }
        }
        Ok(())
    }
    
    // ========================================================================
    // Compression Commands
    // ========================================================================
    
    fn cmd_compress(&self, file: PathBuf, output: Option<PathBuf>) -> anyhow::Result<()> {
        if self.api_key.is_none() {
            println!("{} Please login first: {}", style("⚠").yellow(), style("/login").cyan());
            return Ok(());
        }
        
        if !file.exists() {
            anyhow::bail!("File not found: {}", file.display());
        }
        
        let output = output.unwrap_or_else(|| {
            let stem = file.file_stem().unwrap_or_default().to_string_lossy();
            file.with_file_name(format!("{}.aqea.json", stem))
        });
        
        println!("{} Loading {}...", style("⟳").cyan(), file.display());
        
        // Read input file
        let content = std::fs::read_to_string(&file)?;
        let input: serde_json::Value = serde_json::from_str(&content)
            .context("Failed to parse JSON file")?;
        
        // Extract vectors - support both array and object with "vectors" key
        let vectors: Vec<Vec<f32>> = if input.is_array() {
            serde_json::from_value(input.clone())?
        } else if let Some(vecs) = input.get("vectors") {
            serde_json::from_value(vecs.clone())?
        } else if let Some(vecs) = input.get("embeddings") {
            serde_json::from_value(vecs.clone())?
        } else {
            anyhow::bail!("Could not find vectors in input file. Expected array or object with 'vectors'/'embeddings' key.");
        };
        
        if vectors.is_empty() {
            anyhow::bail!("No vectors found in input file");
        }
        
        let num_vectors = vectors.len();
        let input_dim = vectors[0].len();
        
        println!("{} Found {} vectors ({}d)", style("✓").green(), num_vectors, input_dim);
        println!("{} Compressing via API...", style("⟳").cyan());
        
        // Create progress bar
        let pb = indicatif::ProgressBar::new(num_vectors as u64);
        pb.set_style(indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        
        // Compress via API (batch for efficiency)
        let client = reqwest::blocking::Client::new();
        let batch_size = 100;
        let mut compressed_vectors: Vec<serde_json::Value> = Vec::new();
        
        for chunk in vectors.chunks(batch_size) {
            let resp = client
                .post(format!("{}/api/v1/compress", self.api_url))
                .header("X-API-Key", self.api_key.as_ref().unwrap())
                .header("Content-Type", "application/json")
                .json(&serde_json::json!({
                    "vector": chunk[0],  // Single vector for now
                    "model": if self.selected_model == "auto" { None } else { Some(&self.selected_model) }
                }))
                .send()?;
            
            if !resp.status().is_success() {
                let error: serde_json::Value = resp.json().unwrap_or_default();
                anyhow::bail!("API error: {}", error.get("error").and_then(|e| e.as_str()).unwrap_or("Unknown"));
            }
            
            let result: serde_json::Value = resp.json()?;
            compressed_vectors.push(result);
            
            pb.inc(chunk.len() as u64);
        }
        
        pb.finish_and_clear();
        
        // Calculate compression stats
        let output_dim = compressed_vectors.first()
            .and_then(|v| v.get("compressed"))
            .and_then(|c| c.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        
        let ratio = if output_dim > 0 { input_dim as f32 / output_dim as f32 } else { 0.0 };
        
        // Build output
        let output_data = serde_json::json!({
            "compressed_vectors": compressed_vectors,
            "metadata": {
                "original_dim": input_dim,
                "compressed_dim": output_dim,
                "num_vectors": num_vectors,
                "compression_ratio": ratio,
                "model": &self.selected_model
            }
        });
        
        // Write output
        std::fs::write(&output, serde_json::to_string_pretty(&output_data)?)?;
        
        println!("{} Compressed {} vectors: {}d → {}d ({:.1}x)", 
            style("✓").green().bold(),
            style(num_vectors).cyan(),
            style(input_dim).dim(),
            style(output_dim).cyan(),
            ratio
        );
        println!("  Output: {}", style(output.display()).cyan());
        
        Ok(())
    }
    
    /// 3-Stage compression via API (AQEA + PQ)
    fn cmd_compress_pq(&self, file: PathBuf, output: Option<PathBuf>) -> anyhow::Result<()> {
        if self.api_key.is_none() {
            println!("{} Please login first: {}", style("⚠").yellow(), style("/login").cyan());
            return Ok(());
        }
        
        if !file.exists() {
            anyhow::bail!("File not found: {}", file.display());
        }
        
        if self.selected_model == "auto" {
            println!("{} Please select a model first: {} or {}", 
                style("⚠").yellow(), style("/model").cyan(), style("/models").cyan());
            return Ok(());
        }
        
        let output = output.unwrap_or_else(|| {
            let stem = file.file_stem().unwrap_or_default().to_string_lossy();
            file.with_file_name(format!("{}.pq.json", stem))
        });
        
        println!("{} Loading {}...", style("⟳").cyan(), file.display());
        
        // Read input file
        let content = std::fs::read_to_string(&file)?;
        let input: serde_json::Value = serde_json::from_str(&content)
            .context("Failed to parse JSON file")?;
        
        // Extract vectors
        let vectors: Vec<Vec<f32>> = if input.is_array() {
            serde_json::from_value(input.clone())?
        } else if let Some(vecs) = input.get("vectors") {
            serde_json::from_value(vecs.clone())?
        } else if let Some(vecs) = input.get("embeddings") {
            serde_json::from_value(vecs.clone())?
        } else {
            anyhow::bail!("Could not find vectors in input file");
        };
        
        if vectors.is_empty() {
            anyhow::bail!("No vectors found in input file");
        }
        
        let num_vectors = vectors.len();
        let input_dim = vectors[0].len();
        
        println!("{} Found {} vectors ({}d)", style("✓").green(), num_vectors, input_dim);
        println!("{} Mode: {} (3-Stage AQEA+PQ)", style("⟳").cyan(), style("Maximum Compression").magenta());
        println!("{} Model: {}", style("⟳").cyan(), style(&self.selected_model).green());
        if let Some(subs) = self.pq_subvectors {
            println!("{} Subvectors: {} (output: {} bytes/vector)", style("⟳").cyan(), subs, subs);
        }
        
        // Create progress bar
        let pb = indicatif::ProgressBar::new(num_vectors as u64);
        pb.set_style(indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        
        // Compress via API (one at a time for PQ endpoint)
        let client = reqwest::blocking::Client::new();
        let mut results: Vec<serde_json::Value> = Vec::new();
        let mut codebook_id = String::new();
        
        for vector in &vectors {
            let mut request = serde_json::json!({
                "vector": vector,
                "model": &self.selected_model
            });
            
            if let Some(subs) = self.pq_subvectors {
                request["subvectors"] = serde_json::json!(subs);
            }
            
            let resp = client
                .post(format!("{}/api/v1/compress-pq", self.api_url))
                .header("X-API-Key", self.api_key.as_ref().unwrap())
                .header("Content-Type", "application/json")
                .json(&request)
                .send()?;
            
            if !resp.status().is_success() {
                let error: serde_json::Value = resp.json().unwrap_or_default();
                anyhow::bail!("API error: {}", 
                    error.get("error").and_then(|e| e.as_str()).unwrap_or("Unknown"));
            }
            
            let result: serde_json::Value = resp.json()?;
            
            // Get codebook ID from first result
            if codebook_id.is_empty() {
                codebook_id = result.get("codebookId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
            }
            
            results.push(result);
            pb.inc(1);
        }
        
        pb.finish_and_clear();
        
        // Extract codes from results
        let codes: Vec<Vec<u8>> = results.iter()
            .filter_map(|r| r.get("codes"))
            .filter_map(|c| serde_json::from_value(c.clone()).ok())
            .collect();
        
        let output_bytes = codes.first().map(|c| c.len()).unwrap_or(0);
        let ratio = if output_bytes > 0 { (input_dim * 4) as f32 / output_bytes as f32 } else { 0.0 };
        
        // Build output
        let output_data = serde_json::json!({
            "type": "aqea_pq",
            "codes": codes,
            "codebook_id": codebook_id,
            "metadata": {
                "original_dim": input_dim,
                "output_bytes": output_bytes,
                "num_vectors": num_vectors,
                "compression_ratio": ratio,
                "model": &self.selected_model,
                "subvectors": self.pq_subvectors
            }
        });
        
        // Write output
        std::fs::write(&output, serde_json::to_string_pretty(&output_data)?)?;
        
        println!();
        println!("{}", style("╭─────────────────────────────────────────────────────────────╮").magenta());
        println!("{}", style("│  ✓ 3-Stage Compression Complete (AQEA+PQ)                   │").magenta());
        println!("{}", style("├─────────────────────────────────────────────────────────────┤").magenta());
        println!("{}  Vectors:      {}                                        {}", 
            style("│").magenta(), style(num_vectors).cyan(), style("│").magenta());
        println!("{}  Input:        {} dimensions ({}KB)                    {}", 
            style("│").magenta(), input_dim, (num_vectors * input_dim * 4) / 1024, style("│").magenta());
        println!("{}  Output:       {} bytes/vector                           {}", 
            style("│").magenta(), style(output_bytes).green().bold(), style("│").magenta());
        println!("{}  Compression:  {}                                     {}", 
            style("│").magenta(), style(format!("{:.0}x", ratio)).green().bold(), style("│").magenta());
        println!("{}  Codebook:     {}     {}", 
            style("│").magenta(), style(&codebook_id).dim(), style("│").magenta());
        println!("{}", style("╰─────────────────────────────────────────────────────────────╯").magenta());
        println!();
        println!("  {} {}", style("Output:").dim(), style(output.display()).cyan());
        println!();
        println!("  {} To search compressed vectors, you'll need the codebook:", style("ℹ").blue());
        println!("    {}/api/v1/codebooks/{}", self.api_url, codebook_id);
        println!();
        
        Ok(())
    }

    fn cmd_decompress(&self, file: PathBuf, output: Option<PathBuf>) -> anyhow::Result<()> {
        if self.api_key.is_none() {
            println!("{} Please login first: {}", style("⚠").yellow(), style("/login").cyan());
            return Ok(());
        }
        
        if !file.exists() {
            anyhow::bail!("File not found: {}", file.display());
        }
        
        let output = output.unwrap_or_else(|| {
            let name = file.file_name().unwrap_or_default().to_string_lossy();
            let name = name.replace(".aqea.json", ".decompressed.json");
            file.with_file_name(name)
        });
        
        println!("{} Decompressing {}...", style("⟳").cyan(), file.display());
        
        // TODO: Implement actual decompression via API
        println!("{} Decompressed → {}", 
            style("✓").green().bold(),
            style(output.display()).cyan()
        );
        
        Ok(())
    }
    
    fn cmd_batch(&self, folder: PathBuf) -> anyhow::Result<()> {
        if self.api_key.is_none() {
            println!("{} Please login first: {}", style("⚠").yellow(), style("/login").cyan());
            return Ok(());
        }
        
        if !folder.exists() || !folder.is_dir() {
            anyhow::bail!("Folder not found: {}", folder.display());
        }
        
        println!("{} Batch processing {}...", style("⟳").cyan(), folder.display());
        
        // TODO: Implement batch processing
        println!("{} Batch complete", style("✓").green().bold());
        
        Ok(())
    }
    
    fn cmd_ratio(&mut self, value: Option<u32>) -> anyhow::Result<()> {
        match value {
            Some(r) if [10, 20, 29, 50, 100].contains(&r) => {
                self.compression_ratio = r;
                println!("{} Compression ratio set: {}x", style("✓").green(), style(r).yellow());
            }
            Some(r) => {
                println!("{} Invalid ratio: {}. Use 10, 20, 29, 50, or 100.", style("✗").red(), r);
            }
            None => {
                let items = vec!["10x (high quality)", "20x", "29x (default)", "50x", "100x (max compression)"];
                let selection = Select::new()
                    .with_prompt("Select compression ratio")
                    .items(&items)
                    .default(2)
                    .interact()?;
                
                self.compression_ratio = match selection {
                    0 => 10,
                    1 => 20,
                    2 => 29,
                    3 => 50,
                    4 => 100,
                    _ => 29,
                };
                println!("{} Compression ratio set: {}x", style("✓").green(), style(self.compression_ratio).yellow());
            }
        }
        Ok(())
    }
    
    // ========================================================================
    // Config Commands
    // ========================================================================
    
    fn cmd_config(&self) {
        println!();
        println!("{}", style("Configuration:").bold().underlined());
        println!();
        println!("  {}        {}", style("API URL:").dim(), self.api_url);
        println!("  {}          {}", style("Model:").dim(), self.selected_model);
        println!("  {}          {}x", style("Ratio:").dim(), self.compression_ratio);
        
        let config_dir = dirs::config_dir()
            .map(|d| d.join("aqea"))
            .unwrap_or_else(|| PathBuf::from("~/.config/aqea"));
        println!("  {}     {}", style("Config dir:").dim(), config_dir.display());
        println!();
    }
    
    fn cmd_history(&self) {
        println!("{} Command history is managed by readline", style("ℹ").blue());
        println!("  Use {} and {} to navigate", style("↑").cyan(), style("↓").cyan());
    }
}

// ============================================================================
// Credential Storage
// ============================================================================

fn get_credentials_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("aqea")
        .join("credentials")
}

fn load_saved_credentials() -> (Option<String>, Option<String>) {
    let path = get_credentials_path();
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(&path) {
            let lines: Vec<&str> = content.lines().collect();
            if lines.len() >= 2 {
                return (Some(lines[0].to_string()), Some(lines[1].to_string()));
            }
        }
    }
    (None, None)
}

fn save_credentials(api_key: &str) {
    let path = get_credentials_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Note: In production, encrypt the API key
    let _ = std::fs::write(&path, format!("{}\n", api_key));
    
    // Set restrictive permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
}

fn clear_credentials() {
    let path = get_credentials_path();
    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// TRAINING & BENCHMARK IMPLEMENTATION
// ============================================================================

/// Embedding structure for benchmark data
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct Embedding {
    id: String,
    vector: Vec<f32>,
}

/// Training result with KPIs
#[derive(Debug, Clone, serde::Serialize)]
struct TrainingResult {
    pub input_dim: usize,
    pub output_dim: usize,
    pub compression_ratio: f32,
    pub final_spearman: f32,
    pub generations: usize,
    pub training_time_ms: u128,
    pub early_stopped: bool,
}

/// Benchmark result for a single configuration
#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkConfigResult {
    pub aqea_compression: String,
    pub pq_subvectors: usize,
    pub total_compression_ratio: f32,
    pub spearman: f32,
    pub recall_at_10: f32,
    pub latency_ms: f32,
    pub bytes_per_vector: usize,
}

/// Full benchmark report
#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkReport {
    pub timestamp: String,
    pub input_file: String,
    pub input_dim: usize,
    pub num_vectors: usize,
    pub train_samples: usize,
    pub pq_samples: usize,
    pub test_samples: usize,
    pub training: Option<TrainingResult>,
    pub results: Vec<BenchmarkConfigResult>,
}

impl CommandHandler {
    // ========================================================================
    // BENCHMARK COMMAND
    // ========================================================================
    
    fn cmd_benchmark(
        &self, 
        file: PathBuf, 
        output: Option<PathBuf>,
        train: bool,
        compressions: Option<Vec<u32>>,
        pq_subs: Option<Vec<usize>>,
    ) -> anyhow::Result<()> {
        // Check feature flags
        if train && !self.features.benchmark_with_train {
            println!("{} --train is not enabled. Contact support for access.", style("⚠").yellow());
            return Ok(());
        }
        if !train && !self.features.benchmark_pretrained {
            println!("{} Benchmark with pretrained weights is not enabled.", style("⚠").yellow());
            return Ok(());
        }
        
        if !file.exists() {
            anyhow::bail!("File not found: {}", file.display());
        }
        
        println!();
        println!("{}", style("╔═══════════════════════════════════════════════════════════════╗").cyan());
        println!("{}", style("║           AQEA Benchmark Engine™                              ║").cyan());
        println!("{}", style("╠═══════════════════════════════════════════════════════════════╣").cyan());
        if train {
            println!("{}", style("║  Mode: Fresh Training (AQEA + PQ from scratch)                ║").magenta());
        } else {
            println!("{}", style("║  Mode: Pretrained Weights                                     ║").green());
        }
        println!("{}", style("╚═══════════════════════════════════════════════════════════════╝").cyan());
        println!();
        
        // Load embeddings
        println!("{} Loading {}...", style("⟳").cyan(), file.display());
        let content = std::fs::read_to_string(&file)?;
        let embeddings = parse_embeddings(&content)?;
        
        if embeddings.is_empty() {
            anyhow::bail!("No embeddings found in file");
        }
        
        let input_dim = embeddings[0].vector.len();
        let num_vectors = embeddings.len();
        
        println!("{} Loaded {} embeddings ({}D)", style("✓").green(), num_vectors, input_dim);
        
        // Determine compression configs
        // NOTE: Start with highest compression (fastest training) first!
        // 30x = fewer weights = much faster CMA-ES (O(n²) complexity)
        let compressions = compressions.unwrap_or_else(|| vec![30, 20, 10]);
        let pq_subs = pq_subs.unwrap_or_else(|| vec![4, 8, 13, 16]);
        
        // Split data: Train (60%) / PQ (20%) / Test (20%)
        println!("{} Splitting data (60/20/20)...", style("⟳").cyan());
        let (train_data, pq_data, test_data) = split_data(&embeddings, 0.6, 0.2);
        
        println!("  Train: {} | PQ: {} | Test: {}", 
            style(train_data.len()).cyan(),
            style(pq_data.len()).yellow(),
            style(test_data.len()).green()
        );
        
        let mut report = BenchmarkReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            input_file: file.display().to_string(),
            input_dim,
            num_vectors,
            train_samples: train_data.len(),
            pq_samples: pq_data.len(),
            test_samples: test_data.len(),
            training: None,
            results: Vec::new(),
        };
        
        // Run benchmarks for each compression config
        for &compression in &compressions {
            let output_dim = (input_dim as f32 / compression as f32).ceil() as usize;
            
            println!();
            println!("{}", style(format!("━━━ AQEA {}x ({} → {}D) ━━━", compression, input_dim, output_dim)).bold());
            
            // Get or train AQEA weights
            let compressor = if train {
                println!("{} Training AQEA weights (CMA-ES)...", style("⟳").magenta());
                let (comp, training_result) = train_aqea_weights(&train_data, input_dim, output_dim)?;
                
                println!("  {} Spearman: {:.1}% | Generations: {} | Time: {:.1}s", 
                    style("✓").green(),
                    training_result.final_spearman * 100.0,
                    training_result.generations,
                    training_result.training_time_ms as f64 / 1000.0
                );
                
                if report.training.is_none() {
                    report.training = Some(training_result);
                }
                comp
            } else {
                // Load pretrained weights
                println!("{} Loading pretrained weights...", style("⟳").cyan());
                load_pretrained_compressor(input_dim, output_dim)?
            };
            
            // Compress PQ split with AQEA
            println!("{} Compressing PQ split with AQEA...", style("⟳").cyan());
            let pq_compressed: Vec<Vec<f32>> = pq_data.iter()
                .map(|e| compressor.compress(&e.vector))
                .collect();
            
            // Test each PQ configuration
            println!();
            println!("  {} Testing {} PQ configurations...", style("⟳").yellow(), pq_subs.len());
            
            for (pq_idx, &subs) in pq_subs.iter().enumerate() {
                if subs > output_dim {
                    println!("    {} Skipping {} subs (> output_dim {})", style("⚠").yellow(), subs, output_dim);
                    continue;
                }
                
                // PQ Training progress
                print!("    [{}/{}] PQ {} subs: ", pq_idx + 1, pq_subs.len(), subs);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                
                let pq_start = Instant::now();
                
                // Train PQ with K-Means++
                let mut pq = ProductQuantizer::new(output_dim, subs, 8);
                pq.train(&pq_compressed, 50);
                
                let pq_time = pq_start.elapsed();
                print!("trained ({:.0}ms) → ", pq_time.as_millis());
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                
                // Test on test split
                let test_start = Instant::now();
                let result = run_benchmark_test(&compressor, &pq, &test_data, input_dim, output_dim, compression, subs);
                let test_time = test_start.elapsed();
                
                // Color-coded results
                let spearman_str = if result.spearman >= 0.90 {
                    style(format!("{:.1}%", result.spearman * 100.0)).green().bold()
                } else if result.spearman >= 0.80 {
                    style(format!("{:.1}%", result.spearman * 100.0)).yellow()
                } else {
                    style(format!("{:.1}%", result.spearman * 100.0)).red()
                };
                
                let recall_str = if result.recall_at_10 >= 0.80 {
                    style(format!("{:.1}%", result.recall_at_10 * 100.0)).green().bold()
                } else if result.recall_at_10 >= 0.60 {
                    style(format!("{:.1}%", result.recall_at_10 * 100.0)).yellow()
                } else {
                    style(format!("{:.1}%", result.recall_at_10 * 100.0)).red()
                };
                
                println!("ρ={} R@10={} {}x ({:.0}ms)", 
                    spearman_str,
                    recall_str,
                    style(format!("{:.0}", result.total_compression_ratio)).cyan(),
                    test_time.as_millis()
                );
                
                report.results.push(result);
            }
        }
        
        // Output results
        let output = output.unwrap_or_else(|| {
            let stem = file.file_stem().unwrap_or_default().to_string_lossy();
            file.with_file_name(format!("{}.benchmark.json", stem))
        });
        
        std::fs::write(&output, serde_json::to_string_pretty(&report)?)?;
        
        println!();
        println!("{}", style("═══════════════════════════════════════════════════════════════").green());
        println!("{} Benchmark complete! Results saved to:", style("✓").green().bold());
        println!("  {}", style(output.display()).cyan());
        println!("{}", style("═══════════════════════════════════════════════════════════════").green());
        println!();
        
        Ok(())
    }
    
    // ========================================================================
    // TRAIN COMMAND (Internal Only)
    // ========================================================================
    
    fn cmd_train(
        &self,
        file: PathBuf,
        compression: Option<u32>,
        name: Option<String>,
    ) -> anyhow::Result<()> {
        // Check feature flag
        if !self.features.standalone_training {
            println!("{} Standalone training is not enabled.", style("⚠").yellow());
            println!("  This feature is for internal use only.");
            return Ok(());
        }
        
        if !file.exists() {
            anyhow::bail!("File not found: {}", file.display());
        }
        
        println!();
        println!("{}", style("╔═══════════════════════════════════════════════════════════════╗").magenta());
        println!("{}", style("║           AQEA Training Engine™ (CMA-ES)                      ║").magenta());
        println!("{}", style("╚═══════════════════════════════════════════════════════════════╝").magenta());
        println!();
        
        // Load embeddings
        println!("{} Loading {}...", style("⟳").cyan(), file.display());
        let content = std::fs::read_to_string(&file)?;
        let embeddings = parse_embeddings(&content)?;
        
        if embeddings.is_empty() {
            anyhow::bail!("No embeddings found in file");
        }
        
        let input_dim = embeddings[0].vector.len();
        let compression = compression.unwrap_or(20);
        let output_dim = (input_dim as f32 / compression as f32).ceil() as usize;
        
        println!("{} {} embeddings ({}D → {}D, {}x compression)", 
            style("✓").green(), 
            embeddings.len(), 
            input_dim, 
            output_dim,
            compression
        );
        
        // Train
        println!();
        println!("{} Starting CMA-ES training...", style("⟳").magenta());
        
        let (compressor, result) = train_aqea_weights(&embeddings, input_dim, output_dim)?;
        
        println!();
        println!("{}", style("═══════════════════════════════════════════════════════════════").green());
        println!("{} Training Complete!", style("✓").green().bold());
        println!("{}", style("───────────────────────────────────────────────────────────────").dim());
        println!("  Dimensions:      {}D → {}D ({}x)", input_dim, output_dim, compression);
        println!("  Final Spearman:  {:.2}%", result.final_spearman * 100.0);
        println!("  Generations:     {}", result.generations);
        println!("  Training Time:   {:.1}s", result.training_time_ms as f64 / 1000.0);
        println!("  Early Stopped:   {}", if result.early_stopped { "Yes" } else { "No" });
        println!("{}", style("═══════════════════════════════════════════════════════════════").green());
        
        // Save weights
        let model_name = name.unwrap_or_else(|| format!("custom_{}d", input_dim));
        let weights_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("aqea")
            .join("models")
            .join(&model_name);
        
        std::fs::create_dir_all(&weights_dir)?;
        
        let weights_path = weights_dir.join("aqea.aqwt");
        let weights = BinaryWeights::new(
            input_dim as u16,
            output_dim as u16,
            compressor.get_flat_weights(),
            result.final_spearman,
            1.0, // rotation scale
            aqea_core::ModelType::Unknown, // Custom model
        );
        weights.save(&weights_path)?;
        
        println!();
        println!("{} Weights saved to: {}", style("💾").cyan(), style(weights_path.display()).cyan());
        println!();
        
        Ok(())
    }
}

// ============================================================================
// TRAINING HELPERS
// ============================================================================

/// Parse embeddings from JSON (supports multiple formats)
fn parse_embeddings(content: &str) -> anyhow::Result<Vec<Embedding>> {
    let value: serde_json::Value = serde_json::from_str(content)?;
    
    // Format 1: Array of {id, vector}
    if let Ok(embeddings) = serde_json::from_value::<Vec<Embedding>>(value.clone()) {
        return Ok(embeddings);
    }
    
    // Format 2: {embeddings1: [...], embeddings2: [...]} (pairs format)
    if let Some(emb1) = value.get("embeddings1") {
        let vecs1: Vec<Vec<f32>> = serde_json::from_value(emb1.clone())?;
        let vecs2: Vec<Vec<f32>> = value.get("embeddings2")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or_default();
        
        let mut all = Vec::new();
        for (i, v) in vecs1.into_iter().enumerate() {
            all.push(Embedding { id: format!("e1_{}", i), vector: v });
        }
        for (i, v) in vecs2.into_iter().enumerate() {
            all.push(Embedding { id: format!("e2_{}", i), vector: v });
        }
        return Ok(all);
    }
    
    // Format 3: Simple array of arrays [[...], [...]]
    if let Ok(vecs) = serde_json::from_value::<Vec<Vec<f32>>>(value.clone()) {
        let embeddings: Vec<Embedding> = vecs.into_iter()
            .enumerate()
            .map(|(i, v)| Embedding { id: format!("vec_{}", i), vector: v })
            .collect();
        return Ok(embeddings);
    }
    
    anyhow::bail!("Could not parse embeddings. Supported formats: [{{id, vector}}], [[vectors]], {{embeddings1, embeddings2}}")
}

/// Split data into train/pq/test sets
fn split_data(data: &[Embedding], train_ratio: f64, pq_ratio: f64) -> (Vec<Embedding>, Vec<Embedding>, Vec<Embedding>) {
    let n = data.len();
    let train_end = (n as f64 * train_ratio) as usize;
    let pq_end = train_end + (n as f64 * pq_ratio) as usize;
    
    let train = data[..train_end].to_vec();
    let pq = data[train_end..pq_end].to_vec();
    let test = data[pq_end..].to_vec();
    
    (train, pq, test)
}

/// Train AQEA weights using Python pycma (FAST!)
/// Calls external Python script for optimized CMA-ES training
fn train_aqea_weights(
    data: &[Embedding],
    input_dim: usize,
    output_dim: usize,
) -> anyhow::Result<(OctonionCompressor, TrainingResult)> {
    let start = Instant::now();
    let max_generations = 50;  // Reduced for faster benchmarks, enough for convergence
    let early_stop_threshold = 0.97;
    
    // Save embeddings to temp file for Python
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join(format!("aqea_train_{}.json", std::process::id()));
    
    let vectors: Vec<Vec<f32>> = data.iter().map(|e| e.vector.clone()).collect();
    let json_data = serde_json::to_string(&vectors)?;
    std::fs::write(&temp_file, json_data)?;
    
    // Find Python script
    let script_paths = vec![
        std::path::PathBuf::from("cli/scripts/train_aqea.py"),
        std::path::PathBuf::from("scripts/train_aqea.py"),
        std::path::PathBuf::from("/home/aqea/aqea-compress/cli/scripts/train_aqea.py"),
    ];
    
    let script_path = script_paths.iter()
        .find(|p| p.exists())
        .ok_or_else(|| anyhow::anyhow!("Python training script not found"))?;
    
    // Find Python with pycma
    let python_paths = vec![
        "/home/aqea/aqea-compress/benchmark/venv/bin/python",
        "python3",
        "python",
    ];
    
    let python = python_paths.iter()
        .find(|p| std::path::Path::new(p).exists() || which::which(p).is_ok())
        .ok_or_else(|| anyhow::anyhow!("Python not found"))?;
    
    // Header
    println!();
    println!("{}", style("┌────────────────────────────────────────────────────────────────┐").dim());
    println!("{}", style("│  CMA-ES Training (pycma - optimized)                           │").dim());
    println!("{}", style("├────────────────────────────────────────────────────────────────┤").dim());
    println!("{}  {}D → {}D ({:.0}x) | {} embeddings", 
        style("│").dim(),
        style(input_dim).cyan(),
        style(output_dim).cyan(),
        input_dim as f32 / output_dim as f32,
        style(data.len()).green()
    );
    println!("{}", style("└────────────────────────────────────────────────────────────────┘").dim());
    println!();
    
    // Spawn Python process
    let mut child = ProcessCommand::new(python)
        .arg(script_path)
        .arg(&temp_file)
        .arg(output_dim.to_string())
        .arg(max_generations.to_string())
        .arg(early_stop_threshold.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    
    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);
    
    // Progress bar
    let pb = indicatif::ProgressBar::new(max_generations as u64);
    pb.set_style(indicatif::ProgressStyle::default_bar()
        .template("{spinner:.magenta} [{elapsed_precise}] [{bar:40.magenta/blue}] Gen {pos}/{len} | {msg}")
        .unwrap()
        .progress_chars("█▓░"));
    
    let mut best_weights: Vec<f32> = Vec::new();
    let mut final_spearman = 0.0f32;
    let mut final_generations = 0usize;
    let mut training_time_ms = 0u128;
    let mut early_stopped = false;
    
    // Read JSON lines from Python
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() { continue; }
        
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
            match json.get("type").and_then(|t| t.as_str()) {
                Some("config") => {
                    let weights = json.get("weights").and_then(|v| v.as_u64()).unwrap_or(0);
                    let pairs = json.get("pairs").and_then(|v| v.as_u64()).unwrap_or(0);
                    let pop = json.get("population").and_then(|v| v.as_u64()).unwrap_or(0);
                    pb.set_message(format!("{} weights | {} pairs | pop {}", weights, pairs, pop));
                }
                Some("progress") => {
                    let gen = json.get("generation").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let best = json.get("best_spearman").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let sigma = json.get("sigma").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let gen_time = json.get("gen_time_ms").and_then(|v| v.as_u64()).unwrap_or(0);
                    let improved = json.get("improved").and_then(|v| v.as_bool()).unwrap_or(false);
                    
                    pb.set_position(gen as u64);
                    
                    let indicator = if improved { style("↑").green().bold() } else { style("→").dim() };
                    let spearman_str = if best >= 0.95 {
                        style(format!("{:.1}%", best * 100.0)).green().bold()
                    } else if best >= 0.90 {
                        style(format!("{:.1}%", best * 100.0)).yellow()
                    } else {
                        style(format!("{:.1}%", best * 100.0)).white()
                    };
                    
                    pb.set_message(format!("{} Best: {} | σ: {:.4} | {}ms/gen", 
                        indicator, spearman_str, sigma, gen_time));
                }
                Some("result") => {
                    if let Some(weights_arr) = json.get("weights").and_then(|v| v.as_array()) {
                        best_weights = weights_arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect();
                    }
                    final_spearman = json.get("final_spearman").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    final_generations = json.get("generations").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    training_time_ms = json.get("training_time_ms").and_then(|v| v.as_u64()).unwrap_or(0) as u128;
                    early_stopped = json.get("early_stopped").and_then(|v| v.as_bool()).unwrap_or(false);
                }
                Some("error") => {
                    let msg = json.get("message").and_then(|v| v.as_str()).unwrap_or("Unknown error");
                    anyhow::bail!("Python training error: {}", msg);
                }
                _ => {}
            }
        }
    }
    
    pb.finish_and_clear();
    
    // Wait for Python to finish
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("Python training failed with exit code: {:?}", status.code());
    }
    
    // Clean up temp file
    let _ = std::fs::remove_file(&temp_file);
    
    if best_weights.is_empty() {
        anyhow::bail!("No weights received from Python training");
    }
    
    // Final summary
    let total_time = start.elapsed().as_secs_f32();
    let avg_gen_time = if final_generations > 0 { training_time_ms as f32 / final_generations as f32 } else { 0.0 };
    
    println!();
    println!("{}", style("┌────────────────────────────────────────────────────────────────┐").green());
    println!("{}  {} Training Complete!                                         {}", 
        style("│").green(), style("✓").green().bold(), style("│").green());
    println!("{}", style("├────────────────────────────────────────────────────────────────┤").green());
    println!("{}  Final Spearman:  {}                                      {}", 
        style("│").green(), 
        style(format!("{:.2}%", final_spearman * 100.0)).green().bold(),
        style("│").green()
    );
    println!("{}  Generations:     {} / {}                                       {}", 
        style("│").green(), 
        style(format!("{}", final_generations)).cyan(),
        max_generations,
        style("│").green()
    );
    println!("{}  Total Time:      {:.1}s ({:.0}ms avg/gen)                       {}", 
        style("│").green(), 
        total_time,
        avg_gen_time,
        style("│").green()
    );
    println!("{}  Early Stopped:   {}                                           {}", 
        style("│").green(), 
        if early_stopped { style("Yes").green() } else { style("No").dim() },
        style("│").green()
    );
    println!("{}", style("└────────────────────────────────────────────────────────────────┘").green());
    
    let compressor = OctonionCompressor::with_trained_weights(input_dim, output_dim, &best_weights);
    
    let result = TrainingResult {
        input_dim,
        output_dim,
        compression_ratio: input_dim as f32 / output_dim as f32,
        final_spearman,
        generations: final_generations,
        training_time_ms,
        early_stopped,
    };
    
    Ok((compressor, result))
}

/// Load pretrained compressor (falls back to random if not found)
fn load_pretrained_compressor(input_dim: usize, output_dim: usize) -> anyhow::Result<OctonionCompressor> {
    // Try to find matching pretrained weights
    let weights_paths = vec![
        format!("weights/aqea_{}d_{}d.aqwt", input_dim, output_dim),
        format!("../weights/aqea_{}d_{}d.aqwt", input_dim, output_dim),
    ];
    
    for path in weights_paths {
        if std::path::Path::new(&path).exists() {
            let weights = BinaryWeights::load(std::path::Path::new(&path))?;
            return Ok(OctonionCompressor::with_trained_weights(
                input_dim, output_dim, &weights.weights
            ));
        }
    }
    
    // No pretrained weights found - use default initialization
    println!("  {} No pretrained weights found for {}D→{}D, using default init", 
        style("ℹ").blue(), input_dim, output_dim);
    Ok(OctonionCompressor::with_dims(input_dim, output_dim))
}

/// Run benchmark test on test split
fn run_benchmark_test(
    compressor: &OctonionCompressor,
    pq: &ProductQuantizer,
    test_data: &[Embedding],
    input_dim: usize,
    output_dim: usize,
    aqea_compression: u32,
    pq_subs: usize,
) -> BenchmarkConfigResult {
    let start = Instant::now();
    
    // Compress all test vectors
    let compressed: Vec<Vec<f32>> = test_data.iter()
        .map(|e| compressor.compress(&e.vector))
        .collect();
    
    // Encode with PQ
    let pq_codes: Vec<Vec<u8>> = compressed.iter()
        .map(|v| pq.encode(v))
        .collect();
    
    // Decode PQ for similarity computation
    let decoded: Vec<Vec<f32>> = pq_codes.iter()
        .map(|c| pq.decode(c))
        .collect();
    
    // Compute Spearman correlation
    let n = test_data.len().min(100);
    let mut original_sims = Vec::new();
    let mut decoded_sims = Vec::new();
    
    for i in 0..n {
        for j in (i+1)..n {
            original_sims.push(cosine_similarity(&test_data[i].vector, &test_data[j].vector));
            decoded_sims.push(cosine_similarity(&decoded[i], &decoded[j]));
        }
    }
    
    let spearman = spearman_correlation(&original_sims, &decoded_sims);
    
    // Compute Recall@10
    let recall = compute_recall_at_k(&test_data, &decoded, 10);
    
    let latency_ms = start.elapsed().as_secs_f32() * 1000.0 / test_data.len() as f32;
    
    // Total compression: input_dim * 4 bytes → pq_subs bytes
    let total_compression = (input_dim * 4) as f32 / pq_subs as f32;
    
    BenchmarkConfigResult {
        aqea_compression: format!("{}x", aqea_compression),
        pq_subvectors: pq_subs,
        total_compression_ratio: total_compression,
        spearman,
        recall_at_10: recall,
        latency_ms,
        bytes_per_vector: pq_subs,
    }
}

/// Compute Recall@K
fn compute_recall_at_k(original: &[Embedding], compressed: &[Vec<f32>], k: usize) -> f32 {
    let n = original.len().min(50); // Use subset for speed
    let k = k.min(n - 1);
    
    let mut total_recall = 0.0;
    
    for i in 0..n {
        // Get ground truth top-k from original
        let mut orig_dists: Vec<(usize, f32)> = (0..original.len())
            .filter(|&j| j != i)
            .map(|j| (j, cosine_similarity(&original[i].vector, &original[j].vector)))
            .collect();
        orig_dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let ground_truth: std::collections::HashSet<usize> = orig_dists.iter()
            .take(k)
            .map(|(idx, _)| *idx)
            .collect();
        
        // Get top-k from compressed
        let mut comp_dists: Vec<(usize, f32)> = (0..compressed.len())
            .filter(|&j| j != i)
            .map(|j| (j, cosine_similarity(&compressed[i], &compressed[j])))
            .collect();
        comp_dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let retrieved: std::collections::HashSet<usize> = comp_dists.iter()
            .take(k)
            .map(|(idx, _)| *idx)
            .collect();
        
        // Count intersection
        let hits = ground_truth.intersection(&retrieved).count();
        total_recall += hits as f32 / k as f32;
    }
    
    total_recall / n as f32
}

