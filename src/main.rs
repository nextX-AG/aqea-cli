//! AQEA CLI - Aurora Quantum Encoding Algorithm
//!
//! Usage:
//!   aqea                              # Start interactive REPL
//!   aqea auth login                   # Authenticate with AQEA
//!   aqea compress embeddings.json     # Compress embeddings
//!   aqea test                         # Run quality benchmark
//!   aqea models                       # List available models

use clap::{Parser, Subcommand};
use anyhow::{Result, Context};
use std::path::PathBuf;
use std::io::{self, Read, Write};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use aqea_core::{
    OctonionCompressor, Compressor, BinaryWeights, 
    PreQuantifier, cosine_similarity, spearman_correlation
};

mod config;
mod auth;
mod csv_support;
mod test_data;
mod validate;
mod repl;

use config::Config;
use csv_support::{detect_format, parse_csv, vectors_to_csv_data, write_csv, CsvData, FileFormat};

// ============================================================================
// CLI STRUCTURE
// ============================================================================

#[derive(Parser)]
#[command(name = "aqea")]
#[command(author = "AQEA <hello@aqea.ai>")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "AQEA - Aurora Quantum Encoding Algorithm\n\nRun without arguments for interactive mode.", long_about = None)]
#[command(after_help = "Interactive Mode:\n  $ aqea         Start interactive REPL\n\nDocumentation: https://docs.aqea.ai")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Authenticate with AQEA
    Auth {
        #[command(subcommand)]
        action: AuthCommands,
    },

    /// Compress embeddings (requires authentication)
    /// 
    /// Embeddings are sent to the AQEA API for compression.
    /// Your data is processed securely and not stored.
    Compress {
        /// Input file (JSON or CSV). Use - for stdin.
        input: Option<PathBuf>,

        /// Output file. Omit for stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Model to use (e.g., text-mpnet, openai-large)
        #[arg(short, long)]
        model: Option<String>,

        /// Input/output format: json, csv, or auto (default: auto)
        #[arg(long, default_value = "auto")]
        format: String,

        /// CSV has header row (auto-detected if not specified)
        #[arg(long)]
        header: Option<bool>,

        /// CSV column index containing IDs (0-based, auto-detected if not specified)
        #[arg(long)]
        id_column: Option<usize>,
    },

    /// Run compression quality test
    Test {
        /// Model to test (default: text-mpnet)
        #[arg(short, long)]
        model: Option<String>,

        /// Custom embeddings file for testing
        #[arg(long)]
        custom: Option<PathBuf>,

        /// Custom similarity pairs file
        #[arg(long)]
        pairs: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// List available models
    Models {
        /// Show detailed info for a specific model
        model: Option<String>,
    },

    /// [Internal] Show info about a weights file
    #[command(hide = true)]  // Hidden - requires local weights
    Info {
        /// Weights file (.aqwt)
        weights: PathBuf,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },

    /// Show current usage statistics
    Usage {
        /// Month to show (YYYY-MM format)
        #[arg(long)]
        month: Option<String>,
    },
    
    /// [Internal] Validate compression quality locally
    /// 
    /// NOTE: This command requires local weights files and is intended
    /// for internal development and testing only.
    #[command(hide = true)]  // Hidden from --help for external users
    Validate {
        /// Your embeddings/pairs file (JSON)
        #[arg(short, long)]
        data: PathBuf,
        
        /// Weights file (.aqwt)
        #[arg(short, long)]
        weights: PathBuf,
        
        /// Number of PQ subvectors for full pipeline test (optional)
        #[arg(long)]
        pq_subs: Option<usize>,
        
        /// Output results as JSON
        #[arg(long)]
        json: bool,
        
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
}

#[derive(Subcommand)]
enum AuthCommands {
    /// Login to AuroraQE (opens browser)
    Login,
    /// Logout and remove credentials
    Logout,
    /// Show current authentication status
    Status,
    /// Display API key (masked)
    Token,
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Set a config value
    Set {
        key: String,
        value: String,
    },
    /// Get a config value
    Get {
        key: String,
    },
    /// List all config values
    List,
    /// Show config file path
    Path,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    // If no command given, start interactive REPL
    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            return repl::run_repl(repl::ReplConfig::default());
        }
    };

    match command {
        Commands::Auth { action } => match action {
            AuthCommands::Login => auth_login()?,
            AuthCommands::Logout => auth_logout()?,
            AuthCommands::Status => auth_status()?,
            AuthCommands::Token => auth_token()?,
        },
        Commands::Compress { input, output, model, format, header, id_column } => {
            compress_cmd(input, output, model, format, header, id_column)?;
        },
        Commands::Test { model, custom, pairs, verbose } => {
            test_cmd(model, custom, pairs, verbose)?;
        },
        Commands::Models { model } => {
            models_cmd(model)?;
        },
        Commands::Info { weights } => {
            info_cmd(weights)?;
        },
        Commands::Config { action } => match action {
            ConfigCommands::Set { key, value } => config_set(&key, &value)?,
            ConfigCommands::Get { key } => config_get(&key)?,
            ConfigCommands::List => config_list()?,
            ConfigCommands::Path => config_path()?,
        },
        Commands::Usage { month } => {
            usage_cmd(month)?;
        },
        Commands::Validate { data, weights, pq_subs, json, verbose } => {
            validate_cmd(&data, &weights, pq_subs, json, verbose)?;
        },
    }

    Ok(())
}

// ============================================================================
// VALIDATE COMMAND - User Self-Validation
// ============================================================================

fn validate_cmd(
    data: &PathBuf,
    weights: &PathBuf,
    pq_subs: Option<usize>,
    json_output: bool,
    verbose: bool,
) -> Result<()> {
    if !json_output {
        println!();
        println!("{}", style("🔬 AQEA Validation - Test on YOUR Data").bold().cyan());
        println!("{}", style("═".repeat(60)).dim());
        println!();
        println!("  💡 This validates compression quality on YOUR embeddings.");
        println!("  💡 No data leaves your machine. Results you can trust!");
        println!();
    }
    
    let result = validate::run_validation(weights, data, pq_subs, verbose)
        .map_err(|e| anyhow::anyhow!(e))?;
    
    if json_output {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        validate::print_summary(&result);
    }
    
    Ok(())
}

// ============================================================================
// AUTH COMMANDS
// ============================================================================

fn auth_login() -> Result<()> {
    println!();
    println!("{}", style("🔐 AQEA Authentication").bold());
    println!();
    
    // Check if already logged in
    let config = Config::load()?;
    if config.api_key.is_some() {
        println!("{}  You are already logged in.", style("ℹ").blue());
        println!("   Run {} to see your status.", style("aqea auth status").cyan());
        println!("   Run {} to logout first.", style("aqea auth logout").cyan());
        return Ok(());
    }

    println!("To authenticate, you need an API key from AQEA.");
    println!();
    println!("1. Go to: {}", style("https://aqea.ai/dashboard/api-keys").cyan().underlined());
    println!("2. Create a new API key");
    println!("3. Paste it below");
    println!();
    
    print!("{} ", style("API Key:").bold());
    io::stdout().flush()?;
    
    let mut api_key = String::new();
    io::stdin().read_line(&mut api_key)?;
    let api_key = api_key.trim().to_string();
    
    if api_key.is_empty() {
        anyhow::bail!("No API key provided");
    }
    
    if !api_key.starts_with("aqea_") {
        println!();
        println!("{} API key should start with 'aqea_'", style("⚠").yellow());
        println!("   Are you sure this is correct? (y/N)");
        
        let mut confirm = String::new();
        io::stdin().read_line(&mut confirm)?;
        if !confirm.trim().to_lowercase().starts_with('y') {
            anyhow::bail!("Aborted");
        }
    }
    
    // Save the key
    let mut config = Config::load()?;
    config.api_key = Some(api_key);
    config.save()?;
    
    println!();
    println!("{} Successfully authenticated!", style("✓").green().bold());
    println!("  Your API key has been saved to {}", style(Config::path()?.display()).dim());
    println!();
    println!("  Try: {}", style("aqea test").cyan());
    
    Ok(())
}

fn auth_logout() -> Result<()> {
    let mut config = Config::load()?;
    
    if config.api_key.is_none() {
        println!("{}  Not currently logged in.", style("ℹ").blue());
        return Ok(());
    }
    
    config.api_key = None;
    config.save()?;
    
    println!("{} Logged out successfully.", style("✓").green());
    Ok(())
}

fn auth_status() -> Result<()> {
    let config = Config::load()?;
    
    println!();
    println!("{}", style("🔐 Authentication Status").bold());
    println!();
    
    match &config.api_key {
        Some(key) => {
            let masked = format!("{}...{}", &key[..8.min(key.len())], &key[key.len().saturating_sub(4)..]);
            println!("  Status:    {}", style("Authenticated").green());
            println!("  API Key:   {}", style(&masked).dim());
            println!();
            println!("  Run {} to test your connection.", style("aqea test").cyan());
        }
        None => {
            println!("  Status:    {}", style("Not authenticated").yellow());
            println!();
            println!("  Run {} to login.", style("aqea auth login").cyan());
        }
    }
    
    Ok(())
}

fn auth_token() -> Result<()> {
    let config = Config::load()?;
    
    match &config.api_key {
        Some(key) => {
            let masked = format!("{}...{}", &key[..8.min(key.len())], &key[key.len().saturating_sub(4)..]);
            println!("{}", masked);
        }
        None => {
            eprintln!("{} Not authenticated. Run: aqea auth login", style("Error:").red());
            std::process::exit(1);
        }
    }
    
    Ok(())
}

// ============================================================================
// COMPRESS COMMAND (API-BASED - SECURE)
// ============================================================================

fn compress_cmd(
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    model: Option<String>,
    format: String,
    header: Option<bool>,
    id_column: Option<usize>,
) -> Result<()> {
    // Check authentication first
    let config = Config::load()?;
    let api_key = config.api_key.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "{}\n\n  Run: {}\n  Get your API key at: {}",
            style("Not authenticated").red().bold(),
            style("aqea auth login").cyan(),
            style("https://aqea.ai/dashboard/api-keys").cyan().underlined()
        )
    })?;

    // Read input
    let input_path = input.clone();
    let input_data: String = match &input {
        Some(path) if path.to_string_lossy() != "-" => {
            std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read {}", path.display()))?
        }
        _ => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    // Detect or determine format
    let input_format = match format.as_str() {
        "json" => FileFormat::Json,
        "csv" => FileFormat::Csv,
        "auto" | _ => {
            if let Some(path) = &input_path {
                detect_format(path, Some(&input_data))
            } else {
                let trimmed = input_data.trim();
                if trimmed.starts_with('[') || trimmed.starts_with('{') {
                    FileFormat::Json
                } else {
                    FileFormat::Csv
                }
            }
        }
    };

    // Determine output format
    let output_format = if let Some(output_path) = &output {
        if let Some(ext) = output_path.extension().and_then(|e| e.to_str()) {
            match ext.to_lowercase().as_str() {
                "csv" | "tsv" => FileFormat::Csv,
                "json" => FileFormat::Json,
                _ => input_format,
            }
        } else {
            input_format
        }
    } else {
        input_format
    };

    // Parse input based on format
    let (vectors, ids): (Vec<Vec<f32>>, Option<Vec<String>>) = match input_format {
        FileFormat::Json => {
            let vecs: Vec<Vec<f32>> = serde_json::from_str(&input_data)
                .context("Failed to parse JSON. Expected array of number arrays.")?;
            (vecs, None)
        }
        FileFormat::Csv => {
            let csv_data = parse_csv(&input_data, header, id_column)
                .context("Failed to parse CSV")?;
            let ids = if csv_data.has_ids() {
                Some(csv_data.ids.into_iter().filter_map(|id| id).collect())
            } else {
                None
            };
            (csv_data.vectors, ids)
        }
    };

    if vectors.is_empty() {
        anyhow::bail!("No vectors found in input");
    }

    let input_dim = vectors[0].len();
    let num_vectors = vectors.len();
    let format_str = match input_format {
        FileFormat::Json => "JSON",
        FileFormat::Csv => "CSV",
    };
    eprintln!("{}  Loaded {} vectors ({}D) from {}",
        style("📥").cyan(), num_vectors, input_dim, format_str);

    // Compress via API
    eprintln!("{}  Compressing via AQEA API...", style("📡").cyan());
    
    let api_url = config.api_url();
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    
    // Progress bar
    let pb = ProgressBar::new(num_vectors as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    // Batch compression for efficiency
    let batch_size = 100;
    let mut compressed: Vec<Vec<f32>> = Vec::with_capacity(num_vectors);
    let mut model_name = String::from("auto");
    let mut output_dim = 0;
    
    for chunk in vectors.chunks(batch_size) {
        let request_body = serde_json::json!({
            "vectors": chunk,
            "model": model.as_deref()
        });
        
        let resp = client
            .post(format!("{}/api/v1/compress/batch", api_url))
            .header("X-API-Key", &api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    anyhow::anyhow!("API request timed out. Try with smaller batches or check your connection.")
                } else if e.is_connect() {
                    anyhow::anyhow!(
                        "{}\n\n  Please check:\n  1. Your internet connection\n  2. API status at https://status.aqea.ai",
                        style("Cannot connect to AQEA API").red().bold()
                    )
                } else {
                    anyhow::anyhow!("API request failed: {}", e)
                }
            })?;
        
        if !resp.status().is_success() {
            let status = resp.status();
            let error_body: serde_json::Value = resp.json().unwrap_or_default();
            let error_msg = error_body.get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("Unknown error");
            
            match status.as_u16() {
                401 => anyhow::bail!(
                    "{}: {}\n\n  Run: {}",
                    style("Authentication failed").red().bold(),
                    error_msg,
                    style("aqea auth login").cyan()
                ),
                403 => anyhow::bail!(
                    "{}: {}\n\n  Upgrade at: {}",
                    style("Quota exceeded").red().bold(),
                    error_msg,
                    style("https://aqea.ai/pricing").cyan().underlined()
                ),
                429 => anyhow::bail!(
                    "{}: {}\n\n  Please wait and try again.",
                    style("Rate limited").yellow().bold(),
                    error_msg
                ),
                _ => anyhow::bail!("API error ({}): {}", status, error_msg),
            }
        }
        
        let result: serde_json::Value = resp.json()?;
        
        // Extract model info from first response
        if model_name == "auto" {
            model_name = result.get("model")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown")
                .to_string();
        }
        
        // Extract compressed vectors
        if let Some(batch_compressed) = result.get("compressed").and_then(|c| c.as_array()) {
            for vec_value in batch_compressed {
                if let Some(vec) = vec_value.as_array() {
                    let floats: Vec<f32> = vec.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    if output_dim == 0 && !floats.is_empty() {
                        output_dim = floats.len();
                    }
                    compressed.push(floats);
                }
            }
        }
        
        pb.inc(chunk.len() as u64);
    }
    
    pb.finish_and_clear();

    if compressed.is_empty() {
        anyhow::bail!("No compressed vectors received from API");
    }

    let ratio = input_dim as f32 / output_dim as f32;
    eprintln!("{}  Using model: {}", style("📦").cyan(), style(&model_name).green());
    eprintln!("{}  Compressed to {}D ({:.1}x compression)",
        style("✅").green(), output_dim, ratio);

    // Write output in appropriate format
    let output_str = match output_format {
        FileFormat::Json => {
            serde_json::to_string(&compressed)?
        }
        FileFormat::Csv => {
            let csv_data = vectors_to_csv_data(compressed, ids);
            write_csv(&csv_data, true)?
        }
    };

    match output {
        Some(path) if path.to_string_lossy() != "-" => {
            std::fs::write(&path, &output_str)?;
            let fmt_str = match output_format {
                FileFormat::Json => "JSON",
                FileFormat::Csv => "CSV",
            };
            eprintln!("{}  Saved to {} ({})", style("💾").cyan(), path.display(), fmt_str);
        }
        _ => {
            println!("{}", output_str);
        }
    }

    Ok(())
}

fn find_model_for_dim(dim: usize, preferred: Option<&str>) -> Result<(String, String, f32)> {
    let models = vec![
        ("text-minilm", 384, "aqea_text_minilm_384d.aqwt", 0.971),
        ("text-mpnet", 768, "aqea_text_mpnet_768d.aqwt", 0.983),
        ("text-e5large", 1024, "aqea_text_e5large_1024d.aqwt", 0.982),
        ("mistral-embed", 1024, "aqea_mistral_embed_1024d.aqwt", 0.962),
        ("openai-small", 1536, "aqea_openai_small_1536d.aqwt", 0.950),
        ("openai-large", 3072, "aqea_openai_large_3072d.aqwt", 0.950),
        ("audio-wav2vec2", 768, "aqea_audio_wav2vec2_768d.aqwt", 0.971),
    ];
    
    // If preferred model specified, use it
    if let Some(name) = preferred {
        for (model_name, model_dim, file, spearman) in &models {
            if *model_name == name {
                if *model_dim != dim {
                    eprintln!("{} Model {} expects {}D input, got {}D", 
                        style("⚠").yellow(), name, model_dim, dim);
                }
                return Ok((model_name.to_string(), file.to_string(), *spearman));
            }
        }
        anyhow::bail!("Unknown model: {}. Run 'aqe models' to see available models.", name);
    }
    
    // Auto-detect by dimension
    for (model_name, model_dim, file, spearman) in &models {
        if *model_dim == dim {
            return Ok((model_name.to_string(), file.to_string(), *spearman));
        }
    }
    
    anyhow::bail!(
        "No model found for {}D embeddings. Available dimensions: 384, 768, 1024, 1536, 3072",
        dim
    );
}

fn find_weights_path(filename: &str) -> Result<PathBuf> {
    // Check common locations
    let paths = vec![
        PathBuf::from(format!("weights/{}", filename)),
        PathBuf::from(format!("../weights/{}", filename)),
        dirs::data_dir().map(|d| d.join("aqe").join("weights").join(filename)).unwrap_or_default(),
    ];
    
    for path in paths {
        if path.exists() {
            return Ok(path);
        }
    }
    
    anyhow::bail!(
        "Weights file not found: {}. Run 'aqea models download {}' to download.",
        filename,
        filename.replace(".aqwt", "")
    );
}

// ============================================================================
// TEST COMMAND (API-BASED - SECURE)
// ============================================================================

fn test_cmd(
    model: Option<String>,
    custom: Option<PathBuf>,
    _pairs: Option<PathBuf>,
    verbose: bool,
) -> Result<()> {
    // Check authentication first
    let config = Config::load()?;
    let api_key = config.api_key.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "{}\n\n  Run: {}\n  Get your API key at: {}",
            style("Not authenticated").red().bold(),
            style("aqea auth login").cyan(),
            style("https://aqea.ai/dashboard/api-keys").cyan().underlined()
        )
    })?;

    println!();
    println!("{}", style("🧪 AQEA Compression Quality Test").bold().cyan());
    println!("{}", style("═".repeat(60)).dim());
    println!();

    // Use custom data or built-in test data
    let (embeddings, original_sims, dataset_name) = if let Some(custom_path) = custom {
        eprintln!("Loading custom embeddings from {}...", custom_path.display());
        let data: Vec<Vec<f32>> = serde_json::from_str(&std::fs::read_to_string(&custom_path)?)?;
        
        // Compute pairwise similarities for first 100 pairs
        let mut sims = Vec::new();
        for i in 0..data.len().min(100) {
            for j in (i+1)..data.len().min(100) {
                sims.push(cosine_similarity(&data[i], &data[j]));
            }
        }
        (data, sims, "Custom Dataset".to_string())
    } else {
        // Use built-in test data
        let (emb, sims) = test_data::get_test_data();
        (emb, sims, "Built-in STS-B Sample (100 pairs)".to_string())
    };

    let input_dim = embeddings[0].len();
    
    // Find appropriate model
    let model_name = model.unwrap_or_else(|| {
        match input_dim {
            384 => "text-minilm",
            768 => "text-mpnet",
            1024 => "text-e5large",
            1536 => "openai-small",
            3072 => "openai-large",
            _ => "text-mpnet",
        }.to_string()
    });
    
    println!("  Dataset:     {}", style(&dataset_name).white());
    println!("  Embeddings:  {}", style(format!("{} vectors", embeddings.len())).white());
    println!("  Dimension:   {}D", input_dim);
    println!("  Model:       {}", style(&model_name).green());
    println!();

    // Compress via API
    eprintln!("{}  Compressing via AQEA API...", style("📡").cyan());
    
    let api_url = config.api_url();
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    
    // Progress bar
    let pb = ProgressBar::new(3);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} {msg}")
        .unwrap());

    // Step 1: Send embeddings to API for compression
    pb.set_message("Compressing via API...");
    
    let request_body = serde_json::json!({
        "vectors": &embeddings,
        "model": &model_name
    });
    
    let resp = client
        .post(format!("{}/api/v1/compress/batch", api_url))
        .header("X-API-Key", &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .map_err(|e| {
            if e.is_connect() {
                anyhow::anyhow!(
                    "{}\n\n  Please check your internet connection.",
                    style("Cannot connect to AQEA API").red().bold()
                )
            } else {
                anyhow::anyhow!("API request failed: {}", e)
            }
        })?;
    
    if !resp.status().is_success() {
        let status = resp.status();
        let error_body: serde_json::Value = resp.json().unwrap_or_default();
        let error_msg = error_body.get("error")
            .and_then(|e| e.as_str())
            .unwrap_or("Unknown error");
        
        anyhow::bail!("API error ({}): {}", status, error_msg);
    }
    
    let result: serde_json::Value = resp.json()?;
    pb.inc(1);
    
    // Extract compressed vectors
    pb.set_message("Processing results...");
    let compressed: Vec<Vec<f32>> = result.get("compressed")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_array())
                .map(|vec| {
                    vec.iter()
                        .filter_map(|f| f.as_f64().map(|n| n as f32))
                        .collect()
                })
                .collect()
        })
        .unwrap_or_default();
    
    if compressed.is_empty() {
        anyhow::bail!("No compressed vectors received from API");
    }
    
    let output_dim = compressed[0].len();
    let ratio = input_dim as f32 / output_dim as f32;
    pb.inc(1);

    // Step 2: Compute compressed similarities (locally - this is safe)
    pb.set_message("Computing similarities...");
    let mut compressed_sims = Vec::new();
    let n = embeddings.len().min(100);
    for i in 0..n {
        for j in (i+1)..n {
            compressed_sims.push(cosine_similarity(&compressed[i], &compressed[j]));
        }
    }
    pb.inc(1);
    pb.finish_and_clear();

    // Compute Spearman correlation (locally - this is safe)
    let spearman = spearman_correlation(&original_sims, &compressed_sims);
    
    // Results
    println!();
    println!("  Compression: {}D → {}D ({:.1}x)", input_dim, output_dim, ratio);
    println!();
    println!("{}", style("═".repeat(60)).dim());
    println!("{}", style("                    RESULTS").bold());
    println!("{}", style("═".repeat(60)).dim());
    println!();
    println!("  Model:              {}", style(&model_name).green());
    println!("  Compression:        {}D → {}D ({:.1}x)", input_dim, output_dim, ratio);
    println!("{}", style("─".repeat(60)).dim());
    
    let spearman_pct = spearman * 100.0;
    let spearman_str = format!("{:.2}%", spearman_pct);
    
    if spearman >= 0.95 {
        println!("  Spearman ρ:         {} {}", 
            style(&spearman_str).green().bold(),
            style("✓").green());
    } else if spearman >= 0.90 {
        println!("  Spearman ρ:         {} {}", 
            style(&spearman_str).yellow().bold(),
            style("○").yellow());
    } else {
        println!("  Spearman ρ:         {} {}", 
            style(&spearman_str).red().bold(),
            style("✗").red());
    }
    
    println!("{}", style("─".repeat(60)).dim());
    
    // Verdict
    println!();
    if spearman >= 0.95 {
        println!("  {} {}", 
            style("VERDICT:").bold(),
            style("EXCELLENT - Similarity rankings highly preserved!").green().bold());
    } else if spearman >= 0.90 {
        println!("  {} {}", 
            style("VERDICT:").bold(),
            style("GOOD - Similarity rankings well preserved").yellow().bold());
    } else {
        println!("  {} {}", 
            style("VERDICT:").bold(),
            style("POOR - Consider using a different model").red().bold());
    }
    
    println!();
    println!("{}", style("═".repeat(60)).dim());
    
    if verbose {
        println!();
        println!("  Measured Spearman:  {:.1}%", spearman * 100.0);
        println!("  Pairs tested:       {}", original_sims.len());
        println!("  API endpoint:       {}", api_url);
    }
    
    // Summary message
    println!();
    if spearman >= 0.95 {
        println!("  {} After {:.1}x compression, {:.1}% of similarity rankings are preserved.",
            style("→").cyan(),
            ratio,
            spearman_pct);
        println!("    Your nearest-neighbor searches will return nearly identical results!");
    }
    println!();

    Ok(())
}

// ============================================================================
// MODELS COMMAND
// ============================================================================

fn models_cmd(specific_model: Option<String>) -> Result<()> {
    let models = vec![
        ("text-minilm", 384, 13, 0.971, "all-MiniLM-L6-v2", "Text embeddings (fast)"),
        ("text-mpnet", 768, 26, 0.983, "all-mpnet-base-v2", "Text embeddings (balanced)"),
        ("text-e5large", 1024, 35, 0.982, "e5-large-v2", "Text embeddings (accurate)"),
        ("mistral-embed", 1024, 35, 0.962, "mistral-embed", "Mistral AI embeddings"),
        ("openai-small", 1536, 52, 0.950, "text-embedding-3-small", "OpenAI embeddings"),
        ("openai-large", 3072, 105, 0.950, "text-embedding-3-large", "OpenAI embeddings"),
        ("audio-wav2vec2", 768, 26, 0.971, "wav2vec2-base", "Audio embeddings"),
    ];

    if let Some(name) = specific_model {
        // Show detailed info for specific model
        for (model_name, input_dim, output_dim, spearman, source, desc) in &models {
            if *model_name == name {
                println!();
                println!("{}", style(format!("📦 Model: {}", model_name)).bold().cyan());
                println!("{}", style("─".repeat(50)).dim());
                println!();
                println!("  Description:    {}", desc);
                println!("  Source Model:   {}", source);
                println!("  Input Dim:      {}D", input_dim);
                println!("  Output Dim:     {}D", output_dim);
                println!("  Compression:    {:.1}x", *input_dim as f32 / *output_dim as f32);
                println!("  Spearman ρ:     {:.1}%", spearman * 100.0);
                println!();
                println!("  Weights file:   aqea_{}_{}d.aqwt", 
                    model_name.replace("-", "_"), input_dim);
                println!();
                println!("  Usage:");
                println!("    aqea compress -m {} input.json", model_name);
                println!();
                return Ok(());
            }
        }
        anyhow::bail!("Unknown model: {}", name);
    }

    // List all models
    println!();
    println!("{}", style("📋 Available Models").bold().cyan());
    println!();
    println!("  {:<18} {:<14} {:<10} {}", 
        style("Model").underlined(),
        style("Dimensions").underlined(),
        style("Spearman").underlined(),
        style("Description").underlined());
    println!("  {}", style("─".repeat(70)).dim());
    
    for (model_name, input_dim, output_dim, spearman, _source, desc) in &models {
        let dims = format!("{}D → {}D", input_dim, output_dim);
        let spearman_str = format!("{:.1}%", spearman * 100.0);
        println!("  {:<18} {:<14} {:<10} {}", 
            style(model_name).green(),
            dims,
            spearman_str,
            style(desc).dim());
    }
    
    println!();
    println!("  Run {} for details", style("aqea models <name>").cyan());
    println!("  Run {} to verify quality", style("aqea test -m <name>").cyan());
    println!();

    Ok(())
}

// ============================================================================
// INFO COMMAND
// ============================================================================

fn info_cmd(weights_path: PathBuf) -> Result<()> {
    let weights = BinaryWeights::load(&weights_path)?;
    
    println!();
    println!("{}", style(format!("📦 Weights: {}", weights_path.display())).bold().cyan());
    println!("{}", style("─".repeat(50)).dim());
    println!();
    println!("  Input Dimension:    {}D", weights.original_dim);
    println!("  Output Dimension:   {}D", weights.compressed_dim);
    println!("  Compression Ratio:  {:.1}x", weights.original_dim as f32 / weights.compressed_dim as f32);
    println!("  Spearman ρ:         {:.1}%", weights.spearman * 100.0);
    println!("  Rotation Scale:     {:.2}", weights.rotation_scale);
    println!("  Model Type:         {:?}", weights.model_type);
    println!("  Weights Count:      {}", weights.weights.len());
    println!("  File Size:          {} KB", std::fs::metadata(&weights_path)?.len() / 1024);
    println!();
    
    Ok(())
}

// ============================================================================
// CONFIG COMMANDS
// ============================================================================

fn config_set(key: &str, value: &str) -> Result<()> {
    let mut config = Config::load()?;
    
    match key {
        "default-model" => config.default_model = Some(value.to_string()),
        "api-url" => config.api_url = Some(value.to_string()),
        _ => anyhow::bail!("Unknown config key: {}. Valid keys: default-model, api-url", key),
    }
    
    config.save()?;
    println!("{} Set {} = {}", style("✓").green(), key, value);
    
    Ok(())
}

fn config_get(key: &str) -> Result<()> {
    let config = Config::load()?;
    
    let value = match key {
        "default-model" => config.default_model.as_deref().unwrap_or("(not set)"),
        "api-url" => config.api_url.as_deref().unwrap_or("https://api.aqea.ai"),
        "api-key" => {
            if config.api_key.is_some() {
                "(set - use 'aqe auth token' to view)"
            } else {
                "(not set)"
            }
        }
        _ => anyhow::bail!("Unknown config key: {}", key),
    };
    
    println!("{}", value);
    
    Ok(())
}

fn config_list() -> Result<()> {
    let config = Config::load()?;
    
    println!();
    println!("{}", style("⚙️  Configuration").bold().cyan());
    println!("{}", style("─".repeat(40)).dim());
    println!();
    println!("  default-model:  {}", 
        config.default_model.as_deref().unwrap_or("(not set)"));
    println!("  api-url:        {}", 
        config.api_url.as_deref().unwrap_or("https://api.aqea.ai"));
    println!("  api-key:        {}", 
        if config.api_key.is_some() { "(set)" } else { "(not set)" });
    println!();
    println!("  Config file: {}", Config::path()?.display());
    println!();
    
    Ok(())
}

fn config_path() -> Result<()> {
    println!("{}", Config::path()?.display());
    Ok(())
}

// ============================================================================
// USAGE COMMAND
// ============================================================================

fn usage_cmd(_month: Option<String>) -> Result<()> {
    let config = Config::load()?;
    
    if config.api_key.is_none() {
        println!("{} Not authenticated. Run: aqea auth login", style("Error:").red());
        std::process::exit(1);
    }
    
    println!();
    println!("{}", style("📊 Usage Statistics").bold().cyan());
    println!("{}", style("─".repeat(40)).dim());
    println!();
    println!("  {}  Usage tracking requires API connection.", style("ℹ").blue());
    println!("     Visit: {}", style("https://aqea.ai/dashboard/usage").cyan().underlined());
    println!();
    
    // TODO: Implement API call to fetch usage
    
    Ok(())
}
