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

    /// PQ codebook operations (train, download, list)
    Pq {
        #[command(subcommand)]
        action: PqCommands,
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

    /// Train AQEA weights on your embeddings
    ///
    /// Unified training system with intelligent sampling strategies
    /// and automatic hyperparameter tuning.
    Train {
        /// Input embeddings file (AQED or JSON)
        #[arg(short, long)]
        input: PathBuf,

        /// Output weights file (.aqwt)
        #[arg(short, long)]
        output: PathBuf,

        /// Training split percentage (default: 20%)
        #[arg(long, default_value = "20")]
        train_split: f32,

        /// Fixed sample count for single-run mode (e.g., "500" or "50%")
        /// Disables progressive training - faster for testing
        #[arg(long)]
        samples: Option<String>,

        /// Sampling strategy: random, kmeans, tsne-grid (default: kmeans)
        #[arg(long, default_value = "kmeans")]
        sampling: String,

        /// Target compressed dimension (default: auto-detect)
        #[arg(long)]
        dim: Option<usize>,

        /// Train PQ codebook with N subvectors (e.g., 17 for 241x compression)
        #[arg(long)]
        pq: Option<usize>,

        /// PQ codebook output file
        #[arg(long)]
        pq_output: Option<PathBuf>,

        /// Training report output (JSON)
        #[arg(long)]
        report: Option<PathBuf>,

        /// Quick training mode (fewer iterations)
        #[arg(long)]
        quick: bool,

        /// Instant mode for testing (~5 seconds, minimal quality)
        #[arg(long)]
        instant: bool,

        /// EXPERIMENTAL: Sample from FULL embedding space (all data, not just labeled)
        /// Uses cosine similarity of originals as ground truth
        #[arg(long)]
        full_space: bool,

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

#[derive(Subcommand)]
enum PqCommands {
    /// Train a custom PQ codebook on your data
    Train {
        /// Input file with AQEA-compressed vectors (CSV/JSON)
        input: PathBuf,

        /// Number of subvectors (default: 8)
        #[arg(short, long, default_value = "8")]
        subvectors: usize,

        /// Model name for the codebook
        #[arg(short, long)]
        model: Option<String>,

        /// Custom name for your codebook
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Check status of a PQ training job
    Status {
        /// Job ID from pq train command
        job_id: String,
    },

    /// Download a trained PQ codebook
    Download {
        /// Job ID from pq train command
        job_id: String,

        /// Output file for codebook (JSON)
        #[arg(short, long)]
        output: PathBuf,
    },

    /// List available PQ codebooks
    List,
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
        Commands::Pq { action } => match action {
            PqCommands::Train { input, subvectors, model, name } => {
                pq_train_cmd(input, subvectors, model, name)?;
            },
            PqCommands::Status { job_id } => {
                pq_status_cmd(&job_id)?;
            },
            PqCommands::Download { job_id, output } => {
                pq_download_cmd(&job_id, output)?;
            },
            PqCommands::List => {
                pq_list_cmd()?;
            },
        },
        Commands::Validate { data, weights, pq_subs, json, verbose } => {
            validate_cmd(&data, &weights, pq_subs, json, verbose)?;
        },
        Commands::Train { input, output, train_split, samples, sampling, dim, pq, pq_output, report, quick, instant, full_space, verbose } => {
            train_cmd(&input, &output, train_split, samples.as_deref(), &sampling, dim, pq, pq_output, report, quick, instant, full_space, verbose)?;
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
    println!("1. Go to: {}", style("https://compress.aqea.ai/platform/workspace").cyan().underlined());
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
            style("https://compress.aqea.ai/platform/workspace").cyan().underlined()
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
                    style("https://compress.aqea.ai/pricing").cyan().underlined()
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
            style("https://compress.aqea.ai/platform/workspace").cyan().underlined()
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
    println!("     Visit: {}", style("https://compress.aqea.ai/platform/workspace").cyan().underlined());
    println!();
    
    // TODO: Implement API call to fetch usage
    
    Ok(())
}

// ============================================================================
// PQ COMMANDS
// ============================================================================

/// Train a custom PQ codebook
fn pq_train_cmd(
    input: PathBuf,
    subvectors: usize,
    model: Option<String>,
    name: Option<String>,
) -> Result<()> {
    let config = Config::load()?;
    let api_key = config.api_key.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "{}\n\n  Run: {}\n  Get your API key at: {}",
            style("Not authenticated").red().bold(),
            style("aqea auth login").cyan(),
            style("https://compress.aqea.ai/platform/workspace").cyan().underlined()
        )
    })?;
    let api_url = config.api_url();

    println!();
    println!("{}", style("🎯 PQ Codebook Training").bold().cyan());
    println!("{}", style("─".repeat(50)).dim());
    println!();

    // Read input vectors
    println!("  {} Reading input file...", style("→").dim());
    let content = std::fs::read_to_string(&input)
        .map_err(|e| anyhow::anyhow!("Failed to read input file: {}", e))?;

    // Parse vectors (assuming JSON array of arrays)
    let vectors: Vec<Vec<f32>> = if input.extension().map_or(false, |e| e == "csv") {
        // CSV: parse as rows of floats
        let mut rdr = csv::Reader::from_reader(content.as_bytes());
        rdr.records()
            .filter_map(|r| r.ok())
            .map(|record| {
                record.iter()
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect()
            })
            .filter(|v: &Vec<f32>| !v.is_empty())
            .collect()
    } else {
        // JSON
        serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON: {}", e))?
    };

    if vectors.is_empty() {
        anyhow::bail!("No vectors found in input file");
    }

    println!("  {} {} vectors loaded ({} dimensions)", 
        style("✓").green(), vectors.len(), vectors[0].len());

    // Build request
    let request_body = serde_json::json!({
        "vectors": vectors,
        "options": {
            "model": model,
            "subvectors": subvectors,
            "name": name,
        }
    });

    println!("  {} Submitting training job...", style("→").dim());

    // Submit job
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{}/api/v1/pq/train", api_url))
        .header("X-API-Key", &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()?;

    if !resp.status().is_success() {
        let error: serde_json::Value = resp.json().unwrap_or_default();
        anyhow::bail!("Training failed: {}", 
            error.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown error"));
    }

    let result: serde_json::Value = resp.json()?;
    let job_id = result.get("job_id").and_then(|j| j.as_str()).unwrap_or("unknown");

    println!();
    println!("  {} Training job submitted!", style("✓").green().bold());
    println!();
    println!("  Job ID:  {}", style(job_id).cyan().bold());
    println!();
    println!("  {} Check status:", style("Next steps:").bold());
    println!("     aqea pq status {}", job_id);
    println!();
    println!("  {} Download when ready:", style("Then:").bold());
    println!("     aqea pq download {} -o my-codebook.json", job_id);
    println!();

    Ok(())
}

/// Check PQ training status
fn pq_status_cmd(job_id: &str) -> Result<()> {
    let config = Config::load()?;
    let api_key = config.api_key.clone().ok_or_else(|| {
        anyhow::anyhow!("Not authenticated. Run: aqea auth login")
    })?;
    let api_url = config.api_url();

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(format!("{}/api/v1/pq/train/{}", api_url, job_id))
        .header("X-API-Key", &api_key)
        .send()?;

    if !resp.status().is_success() {
        let error: serde_json::Value = resp.json().unwrap_or_default();
        anyhow::bail!("{}", 
            error.get("message").and_then(|m| m.as_str()).unwrap_or("Job not found"));
    }

    let result: serde_json::Value = resp.json()?;

    println!();
    println!("{}", style("📊 PQ Training Status").bold().cyan());
    println!("{}", style("─".repeat(40)).dim());
    println!();
    println!("  Job ID:    {}", style(job_id).cyan());
    println!("  Status:    {}", 
        style(result.get("status").and_then(|s| s.as_str()).unwrap_or("unknown")).yellow());
    println!("  Progress:  {}%", 
        result.get("progress_percent").and_then(|p| p.as_f64()).unwrap_or(0.0) as i32);
    println!("  Step:      {}", 
        result.get("current_step").and_then(|s| s.as_str()).unwrap_or(""));
    
    if let Some(r) = result.get("result") {
        println!();
        println!("  {}", style("Result:").bold().green());
        println!("    Compression: {}x", 
            r.get("compression_ratio").and_then(|c| c.as_f64()).unwrap_or(0.0) as i32);
        println!("    Quality:     {:.1}%", 
            r.get("expected_quality").and_then(|q| q.as_f64()).unwrap_or(0.0) * 100.0);
        println!("    Trained on:  {} vectors", 
            r.get("vectors_trained").and_then(|v| v.as_i64()).unwrap_or(0));
        println!();
        println!("  {} Download with:", style("Ready!").green().bold());
        println!("     aqea pq download {} -o codebook.json", job_id);
    }
    println!();

    Ok(())
}

/// Download a trained PQ codebook
fn pq_download_cmd(job_id: &str, output: PathBuf) -> Result<()> {
    let config = Config::load()?;
    let api_key = config.api_key.clone().ok_or_else(|| {
        anyhow::anyhow!("Not authenticated. Run: aqea auth login")
    })?;
    let api_url = config.api_url();

    println!();
    println!("{}", style("📥 Downloading PQ Codebook").bold().cyan());
    println!();

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(format!("{}/api/v1/pq/train/{}/codebook", api_url, job_id))
        .header("X-API-Key", &api_key)
        .send()?;

    if !resp.status().is_success() {
        let status = resp.status();
        let error: serde_json::Value = resp.json().unwrap_or_default();
        if status == reqwest::StatusCode::ACCEPTED {
            anyhow::bail!("Training not completed yet. Check status with: aqea pq status {}", job_id);
        }
        anyhow::bail!("{}", 
            error.get("message").and_then(|m| m.as_str()).unwrap_or("Download failed"));
    }

    let codebook: serde_json::Value = resp.json()?;
    let json = serde_json::to_string_pretty(&codebook)?;
    
    std::fs::write(&output, &json)?;

    let codebook_id = codebook.get("codebook_id").and_then(|c| c.as_str()).unwrap_or("unknown");

    println!("  {} Codebook saved to: {}", style("✓").green(), output.display());
    println!("  {} Codebook ID: {}", style("ℹ").blue(), codebook_id);
    println!();
    println!("  {} Use with compress-pq:", style("Usage:").bold());
    println!("     aqea compress-pq input.csv --codebook {} -o output.csv", output.display());
    println!();

    Ok(())
}

/// List available PQ codebooks
fn pq_list_cmd() -> Result<()> {
    let config = Config::load()?;
    let api_url = config.api_url();

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(format!("{}/api/v1/pq/codebooks", api_url))
        .send()?;

    if !resp.status().is_success() {
        anyhow::bail!("Failed to fetch codebooks");
    }

    let result: serde_json::Value = resp.json()?;
    let codebooks = result.get("codebooks").and_then(|c| c.as_array());

    println!();
    println!("{}", style("📚 Available PQ Codebooks").bold().cyan());
    println!("{}", style("─".repeat(70)).dim());
    println!();
    println!("  {:<30} {:>10} {:>12} {:>10}", 
        style("ID").bold(), style("Ratio").bold(), style("Quality").bold(), style("Dims").bold());
    println!("  {}", style("─".repeat(66)).dim());

    if let Some(cbs) = codebooks {
        for cb in cbs {
            let id = cb.get("id").and_then(|i| i.as_str()).unwrap_or("");
            let ratio = cb.get("compression_ratio").and_then(|r| r.as_f64()).unwrap_or(0.0);
            let quality = cb.get("expected_quality").and_then(|q| q.as_f64()).unwrap_or(0.0);
            let dim = cb.get("original_dim").and_then(|d| d.as_i64()).unwrap_or(0);

            println!("  {:<30} {:>9.0}x {:>11.1}% {:>10}", 
                id, ratio, quality * 100.0, dim);
        }
    }

    println!();
    println!("  {} Train your own: aqea pq train <input.csv>", style("Tip:").blue());
    println!();

    Ok(())
}

// ============================================================================
// TRAIN COMMAND - Unified Training System
// ============================================================================

fn train_cmd(
    input: &PathBuf,
    output: &PathBuf,
    train_split: f32,
    samples: Option<&str>,
    sampling: &str,
    dim: Option<usize>,
    pq_subs: Option<usize>,
    pq_output: Option<PathBuf>,
    report_path: Option<PathBuf>,
    quick: bool,
    instant: bool,
    full_space: bool,
    verbose: bool,
) -> Result<()> {
    use aqea_core::training::{
        AutoTrainer, AutoTrainerConfig, SamplingStrategy, SampleProgression,
        loader::{load_auto, EmbeddingData},
    };
    use aqea_core::pq::ProductQuantizer;
    use std::time::Instant;

    println!();
    println!("{}", style("🚀 AQEA Training System").bold().cyan());
    println!("{}", style("═".repeat(60)).dim());
    println!();

    // Full-space mode warning
    if full_space {
        println!("  {} FULL-SPACE MODE (experimental)", style("🌍").yellow().bold());
        println!("  {} Sampling from ALL embeddings, not just labeled subset", style("→").dim());
        println!("  {} Using original cosine similarity as ground truth", style("→").dim());
        println!();
    }

    // Parse sampling strategy
    let sampling_strategy: SamplingStrategy = sampling.parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;

    println!("  {} Loading data from: {}", style("📂").dim(), input.display());

    let start = Instant::now();
    let data = load_auto(input)?;
    let load_time = start.elapsed();

    println!("  {} Loaded {} embeddings ({}D) in {:.1}s",
        style("✓").green(),
        data.len(),
        data.dimension(),
        load_time.as_secs_f32()
    );

    // Validate data
    if data.len() < 100 {
        anyhow::bail!("Need at least 100 embeddings for training, got {}", data.len());
    }

    // Split into training pairs
    // In full_space mode: sample from ALL data using t-SNE grid
    // In normal mode: use train_split to separate train/validation
    let train_fraction = if full_space {
        // In full-space mode, we sample uniformly from entire space
        // Use larger fraction since we're covering more ground
        (train_split / 100.0).max(0.3)
    } else {
        train_split / 100.0
    };
    
    let (train_data, val_data) = data.split(train_fraction, 42);

    println!("  {} Training: {} samples, Validation: {} samples",
        style("📊").dim(),
        train_data.len(),
        val_data.len()
    );
    println!("  {} Sampling strategy: {}{}",
        style("🎯").dim(),
        sampling_strategy,
        if full_space { " (full-space)" } else { "" }
    );

    // Determine compressed dimension
    let original_dim = data.dimension();
    let compressed_dim = dim.unwrap_or_else(|| {
        // Default: approximately sqrt(original_dim) * 3, aligned to 8
        let target = ((original_dim as f32).sqrt() * 3.0) as usize;
        ((target + 7) / 8) * 8  // Round up to nearest 8
    });

    println!("  {} Compression: {}D → {}D ({:.1}x)",
        style("📦").dim(),
        original_dim,
        compressed_dim,
        original_dim as f32 / compressed_dim as f32
    );
    println!();

    // Configure training
    let mut config = if instant {
        println!("  {} INSTANT mode: 10 generations × 15 population (~5s)",
            style("⚡").yellow().bold());
        AutoTrainerConfig::instant()
    } else if quick {
        AutoTrainerConfig::quick()
    } else {
        AutoTrainerConfig::default()
    }
    .with_compressed_dim(compressed_dim)
    .with_sampling_strategy(sampling_strategy);

    // Parse --samples argument for single-run mode
    if let Some(samples_str) = samples {
        if samples_str.ends_with('%') {
            // Percentage mode: "50%" -> 50.0
            let pct: f32 = samples_str[..samples_str.len()-1]
                .parse()
                .context("Invalid percentage format for --samples")?;
            config = config.with_percent_samples(pct);
            println!("  {} Single-run mode: {:.0}% of training data",
                style("⚡").yellow().bold(), pct);
        } else {
            // Absolute number: "500" -> 500
            let n: usize = samples_str.parse()
                .context("Invalid number format for --samples (use e.g. '500' or '50%')")?;
            config = config.with_single_samples(n);
            println!("  {} Single-run mode: {} samples",
                style("⚡").yellow().bold(), n);
        }
    }

    let config = if verbose {
        config
    } else {
        config.quiet()
    };

    // Train
    println!("{}", style("Training AQEA weights...").bold());
    println!();

    let train_start = Instant::now();
    let trainer = AutoTrainer::new(config);

    // Use embeddings for both e1 and e2 (self-similarity training)
    let result = trainer.train(
        &train_data.embeddings,
        &train_data.embeddings,
    ).map_err(|e| anyhow::anyhow!(e))?;

    let train_time = train_start.elapsed();

    println!();
    println!("{}", style("═".repeat(60)).dim());
    println!();
    println!("  {} Training complete in {:.1}s",
        style("✓").green().bold(),
        train_time.as_secs_f32()
    );
    println!("  {} Final validation score: {:.1}%",
        style("📈").dim(),
        result.final_val_score * 100.0
    );
    println!("  {} Optimal sample size: {}",
        style("🎯").dim(),
        result.optimal_sample_size
    );

    // Save weights
    let compressor = result.to_compressor();
    let weights_data = compressor.get_flat_weights();

    // Create simple weights file (JSON for now)
    let weights_json = serde_json::json!({
        "version": 1,
        "original_dim": original_dim,
        "compressed_dim": compressed_dim,
        "weights": weights_data,
        "train_score": result.final_train_score,
        "val_score": result.final_val_score,
        "optimal_samples": result.optimal_sample_size,
    });

    std::fs::write(output, serde_json::to_string_pretty(&weights_json)?)?;
    println!("  {} Weights saved to: {}", style("💾").dim(), output.display());

    // Train PQ if requested
    if let Some(n_subvectors) = pq_subs {
        println!();
        println!("{}", style("Training PQ codebook...").bold());

        // Compress training data with AQEA first
        let compressed: Vec<Vec<f32>> = train_data.embeddings.iter()
            .map(|e| compressor.compress(e))
            .collect();

        let mut pq = ProductQuantizer::new(compressed_dim, n_subvectors, 8);
        pq.train(&compressed, 100);

        let pq_path = pq_output.unwrap_or_else(|| {
            let mut p = output.clone();
            p.set_extension("pq.json");
            p
        });

        pq.save(&pq_path)?;

        println!("  {} PQ codebook saved to: {}", style("💾").dim(), pq_path.display());
        println!("  {} Subvectors: {}, Total compression: {:.0}x",
            style("📦").dim(),
            n_subvectors,
            (original_dim as f32 * 4.0) / (n_subvectors as f32)
        );
    }

    // Save report if requested
    if let Some(report_file) = report_path {
        let report_json = result.report.to_json()?;
        std::fs::write(&report_file, report_json)?;
        println!("  {} Report saved to: {}", style("📋").dim(), report_file.display());
    }

    println!();
    println!("{}", style("🎉 Training complete!").bold().green());
    println!();

    Ok(())
}
