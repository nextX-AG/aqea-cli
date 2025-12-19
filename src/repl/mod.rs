//! Interactive REPL for AQEA CLI
//!
//! Provides a Claude CLI-style interactive terminal interface.

mod commands;
mod handler;

pub use commands::{Command, parse_command};
pub use handler::CommandHandler;

use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use console::style;
use std::path::PathBuf;

/// REPL configuration
pub struct ReplConfig {
    pub history_file: PathBuf,
    pub api_url: String,
}

impl Default for ReplConfig {
    fn default() -> Self {
        let history_file = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("aqea")
            .join("history");
        
        Self {
            history_file,
            api_url: "https://api.aqea.ai".to_string(),
        }
    }
}

/// Print the welcome banner
fn print_banner() {
    println!();
    println!("{}", style("╔═══════════════════════════════════════════════════════════════╗").cyan());
    println!("{}", style("║           AQEA Compress™ - Interactive CLI                    ║").cyan());
    println!("{}", style("╠═══════════════════════════════════════════════════════════════╣").cyan());
    println!("{}", style("║  Type /help for commands, /login to authenticate              ║").cyan());
    println!("{}", style("╚═══════════════════════════════════════════════════════════════╝").cyan());
    println!();
}

/// Run the interactive REPL
pub fn run_repl(config: ReplConfig) -> anyhow::Result<()> {
    print_banner();
    
    let mut rl = DefaultEditor::new()?;
    
    // Load history
    if config.history_file.exists() {
        let _ = rl.load_history(&config.history_file);
    }
    
    // Ensure history directory exists
    if let Some(parent) = config.history_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let mut handler = CommandHandler::new(config.api_url.clone());
    
    loop {
        let prompt = handler.get_prompt();
        
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                
                if line.is_empty() {
                    continue;
                }
                
                // Add to history
                let _ = rl.add_history_entry(line);
                
                // Parse and execute command
                match parse_command(line) {
                    Ok(cmd) => {
                        match handler.execute(cmd) {
                            Ok(should_exit) => {
                                if should_exit {
                                    println!("{}", style("Goodbye! 👋").green());
                                    break;
                                }
                            }
                            Err(e) => {
                                println!("{} {}", style("Error:").red().bold(), e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("{} {}", style("Error:").red().bold(), e);
                        println!("Type {} for available commands", style("/help").cyan());
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("{}", style("Use /exit to quit").yellow());
            }
            Err(ReadlineError::Eof) => {
                println!("{}", style("Goodbye! 👋").green());
                break;
            }
            Err(err) => {
                println!("{} {:?}", style("Error:").red().bold(), err);
                break;
            }
        }
    }
    
    // Save history
    let _ = rl.save_history(&config.history_file);
    
    Ok(())
}

