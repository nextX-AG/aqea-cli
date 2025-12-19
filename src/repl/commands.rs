//! Command parsing for REPL

use std::path::PathBuf;

/// All available REPL commands
#[derive(Debug, Clone)]
pub enum Command {
    // Core commands
    Help,
    Exit,
    Clear,
    
    // Auth commands
    Login { api_key: Option<String> },
    Logout,
    Status,
    
    // Model commands
    Models,
    Model { name: Option<String> },
    
    // Mode commands (2-Stage vs 3-Stage)
    Mode { mode: Option<String> },
    Subvectors { count: Option<usize> },
    
    // Compression commands
    Compress { file: PathBuf, output: Option<PathBuf> },
    CompressPq { file: PathBuf, output: Option<PathBuf> },
    Decompress { file: PathBuf, output: Option<PathBuf> },
    Batch { folder: PathBuf },
    Ratio { value: Option<u32> },
    
    // Benchmark & Training commands (NEW!)
    Benchmark { 
        file: PathBuf, 
        output: Option<PathBuf>,
        train: bool,  // --train flag for fresh training
        compressions: Option<Vec<u32>>,  // e.g., 10,20,30
        pq_subs: Option<Vec<usize>>,     // e.g., 4,8,13,16
    },
    Train { 
        file: PathBuf, 
        compression: Option<u32>,
        name: Option<String>,
    },
    
    // Config commands
    Config,
    History,
    
    // Direct input (non-slash command)
    Unknown { input: String },
}

/// Parse a command string into a Command
pub fn parse_command(input: &str) -> anyhow::Result<Command> {
    let input = input.trim();
    
    // Check if it's a slash command
    if !input.starts_with('/') {
        return Ok(Command::Unknown { input: input.to_string() });
    }
    
    let parts: Vec<&str> = input[1..].split_whitespace().collect();
    
    if parts.is_empty() {
        anyhow::bail!("Empty command");
    }
    
    let cmd = parts[0].to_lowercase();
    let args = &parts[1..];
    
    match cmd.as_str() {
        // Core commands
        "help" | "h" | "?" => Ok(Command::Help),
        "exit" | "quit" | "q" => Ok(Command::Exit),
        "clear" | "cls" => Ok(Command::Clear),
        
        // Auth commands
        "login" => {
            let api_key = args.first().map(|s| s.to_string());
            Ok(Command::Login { api_key })
        }
        "logout" => Ok(Command::Logout),
        "status" | "whoami" => Ok(Command::Status),
        
        // Model commands
        "models" | "list" => Ok(Command::Models),
        "model" | "m" => {
            let name = args.first().map(|s| s.to_string());
            Ok(Command::Model { name })
        }
        
        // Mode commands
        "mode" => {
            let mode = args.first().map(|s| s.to_string());
            Ok(Command::Mode { mode })
        }
        "subvectors" | "subs" => {
            let count = args.first().and_then(|s| s.parse().ok());
            Ok(Command::Subvectors { count })
        }
        
        // Compression commands
        "compress" | "c" => {
            if args.is_empty() {
                anyhow::bail!("Usage: /compress <file> [-o output]");
            }
            let file = PathBuf::from(args[0]);
            let output = parse_output_arg(args);
            Ok(Command::Compress { file, output })
        }
        "compress-pq" | "cpq" | "pq" => {
            if args.is_empty() {
                anyhow::bail!("Usage: /compress-pq <file> [-o output]");
            }
            let file = PathBuf::from(args[0]);
            let output = parse_output_arg(args);
            Ok(Command::CompressPq { file, output })
        }
        "decompress" | "d" => {
            if args.is_empty() {
                anyhow::bail!("Usage: /decompress <file> [-o output]");
            }
            let file = PathBuf::from(args[0]);
            let output = parse_output_arg(args);
            Ok(Command::Decompress { file, output })
        }
        "batch" | "b" => {
            if args.is_empty() {
                anyhow::bail!("Usage: /batch <folder>");
            }
            Ok(Command::Batch { folder: PathBuf::from(args[0]) })
        }
        "ratio" | "r" => {
            let value = args.first().and_then(|s| s.parse().ok());
            Ok(Command::Ratio { value })
        }
        
        // Benchmark & Training commands
        "benchmark" | "bench" => {
            if args.is_empty() {
                anyhow::bail!("Usage: /benchmark <file> [--train] [--compressions 10,20,30] [--pq-subs 4,8,13,16] [-o output]");
            }
            let file = PathBuf::from(args[0]);
            let output = parse_output_arg(args);
            let train = args.iter().any(|a| *a == "--train" || *a == "-t");
            let compressions = parse_list_arg(args, "--compressions");
            let pq_subs = parse_list_arg(args, "--pq-subs");
            Ok(Command::Benchmark { file, output, train, compressions, pq_subs })
        }
        "train" => {
            if args.is_empty() {
                anyhow::bail!("Usage: /train <file> [--compression 20] [--name model-name]");
            }
            let file = PathBuf::from(args[0]);
            let compression = parse_named_arg(args, "--compression").and_then(|s| s.parse().ok());
            let name = parse_named_arg(args, "--name");
            Ok(Command::Train { file, compression, name })
        }
        
        // Config commands
        "config" => Ok(Command::Config),
        "history" => Ok(Command::History),
        
        _ => anyhow::bail!("Unknown command: /{}. Type /help for available commands.", cmd),
    }
}

/// Parse -o/--output argument from args
fn parse_output_arg(args: &[&str]) -> Option<PathBuf> {
    for (i, arg) in args.iter().enumerate() {
        if (*arg == "-o" || *arg == "--output") && i + 1 < args.len() {
            return Some(PathBuf::from(args[i + 1]));
        }
    }
    None
}

/// Parse a named argument value (e.g., --name value)
fn parse_named_arg(args: &[&str], name: &str) -> Option<String> {
    for (i, arg) in args.iter().enumerate() {
        if *arg == name && i + 1 < args.len() {
            return Some(args[i + 1].to_string());
        }
    }
    None
}

/// Parse comma-separated list argument (e.g., --compressions 10,20,30)
fn parse_list_arg<T: std::str::FromStr>(args: &[&str], name: &str) -> Option<Vec<T>> {
    parse_named_arg(args, name).map(|s| {
        s.split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_help() {
        assert!(matches!(parse_command("/help").unwrap(), Command::Help));
        assert!(matches!(parse_command("/h").unwrap(), Command::Help));
        assert!(matches!(parse_command("/?").unwrap(), Command::Help));
    }
    
    #[test]
    fn test_parse_exit() {
        assert!(matches!(parse_command("/exit").unwrap(), Command::Exit));
        assert!(matches!(parse_command("/quit").unwrap(), Command::Exit));
        assert!(matches!(parse_command("/q").unwrap(), Command::Exit));
    }
    
    #[test]
    fn test_parse_compress() {
        match parse_command("/compress test.json").unwrap() {
            Command::Compress { file, output } => {
                assert_eq!(file, PathBuf::from("test.json"));
                assert!(output.is_none());
            }
            _ => panic!("Expected Compress command"),
        }
    }
    
    #[test]
    fn test_parse_compress_with_output() {
        match parse_command("/compress test.json -o out.json").unwrap() {
            Command::Compress { file, output } => {
                assert_eq!(file, PathBuf::from("test.json"));
                assert_eq!(output, Some(PathBuf::from("out.json")));
            }
            _ => panic!("Expected Compress command"),
        }
    }
}

