# AQEA CLI Usage Guide

## Table of Contents

- [Interactive Mode](#interactive-mode)
- [Commands](#commands)
- [File Formats](#file-formats)
- [Configuration](#configuration)
- [Examples](#examples)

## Interactive Mode

Start the CLI without arguments to enter interactive mode:

```bash
aqea
```

You'll see:
```
🔷 AQEA CLI v0.1.0
Type /help for commands, or enter vectors to compress

aqea>
```

## Commands

### Authentication

```bash
/login              # Enter API key interactively
/logout             # Clear stored credentials
/status             # Show auth status and quota
```

### Models

```bash
/models             # List all available models
/model <name>       # Select a model (e.g., /model text-mpnet)
```

### Compression

```bash
/compress <file>    # Compress vectors from file
/decompress <file>  # Decompress vectors
/validate <file>    # Test compression quality
```

### Session

```bash
/help               # Show all commands
/clear              # Clear screen
/quit               # Exit CLI
```

## File Formats

### Input: JSON Array

```json
[
  [0.123, -0.456, 0.789, ...],
  [0.234, -0.567, 0.890, ...],
  ...
]
```

### Input: JSON with IDs

```json
{
  "vectors": [
    {"id": "doc1", "embedding": [0.123, -0.456, ...]},
    {"id": "doc2", "embedding": [0.234, -0.567, ...]}
  ]
}
```

### Input: CSV

```csv
0.123,-0.456,0.789,...
0.234,-0.567,0.890,...
```

### Output

Same format as input, with compressed dimensions.

## Configuration

Config location:
- Linux/macOS: `~/.config/aqea/config.toml`
- Windows: `%APPDATA%\aqea\config.toml`

```toml
# config.toml
default_model = "text-mpnet"
output_format = "json"
verbose = false
```

## Examples

### Compress MiniLM Embeddings

```bash
aqea> /login
Enter API key: aqea_xxxxx
✅ Authenticated

aqea> /model text-minilm
✅ Selected: text-minilm (384D → 13D)

aqea> /compress embeddings.json
📥 Loaded 1000 vectors (384D)
⚡ Compressing...
✅ Compressed to 13D (30x compression)
💾 Saved to embeddings_compressed.json
```

### Validate Quality

```bash
aqea> /validate my_data.json
🔬 Testing compression quality...
   Original pairs: 500
   Spearman ρ: 0.947 (94.7%)
   ✅ EXCELLENT - Safe to use!
```

### Non-Interactive Usage

```bash
# Direct compression
aqea compress input.json -o output.json

# With model selection
aqea compress input.json --model text-mpnet -o output.json

# Validate only
aqea validate input.json
```

## Troubleshooting

### "API key invalid"

1. Check key format: should start with `aqea_`
2. Generate new key at https://aqea.ai/dashboard

### "Dimension mismatch"

Make sure your embedding dimensions match the model:
- 384D → text-minilm
- 768D → text-mpnet, audio-wav2vec2
- 1024D → text-e5large, mistral-embed
- 1536D → openai-small
- 3072D → openai-large

### "Rate limit exceeded"

- Free tier: 10 req/min
- Upgrade at https://aqea.ai/pricing
