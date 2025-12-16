# AQEA CLI

> Compress embeddings up to 3000x while preserving 85-97% similarity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/nextX-AG/aqea-cli)](https://github.com/nextX-AG/aqea-cli/releases)
[![Platform Support](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-blue)](https://github.com/nextX-AG/aqea-cli/releases)

AQEA Compress is a CLI tool for extreme embedding compression. Reduce your vector storage costs by up to 99% while maintaining semantic search quality.

## Quick Install

### macOS / Linux

```bash
curl -fsSL https://aqea.ai/install.sh | bash
```

### Windows (PowerShell)

```powershell
irm https://aqea.ai/install.ps1 | iex
```

### Homebrew (macOS/Linux)

```bash
brew install nextx-ag/tap/aqea
```

### Manual Download

Download the latest release for your platform from [GitHub Releases](https://github.com/nextX-AG/aqea-cli/releases).

## Supported Platforms

| OS | x86_64 (Intel/AMD) | ARM64 (Apple Silicon/ARM) |
|----|:------------------:|:-------------------------:|
| Linux | ✅ | ✅ |
| macOS | ✅ | ✅ |
| Windows | ✅ | ✅ |

## Usage

### Interactive Mode (REPL)

```bash
# Start interactive shell
aqea

# You'll see:
# 🔷 AQEA Compress CLI v0.1.0
# Type /help for commands, or enter vectors to compress
```

### Commands

| Command | Description |
|---------|-------------|
| `/login` | Authenticate with your API key |
| `/models` | List available compression models |
| `/model <name>` | Select a compression model |
| `/compress <file>` | Compress vectors from file |
| `/decompress <file>` | Decompress vectors |
| `/validate <file>` | Test compression quality on your data |
| `/status` | Show current session status |
| `/help` | Show all available commands |
| `/quit` | Exit the CLI |

### Quick Examples

```bash
# Login with API key
aqea
> /login
Enter API key: aqea_xxxxx
✅ Authenticated successfully!

# Compress a file
> /compress embeddings.json
📥 Loaded 1000 vectors (384D)
📦 Using model: text-minilm
✅ Compressed to 13D (30x compression)
💾 Saved to embeddings_compressed.json

# Validate compression quality
> /validate my_embeddings.json
🔬 Testing compression quality...
   Spearman correlation: 94.7%
   ✅ EXCELLENT - Safe to use!
```

### Direct Commands (Non-Interactive)

```bash
# Compress directly
aqea compress embeddings.json -o compressed.json

# With specific model
aqea compress embeddings.json --model text-mpnet -o compressed.json

# Validate quality
aqea validate embeddings.json
```

## Compression Performance

| Model | Input → Output | Compression | Quality |
|-------|----------------|-------------|---------|
| MiniLM | 384D → 13D | 30x | 88.8% |
| MPNet | 768D → 26D | 30x | 94.7% |
| E5-Large | 1024D → 35D | 29x | 94.9% |
| OpenAI Small | 1536D → 52D | 30x | 88-89% |
| Audio (wav2vec2) | 768D → 26D | 30x | 97.0% |
| Protein (ESM) | 320D → 11D | 29x | 93.8% |

With AQEA+PQ mode: **213-3072x compression** at 74-97% quality!

## Get Your API Key

1. Sign up at [https://aqea.ai](https://aqea.ai)
2. Go to Dashboard → API Keys
3. Create a new key
4. Use `/login` in the CLI

**Free tier:** 10,000 compressions/month

## Configuration

Config files are stored in:
- **Linux/macOS:** `~/.config/aqea/`
- **Windows:** `%APPDATA%\aqea\`

```
~/.config/aqea/
├── config.toml    # Settings
└── credentials    # Encrypted API key
```

## Uninstall

### macOS / Linux

```bash
curl -fsSL https://aqea.ai/uninstall.sh | bash
```

### Windows

```powershell
irm https://aqea.ai/uninstall.ps1 | iex
```

### Homebrew

```bash
brew uninstall aqea
```

## Building from Source

Requires Rust 1.70+:

```bash
git clone https://github.com/nextX-AG/aqea-cli.git
cd aqea-cli
cargo build --release
./target/release/aqea
```

## Links

- [Website](https://aqea.ai)
- [Documentation](https://aqea.ai/docs)
- [API Reference](https://aqea.ai/docs/api-reference)
- [Pricing](https://aqea.ai/pricing)

## License

MIT License - see [LICENSE](LICENSE) for details.

## About

AQEA Compress is developed by [nextX AG](https://nextx.ch), a Swiss AI company.

---

**Patent Pending** - Protected by 3 pending patents.
