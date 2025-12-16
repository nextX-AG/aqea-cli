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
| `/mode` | **NEW:** Select compression mode (AQEA or AQEA+PQ) |
| `/subs [n]` | **NEW:** Set PQ subvector count (3-Stage only) |
| `/compress <file>` | Compress vectors (2-Stage AQEA) |
| `/compress-pq <file>` | **NEW:** 3-Stage compression (AQEA+PQ) |
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

# 2-Stage Compression (AQEA only, ~29x)
> /compress embeddings.json
📥 Loaded 1000 vectors (384D)
📦 Using model: text-minilm
✅ Compressed to 13D (29x compression)
💾 Saved to embeddings_compressed.json

# 3-Stage Compression (AQEA+PQ, up to 3072x) 🔥
> /mode
? Select compression mode:
  ❯ AQEA (2-Stage)      ~29x compression
    AQEA+PQ (3-Stage)   213-3072x compression ⚡

> /model audio-wav2vec2
> /subs 13
> /compress-pq audio_embeddings.json
📥 Loaded 1000 vectors (768D)
📦 Model: audio-wav2vec2 | Mode: AQEA+PQ (13 subvectors)
✅ Compressed: 768D → 13 bytes (236x)
💾 Saved to audio_embeddings.pq.json

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

### 2-Stage (AQEA only) - Works with ANY vector DB

| Model | Input → Output | Compression | Quality |
|-------|----------------|-------------|---------|
| MiniLM | 384D → 13D | 29x | 97.1% |
| MPNet | 768D → 26D | 29x | 98.3% |
| E5-Large | 1024D → 35D | 29x | 98.2% |
| OpenAI Small | 1536D → 52D | 29x | 95.0% |
| OpenAI Large | 3072D → 105D | 29x | 95.0% |
| Audio (wav2vec2) | 768D → 26D | 29x | 97.1% |
| Protein (ESM) | 320D → 11D | 29x | 95.0% |

### 3-Stage (AQEA+PQ) - Maximum Compression 🔥

| Model | Subvectors | Output | Compression | Quality |
|-------|------------|--------|-------------|---------|
| Audio (wav2vec2) | 13 | 13 bytes | **236x** | 96.5% |
| Audio (wav2vec2) | 4 | 4 bytes | **768x** | 88.4% |
| Protein (ESM) | 6 | 6 bytes | **213x** | 93.4% |
| OpenAI Large | 10 | 10 bytes | **1229x** | 87.8% |
| OpenAI Large | 4 | 4 bytes | **3072x** | 79.0% |

> **Note:** 3-Stage compression requires the codebook for similarity search. Download via `/codebooks` or API.

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
