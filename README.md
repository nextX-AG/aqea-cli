# AQEA CLI

> Compress embeddings up to 3000x while preserving 85-97% similarity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/nextX-AG/aqea-cli)](https://github.com/nextX-AG/aqea-cli/releases)
[![Platform Support](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-blue)](https://github.com/nextX-AG/aqea-cli/releases)

AQEA Compress is a CLI tool for extreme embedding compression. Reduce your vector storage costs by up to 99% while maintaining semantic search quality.

## 🆕 What's New in v0.3.0

- **Go-Live** - Full API integration with compress.aqea.ai
- **`aqea train`** - Train custom AQEA weights with Focus Steering
- **Sampling Profiles** - Choose between `random-v1`, `coverage-v1`, or `focus-v1`
- **`aqea pq train`** - Train custom PQ codebooks for extreme compression (up to 585x)
- **`aqea pq list`** - List all available PQ codebooks

## Quick Install

### macOS / Linux

```bash
curl -fsSL https://compress.aqea.ai/install.sh | bash
```

### Windows (PowerShell)

```powershell
irm https://compress.aqea.ai/install.ps1 | iex
```

### Manual Download

Download the latest release for your platform from [GitHub Releases](https://github.com/nextX-AG/aqea-cli/releases/latest).

| Platform | Download |
|----------|----------|
| macOS (Apple Silicon) | [aqea-aarch64-apple-darwin.tar.gz](https://github.com/nextX-AG/aqea-cli/releases/download/v0.3.0/aqea-aarch64-apple-darwin.tar.gz) |
| macOS (Intel) | [aqea-x86_64-apple-darwin.tar.gz](https://github.com/nextX-AG/aqea-cli/releases/download/v0.3.0/aqea-x86_64-apple-darwin.tar.gz) |
| Linux (x86_64) | [aqea-x86_64-unknown-linux-gnu.tar.gz](https://github.com/nextX-AG/aqea-cli/releases/download/v0.3.0/aqea-x86_64-unknown-linux-gnu.tar.gz) |
| Linux (ARM64) | [aqea-aarch64-unknown-linux-gnu.tar.gz](https://github.com/nextX-AG/aqea-cli/releases/download/v0.3.0/aqea-aarch64-unknown-linux-gnu.tar.gz) |
| Windows | [aqea-x86_64-pc-windows-msvc.zip](https://github.com/nextX-AG/aqea-cli/releases/download/v0.3.0/aqea-x86_64-pc-windows-msvc.zip) |

## Supported Platforms

| OS | x86_64 (Intel/AMD) | ARM64 (Apple Silicon/ARM) |
|----|:------------------:|:-------------------------:|
| Linux | ✅ | ✅ |
| macOS | ✅ | ✅ |
| Windows | ✅ | ✅ |

## Usage

### Commands Overview

```bash
aqea --help
```

| Command | Description |
|---------|-------------|
| `aqea auth login` | Authenticate with your API key |
| `aqea auth status` | Check authentication status |
| `aqea models` | List available compression models |
| `aqea compress` | Compress embeddings via API |
| `aqea test` | Run compression quality test |
| `aqea train` | **NEW:** Train custom AQEA weights |
| `aqea pq train` | **NEW:** Train custom PQ codebooks |
| `aqea pq list` | List available PQ codebooks |
| `aqea usage` | Show API usage statistics |
| `aqea config` | Manage configuration |

### Quick Start

```bash
# 1. Login with your API key
aqea auth login

# 2. Check available models
aqea models

# 3. Run a quality test
aqea test --model text-mpnet

# 4. Compress your embeddings
aqea compress embeddings.json -o compressed.json --model text-mpnet
```

## 🎯 Focus Steering (NEW in v0.2.5)

Train custom "lenses" that change retrieval behavior without retraining your base transformer!

### Sampling Profiles

| Profile | Focus | Best For |
|---------|-------|----------|
| `random-v1` | Balanced | General purpose |
| `focus-v1` | Cluster centers | Precision (tight clusters) |
| `coverage-v1` | Uniform coverage | Discovery (diverse results) |

### Training Examples

```bash
# Basic training with focus sampling (default)
aqea train \
  --input embeddings.json \
  --output my_weights.aqwt \
  --sampling focus-v1

# Discovery-focused training (uniform coverage)
aqea train \
  --input embeddings.json \
  --output discovery_weights.aqwt \
  --sampling coverage-v1 \
  --train-split 30

# Train with PQ for extreme compression (585x)
aqea train \
  --input embeddings.json \
  --output extreme.aqwt \
  --pq 7 \
  --pq-output codebook.json
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | required | Input embeddings (JSON or AQED format) |
| `--output` | required | Output weights file (.aqwt) |
| `--train-split` | 20% | Percentage of data for training |
| `--sampling` | focus-v1 | Sampling profile (random-v1, focus-v1, coverage-v1) |
| `--samples` | auto | Fixed sample count (e.g., "500" or "50%") |
| `--pq` | - | Train PQ codebook with N subvectors |
| `--quick` | false | Faster training, slightly lower quality |

## Compression Performance

### Standard Compression - Works with ANY vector DB

| Model | Input → Output | Compression | Quality |
|-------|----------------|-------------|---------|
| MiniLM | 384D → 13D | 29x | 97.1% |
| MPNet | 768D → 26D | 29x | 98.3% |
| E5-Large | 1024D → 35D | 29x | 98.2% |
| OpenAI Small | 1536D → 52D | 29x | 95.0% |
| OpenAI Large | 3072D → 105D | 29x | 95.0% |
| Audio (wav2vec2) | 768D → 26D | 29x | 97.1% |

### Extreme Compression - Maximum Compression 🔥

| Model | Subvectors | Output | Compression | Quality |
|-------|------------|--------|-------------|---------|
| E5-Large | 17 | 17 bytes | **585x** | 93.5% |
| E5-Large | 31 | 31 bytes | **241x** | 97.2% |
| OpenAI Large | 10 | 10 bytes | **1229x** | 87.8% |

## Interactive Mode (REPL)

```bash
# Start interactive shell
aqea

# You'll see:
# 🔷 AQEA Compress CLI v0.3.0
# Type /help for commands, or enter vectors to compress
```

### REPL Commands

| Command | Description |
|---------|-------------|
| `/login` | Authenticate with your API key |
| `/models` | List available compression models |
| `/model <name>` | Select a compression model |
| `/compress <file>` | Compress vectors |
| `/validate <file>` | Test compression quality |
| `/status` | Show current session status |
| `/help` | Show all available commands |
| `/quit` | Exit the CLI |

## Get Your API Key

1. Sign up at [https://compress.aqea.ai](https://compress.aqea.ai)
2. Go to Dashboard → API Keys
3. Create a new key
4. Use `aqea auth login` in the CLI

**Free tier:** 10,000 compressions/month

## Configuration

Config files are stored in:
- **Linux/macOS:** `~/.config/aqea/` or `~/.aqea/`
- **Windows:** `%APPDATA%\aqea\`

```
~/.aqea/
├── config.toml    # Settings
└── credentials    # API key
```

## Building from Source

Requires Rust 1.70+:

```bash
git clone https://github.com/nextX-AG/aqea-cli.git
cd aqea-cli
cargo build --release
./target/release/aqea --version
```

## Uninstall

### macOS / Linux

```bash
curl -fsSL https://compress.aqea.ai/uninstall.sh | bash
```

### Windows

```powershell
irm https://compress.aqea.ai/uninstall.ps1 | iex
```


## Links

- 🌐 [Website](https://compress.aqea.ai)
- 🎮 [Live Demo](https://compress.aqea.ai/demo)
- 🚀 [Platform](https://platform.aqea.ai)
- 📖 [Documentation](https://compress.aqea.ai/docs)
- 📚 [API Reference](https://compress.aqea.ai/docs/api-reference)

## Related Repositories

| Repository | Description |
|------------|-------------|
| [aqea-technical-report](https://github.com/nextX-AG/aqea-technical-report) | Scientific methodology, benchmarks & reproducibility |

## Changelog

### v0.3.0 (2026-01-02)
- 🚀 **Go-Live Release**
- 🔗 Updated all URLs to compress.aqea.ai
- 📦 Stable sampling profiles: `random-v1`, `focus-v1`, `coverage-v1`
- ✅ Full API integration tested

### v0.2.5 (2025-12-23)
- ✨ Added `aqea train` command for custom weight training
- ✨ Added Focus Steering with sampling strategies
- ✨ Added `aqea pq train` for custom PQ codebook training
- ✨ Progressive training with early stopping
- 🔧 Unified training system

### v0.2.4 (2025-12-19)
- API-only compression (no local weights required)
- Multi-platform binaries
- Improved error handling

## License

MIT License - see [LICENSE](LICENSE) for details.

## About

AQEA Compress is developed by [nextX AG](https://nextx.ch), a Swiss AI company.

---

**Patent Pending** - Protected by 4 pending patents.
