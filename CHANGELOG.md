# Changelog

All notable changes to AQEA CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-12-19

### Changed (Breaking)
- **Compression now requires API authentication** - no local compression
- Removed `--weights` flag (security improvement)
- Removed `--offline` flag (security improvement)
- Internal commands (`validate`, `info`) hidden from help

### Added
- API-based compression via `/api/v1/compress/batch`
- Clear error messages for unauthenticated users
- Batch compression support for better performance

### Security
- AQEA weights never leave the server
- CLI cannot compress without valid API key
- All embeddings processed securely via HTTPS

## [0.1.0] - 2025-12-16

### Added
- Initial public release
- Interactive REPL mode with syntax highlighting
- API authentication with secure credential storage
- **3-Stage Compression Pipeline (AQEA+PQ)** - Up to 3072x compression!
- Support for 7 embedding models:
  - text-minilm (384D → 13D)
  - text-mpnet (768D → 26D)
  - text-e5large (1024D → 35D)
  - openai-small (1536D → 52D)
  - openai-large (3072D → 105D)
  - audio-wav2vec2 (768D → 26D)
  - protein-esm (320D → 11D)
- 2-Stage compression: `/compress` (29x, works with any vector DB)
- 3-Stage compression: `/compress-pq` (213-3072x, requires codebook)
- New commands:
  - `/mode` - Select compression mode (AQEA or AQEA+PQ)
  - `/subs [n]` - Set PQ subvector count for 3-Stage
  - `/compress-pq <file>` - 3-Stage compression
- Quality validation: `/validate`
- Model selection: `/models`, `/model <name>`
- Session management: `/login`, `/logout`, `/status`
- Cross-platform support (Linux, macOS, Windows)
- ARM64 support (Apple Silicon, Linux ARM)
- One-line installers for all platforms

### Performance
- Audio embeddings: 236x compression @ 96.5% quality
- Protein embeddings: 213x compression @ 93.4% quality
- OpenAI Large: 3072x compression @ 79% quality (maximum)

### Security
- Encrypted credential storage
- API key never stored in plaintext
- Secure HTTPS-only communication

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2025-01 | Initial release |

[Unreleased]: https://github.com/nextX-AG/aqea-cli/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nextX-AG/aqea-cli/releases/tag/v0.1.0
