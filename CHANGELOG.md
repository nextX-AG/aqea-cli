# Changelog

All notable changes to AQEA CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-XX

### Added
- Initial public release
- Interactive REPL mode with syntax highlighting
- API authentication with secure credential storage
- Support for 8 embedding models:
  - text-minilm (384D)
  - text-mpnet (768D)
  - text-e5large (1024D)
  - mistral-embed (1024D)
  - openai-small (1536D)
  - openai-large (3072D)
  - audio-wav2vec2 (768D)
  - protein-esm (320D)
- Compression commands: `/compress`, `/decompress`
- Quality validation: `/validate`
- Model selection: `/models`, `/model <name>`
- Session management: `/login`, `/logout`, `/status`
- Cross-platform support (Linux, macOS, Windows)
- ARM64 support (Apple Silicon, Linux ARM)
- One-line installers for all platforms
- Offline mode with local weights

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
