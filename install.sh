#!/bin/bash
# AQEA CLI Installer for macOS and Linux
# Usage: curl -fsSL https://aqea.ai/install.sh | bash

set -e

REPO="nextX-AG/aqea-cli"
BINARY_NAME="aqea"
INSTALL_DIR="${AQEA_INSTALL_DIR:-$HOME/.aqea/bin}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              AQEA CLI Installer                               ║"
echo "║       Compress embeddings up to 3000x                         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Detect OS and Architecture
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$os" in
        linux*)  os="unknown-linux-gnu" ;;
        darwin*) os="apple-darwin" ;;
        *)       echo -e "${RED}Unsupported OS: $os${NC}"; exit 1 ;;
    esac

    case "$arch" in
        x86_64|amd64)  arch="x86_64" ;;
        arm64|aarch64) arch="aarch64" ;;
        *)             echo -e "${RED}Unsupported architecture: $arch${NC}"; exit 1 ;;
    esac

    echo "${arch}-${os}"
}

# Get latest release version
get_latest_version() {
    curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null | \
        grep '"tag_name":' | sed -E 's/.*"v?([^"]+)".*/\1/' || echo "0.1.0"
}

# Download and install
install_aqea() {
    local platform=$(detect_platform)
    local version=$(get_latest_version)

    echo -e "${CYAN}Platform:${NC} $platform"
    echo -e "${CYAN}Version:${NC}  $version"
    echo ""

    # Create install directory
    mkdir -p "$INSTALL_DIR"

    # Download URL
    local download_url="https://github.com/${REPO}/releases/download/v${version}/aqea-${platform}.tar.gz"
    local tmp_dir=$(mktemp -d)

    echo -e "${YELLOW}Downloading AQEA CLI...${NC}"

    # Try GitHub releases first
    if curl -fsSL "$download_url" -o "$tmp_dir/aqea.tar.gz" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Downloaded from GitHub releases"
        tar -xzf "$tmp_dir/aqea.tar.gz" -C "$tmp_dir"

        # Find the binary (might be in root or subdirectory)
        if [ -f "$tmp_dir/aqea" ]; then
            mv "$tmp_dir/aqea" "$INSTALL_DIR/$BINARY_NAME"
        elif [ -f "$tmp_dir/*/aqea" ]; then
            mv "$tmp_dir"/*/aqea "$INSTALL_DIR/$BINARY_NAME"
        else
            echo -e "${RED}Binary not found in archive${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Pre-built binary not available for this platform.${NC}"
        echo -e "${YELLOW}Building from source...${NC}"

        # Check for Rust
        if ! command -v cargo &> /dev/null; then
            echo -e "${YELLOW}Installing Rust...${NC}"
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
        fi

        # Clone and build
        git clone --depth 1 "https://github.com/${REPO}.git" "$tmp_dir/aqea-cli"
        cd "$tmp_dir/aqea-cli"
        cargo build --release
        mv target/release/aqea "$INSTALL_DIR/$BINARY_NAME"
    fi

    chmod +x "$INSTALL_DIR/$BINARY_NAME"

    # Cleanup
    rm -rf "$tmp_dir"

    echo -e "${GREEN}✓${NC} Installed to $INSTALL_DIR/$BINARY_NAME"
}

# Add to PATH
setup_path() {
    local shell_rc=""
    local shell_name=""

    # Detect shell
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        shell_rc="$HOME/.zshrc"
        shell_name="zsh"
    elif [ -n "$BASH_VERSION" ] || [ -f "$HOME/.bashrc" ]; then
        shell_rc="$HOME/.bashrc"
        shell_name="bash"
    elif [ -f "$HOME/.profile" ]; then
        shell_rc="$HOME/.profile"
        shell_name="sh"
    fi

    if [ -n "$shell_rc" ]; then
        local path_line="export PATH=\"\$HOME/.aqea/bin:\$PATH\""

        if ! grep -q ".aqea/bin" "$shell_rc" 2>/dev/null; then
            echo "" >> "$shell_rc"
            echo "# AQEA CLI" >> "$shell_rc"
            echo "$path_line" >> "$shell_rc"
            echo -e "${GREEN}✓${NC} Added to PATH in $shell_rc"
        else
            echo -e "${GREEN}✓${NC} PATH already configured"
        fi
    fi
}

# Verify installation
verify_installation() {
    if [ -x "$INSTALL_DIR/$BINARY_NAME" ]; then
        local version=$("$INSTALL_DIR/$BINARY_NAME" --version 2>/dev/null || echo "installed")

        echo ""
        echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║              Installation Complete!                           ║${NC}"
        echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "Restart your terminal or run:"
        echo -e "  ${CYAN}source ~/.zshrc${NC}  (or ~/.bashrc)"
        echo ""
        echo -e "Then start AQEA:"
        echo -e "  ${CYAN}aqea${NC}           Start interactive mode"
        echo -e "  ${CYAN}aqea --help${NC}    Show help"
        echo ""
        echo -e "Get your API key at: ${CYAN}https://aqea.ai${NC}"
        echo ""
    else
        echo -e "${RED}Installation failed!${NC}"
        exit 1
    fi
}

# Main
main() {
    install_aqea
    setup_path
    verify_installation
}

main "$@"
