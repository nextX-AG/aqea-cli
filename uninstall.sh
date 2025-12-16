#!/bin/bash
# AQEA CLI Uninstaller for macOS and Linux
# Usage: curl -fsSL https://aqea.ai/uninstall.sh | bash

set -e

INSTALL_DIR="$HOME/.aqea"
BINARY_NAME="aqea"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              AQEA CLI Uninstaller                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Remove binary and directory
if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}Removing AQEA installation...${NC}"
    rm -rf "$INSTALL_DIR"
    echo -e "${GREEN}✓${NC} Removed $INSTALL_DIR"
else
    echo -e "${YELLOW}AQEA not found at $INSTALL_DIR${NC}"
fi

# Remove from PATH in shell config
remove_from_path() {
    local file="$1"
    if [ -f "$file" ]; then
        if grep -q ".aqea/bin" "$file" 2>/dev/null; then
            # Create backup
            cp "$file" "$file.bak"
            # Remove AQEA lines
            grep -v ".aqea" "$file.bak" | grep -v "# AQEA CLI" > "$file" || true
            echo -e "${GREEN}✓${NC} Removed PATH entry from $file"
        fi
    fi
}

remove_from_path "$HOME/.zshrc"
remove_from_path "$HOME/.bashrc"
remove_from_path "$HOME/.profile"

# Remove config directory (optional)
CONFIG_DIR="$HOME/.config/aqea"
if [ -d "$CONFIG_DIR" ]; then
    echo ""
    echo -e "${YELLOW}Found configuration at $CONFIG_DIR${NC}"
    read -p "Remove configuration and credentials? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$CONFIG_DIR"
        echo -e "${GREEN}✓${NC} Removed configuration"
    else
        echo -e "${CYAN}Kept configuration${NC}"
    fi
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Uninstallation Complete                          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Restart your terminal to complete the process."
echo ""
