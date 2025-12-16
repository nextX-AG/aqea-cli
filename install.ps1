# AQEA CLI Installer for Windows
# Usage: irm https://aqea.ai/install.ps1 | iex

$ErrorActionPreference = "Stop"

$Repo = "nextX-AG/aqea-cli"
$BinaryName = "aqea.exe"
$InstallDir = "$env:USERPROFILE\.aqea\bin"

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              AQEA CLI Installer                               ║" -ForegroundColor Cyan
Write-Host "║       Compress embeddings up to 3000x                         ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Detect architecture
function Get-Platform {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    switch ($arch) {
        "X64"   { return "x86_64-pc-windows-msvc" }
        "Arm64" { return "aarch64-pc-windows-msvc" }
        default {
            Write-Host "Unsupported architecture: $arch" -ForegroundColor Red
            exit 1
        }
    }
}

# Get latest version
function Get-LatestVersion {
    try {
        $release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
        return $release.tag_name -replace '^v', ''
    } catch {
        return "0.1.0"
    }
}

# Download and install
function Install-AQEA {
    $platform = Get-Platform
    $version = Get-LatestVersion

    Write-Host "Platform: $platform" -ForegroundColor Cyan
    Write-Host "Version:  $version" -ForegroundColor Cyan
    Write-Host ""

    # Create install directory
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

    $downloadUrl = "https://github.com/$Repo/releases/download/v$version/aqea-$platform.zip"
    $tempFile = "$env:TEMP\aqea.zip"
    $tempDir = "$env:TEMP\aqea-extract"

    Write-Host "Downloading AQEA CLI..." -ForegroundColor Yellow

    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $tempFile -UseBasicParsing
        Write-Host "✓ Downloaded from GitHub releases" -ForegroundColor Green

        # Extract
        if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
        Expand-Archive -Path $tempFile -DestinationPath $tempDir -Force

        # Find the exe
        $exePath = Get-ChildItem -Path $tempDir -Filter "aqea.exe" -Recurse | Select-Object -First 1
        if ($exePath) {
            Move-Item -Force $exePath.FullName "$InstallDir\$BinaryName"
        } else {
            throw "Binary not found in archive"
        }

        # Cleanup
        Remove-Item -Force $tempFile -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
    } catch {
        Write-Host "Pre-built binary not available. Building from source..." -ForegroundColor Yellow

        # Check for Rust
        if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
            Write-Host "Installing Rust..." -ForegroundColor Yellow
            Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "$env:TEMP\rustup-init.exe"
            & "$env:TEMP\rustup-init.exe" -y
            $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
        }

        # Clone and build
        $buildDir = "$env:TEMP\aqea-build"
        if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
        git clone --depth 1 "https://github.com/$Repo.git" $buildDir
        Push-Location $buildDir
        cargo build --release
        Move-Item -Force "target\release\aqea.exe" "$InstallDir\$BinaryName"
        Pop-Location

        # Cleanup
        Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
    }

    Write-Host "✓ Installed to $InstallDir\$BinaryName" -ForegroundColor Green
}

# Add to PATH
function Set-AQEAPath {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")

    if ($userPath -notlike "*\.aqea\bin*") {
        $newPath = "$InstallDir;$userPath"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        $env:PATH = "$InstallDir;$env:PATH"
        Write-Host "✓ Added to PATH" -ForegroundColor Green
    } else {
        Write-Host "✓ PATH already configured" -ForegroundColor Green
    }
}

# Verify installation
function Test-Installation {
    if (Test-Path "$InstallDir\$BinaryName") {
        Write-Host ""
        Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
        Write-Host "║              Installation Complete!                           ║" -ForegroundColor Green
        Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
        Write-Host ""
        Write-Host "Restart your terminal, then run:"
        Write-Host ""
        Write-Host "  aqea           Start interactive mode" -ForegroundColor Cyan
        Write-Host "  aqea --help    Show help" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Get your API key at: https://aqea.ai" -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host "Installation failed!" -ForegroundColor Red
        exit 1
    }
}

# Main
Install-AQEA
Set-AQEAPath
Test-Installation
