# AQEA CLI Uninstaller for Windows
# Usage: irm https://aqea.ai/uninstall.ps1 | iex

$ErrorActionPreference = "Stop"

$InstallDir = "$env:USERPROFILE\.aqea"
$ConfigDir = "$env:APPDATA\aqea"

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              AQEA CLI Uninstaller                             ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Remove installation directory
if (Test-Path $InstallDir) {
    Write-Host "Removing AQEA installation..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $InstallDir
    Write-Host "✓ Removed $InstallDir" -ForegroundColor Green
} else {
    Write-Host "AQEA not found at $InstallDir" -ForegroundColor Yellow
}

# Remove from PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -like "*\.aqea\bin*") {
    $newPath = ($userPath -split ';' | Where-Object { $_ -notlike "*\.aqea*" }) -join ';'
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "✓ Removed from PATH" -ForegroundColor Green
}

# Remove config (optional)
if (Test-Path $ConfigDir) {
    Write-Host ""
    Write-Host "Found configuration at $ConfigDir" -ForegroundColor Yellow
    $response = Read-Host "Remove configuration and credentials? [y/N]"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Remove-Item -Recurse -Force $ConfigDir
        Write-Host "✓ Removed configuration" -ForegroundColor Green
    } else {
        Write-Host "Kept configuration" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              Uninstallation Complete                          ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Restart your terminal to complete the process."
Write-Host ""
