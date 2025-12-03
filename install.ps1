Param(
    [string]$InstallDir = "$Env:LOCALAPPDATA\Programs\tokuin",
    [switch]$SkipModels
)

$ErrorActionPreference = 'Stop'

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' is not available."
    }
}

Require-Command -Name 'Invoke-WebRequest'
Require-Command -Name 'Expand-Archive'
Require-Command -Name 'Get-FileHash'

$repo = 'nooscraft/tokuin'
$apiUrl = "https://api.github.com/repos/$repo/releases/latest"

$osArch = (Get-CimInstance Win32_OperatingSystem).OSArchitecture
if ($osArch -notmatch '64') {
    throw "Unsupported architecture: $osArch (requires 64-bit Windows)."
}

$target = 'x86_64-pc-windows-msvc'

Write-Host "Fetching latest release metadata for $target..."
$response = Invoke-RestMethod -Uri $apiUrl -UseBasicParsing

$asset = $response.assets | Where-Object { $_.name -like "*${target}.zip" } | Select-Object -First 1
if (-not $asset) {
    throw "Unable to find release asset for $target."
}

$checksumAsset = $response.assets | Where-Object { $_.name -eq 'checksums.txt' } | Select-Object -First 1

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tempDir | Out-Null

try {
    $zipPath = Join-Path $tempDir $asset.name
    Write-Host "Downloading $($asset.name)..."
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath -UseBasicParsing

    if ($checksumAsset) {
        $checksumPath = Join-Path $tempDir $checksumAsset.name
        Write-Host 'Downloading checksums.txt...'
        Invoke-WebRequest -Uri $checksumAsset.browser_download_url -OutFile $checksumPath -UseBasicParsing

        $expectedLine = Select-String -Path $checksumPath -Pattern [Regex]::Escape($asset.name) | Select-Object -First 1
        if ($expectedLine) {
            $expected = ($expectedLine.Line -split '\s+')[0]
            $actual = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash.ToLower()
            if ($expected.ToLower() -ne $actual) {
                throw "Checksum verification failed for $($asset.name)."
            } else {
                Write-Host 'Checksum verified.'
            }
        } else {
            Write-Warning "Checksum entry not found for $($asset.name); skipping verification."
        }
    } else {
        Write-Warning 'checksums.txt not found; skipping checksum verification.'
    }

    $extractDir = Join-Path $tempDir 'extracted'
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    $binaryPath = Join-Path $extractDir 'tokuin.exe'
    if (-not (Test-Path $binaryPath)) {
        throw 'tokuin.exe not found in extracted archive.'
    }

    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    $destination = Join-Path $InstallDir 'tokuin.exe'
    Copy-Item -Path $binaryPath -Destination $destination -Force

    Write-Host "Installed tokuin to $destination"

    $pathEntries = ($Env:Path -split ';')
    if (-not ($pathEntries -contains $InstallDir)) {
        Write-Warning "$InstallDir is not currently on your PATH. Add it to launch tokuin directly."
    }

} finally {
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
}

Write-Host 'Done!'

# Optionally setup embedding models
if (-not $SkipModels) {
    Write-Host ''
    Write-Host '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
    Write-Host 'Setting up embedding models...'
    Write-Host '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
    try {
        & $destination setup models
        Write-Host '✓ Models setup complete!'
    } catch {
        Write-Warning "Model setup failed or skipped. You can run 'tokuin setup models' later."
    }
} else {
    Write-Host ''
    Write-Host "Skipping model setup. Run 'tokuin setup models' to download embedding models."
}

