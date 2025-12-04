param(
    # Where to put the zip (defaults to repo root)
    [string]$OutputDir = "."
)

$ErrorActionPreference = "Stop"

# Adjust if you ever move the repo
$repoRoot = "C:\aiproj\microbrain"

Push-Location $repoRoot
try {
    # Timestamped filename like microbrain-src_20251204_001535.zip
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $zipName   = "microbrain-src_$timestamp.zip"
    $destPath  = Join-Path $OutputDir $zipName

    if (Test-Path $destPath) {
        Remove-Item $destPath -Force
    }

    # 1) All tracked files
    $trackedFiles = git ls-files

    # 2) Extra non-tracked files you still want (optional)
    $extraFiles = @(
        "DxDiag.txt",
        "installed_apps.txt",
        "AMD Radeon RX 6600.json"
    ) | Where-Object { Test-Path $_ }

    $allRelPaths = $trackedFiles + $extraFiles

    if ($allRelPaths.Count -eq 0) {
        Write-Host "No files to archive; did git ls-files return anything?" -ForegroundColor Yellow
        return
    }

    # 3) Create a temporary staging directory
    $stageRoot = Join-Path $env:TEMP ("microbrain_srcpack_" + $timestamp)
    if (Test-Path $stageRoot) {
        Remove-Item $stageRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stageRoot | Out-Null

    # 4) Copy each file into the staging area, preserving relative paths
    foreach ($rel in $allRelPaths) {
        $src = Join-Path $repoRoot $rel
        if (-not (Test-Path $src)) {
            continue
        }
        $dest = Join-Path $stageRoot $rel
        $destDir = Split-Path $dest -Parent
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Copy-Item -Path $src -Destination $dest -Force
    }

    # 5) Compress the staged tree, preserving structure
    Compress-Archive -Path (Join-Path $stageRoot '*') -DestinationPath $destPath -CompressionLevel Optimal

    Write-Host "Created archive: $destPath" -ForegroundColor Green
    Write-Host "Included $($trackedFiles.Count) tracked files and $($extraFiles.Count) extra files."

} finally {
    # 6) Clean up staging dir
    if ($stageRoot -and (Test-Path $stageRoot)) {
        Remove-Item $stageRoot -Recurse -Force
    }
    Pop-Location
}
