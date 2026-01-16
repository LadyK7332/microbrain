param(
    [string]$RepoRoot   = "C:\aiproj\microbrain",
    [string]$ArchiveDir = "$RepoRoot\archives"
)

Write-Host "=== MicroBrain archiver ==="
Write-Host "Repo root : $RepoRoot"
Write-Host "Archive   : $ArchiveDir"
Write-Host ""

# 1) Go to repo root
Set-Location $RepoRoot

# 2) Find all .py files recursively
Write-Host "Scanning for .py files..."
$pyFiles = Get-ChildItem -Path $RepoRoot -Recurse -Include *.py -File
    Where-Object { $_.FullName -notmatch '\\\.venv(\\|$)' }
    
if (-not $pyFiles -or $pyFiles.Count -eq 0) {
    Write-Host "No .py files found. Nothing to add to git."
} else {
    Write-Host "Found $($pyFiles.Count) .py files. Adding to git..."
    foreach ($file in $pyFiles) {
        # Use path relative to repo root so git is happy
        $relPath = Resolve-Path -LiteralPath $file.FullName -Relative
        git add $relPath
    }
}

Write-Host "Git add complete."
Write-Host ""

# 3) Prepare archive directory
if (-not (Test-Path $ArchiveDir)) {
    Write-Host "Creating archive directory: $ArchiveDir"
    New-Item -ItemType Directory -Path $ArchiveDir | Out-Null
}

# 4) Build archive name
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipName   = "microbrain-src_$timestamp.zip"
$zipPath   = Join-Path $ArchiveDir $zipName

Write-Host "Creating archive: $zipPath"
Write-Host ""

# 5) Collect files to zip (exclude noise)
$itemsToZip = Get-ChildItem $RepoRoot -Recurse -Force |
    Where-Object {
        $_.FullName -notmatch '\\\.git(\\|$)'        -and
        $_.FullName -notmatch '\\\.venv(\\|$)'       -and
        $_.FullName -notmatch '\\archives(\\|$)'     -and
        $_.FullName -notmatch '\\__pycache__(\\|$)'
    }

if (-not $itemsToZip -or $itemsToZip.Count -eq 0) {
    Write-Host "No files selected for archive. Aborting zip."
    exit 1
}

Compress-Archive -Path $itemsToZip.FullName -DestinationPath $zipPath -Force

Write-Host ""
Write-Host "Archive created:"
Write-Host "  $zipPath"
Write-Host "Done."
