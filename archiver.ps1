# Build a source zip of tracked + untracked files that are NOT ignored.
# Requires: Git in PATH, PowerShell 5+

$stamp = Get-Date -Format yyyyMMdd_HHmmss
$zip   = "microbrain-src_$stamp.zip"

# List files that are tracked OR untracked-but-not-ignored
$files = git ls-files -co --exclude-standard

if (-not $files) {
    Write-Host "No files to archive (check your ignore rules)." -ForegroundColor Yellow
    exit 0
}

# Compress-Archive can take an array of relative paths
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $files -DestinationPath $zip -Force
Write-Host "Wrote $zip"
