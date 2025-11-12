# --- CONFIG ---
$Root         = 'C:\aiproj\microbrain'  # project root
$MaxSizeBytes = 5MB                     # skip larger files
$DryRun       = $false                  # set $true for preview only (no writes)

# Folders to skip (regex)
$SkipPattern = '\\\.git\\|\\\.venv\\|\\__pycache__\\|\\build\\|\\dist\\|\\node_modules\\|\\\.idea\\|\\\.vscode\\'

# Text-like extensions to scan (lowercase)
$TextExt = '.py','.ps1','.psm1','.psd1','.bat','.cmd','.txt','.md','.toml','.ini','.cfg','.conf',
           '.json','.jsonl','.yaml','.yml','.xml','.html','.htm','.css','.js','.ts','.csproj','.sln',
           '.pyproj','.ipynb'

# Literal replacements (handle raw, forward-slash, and string-escaped backslashes)
$Replacements = @(
  @{ From = 'G:\\microbrain'; To = 'C:\\aiproj\\microbrain' }, # in code strings
  @{ From = 'G:\microbrain' ; To = 'C:\aiproj\microbrain'  },  # normal path
  @{ From = 'G:/microbrain' ; To = 'C:\aiproj\microbrain'  }   # forward slashes
)

# --- WORK ---
Set-Location $Root
$Report     = New-Object System.Collections.Generic.List[object]
$Changed    = 0
$Scanned    = 0
$maxPerFile = 5

Get-ChildItem -Path $Root -Recurse -File -Force |
  Where-Object {
    $_.Length -lt $MaxSizeBytes -and
    $_.FullName -notmatch $SkipPattern -and
    ($TextExt -contains ($_.Extension.ToLower()))
  } |
  ForEach-Object {
    $path = $_.FullName; $Scanned++

    # Read text (favor UTF-8; fall back if needed)
    try { $old = Get-Content -Raw -Encoding UTF8 -LiteralPath $path }
    catch { $old = Get-Content -Raw -LiteralPath $path }

    # Apply replacements
    $new = $old
    foreach ($rep in $Replacements) { $new = $new.Replace($rep.From, $rep.To) }

    if ($new -ne $old) {
      $Changed++

      if (-not $DryRun) {
        # Backup once per file
        $bak = "$path.bak"
        if (-not (Test-Path -LiteralPath $bak)) {
          Copy-Item -LiteralPath $path -Destination $bak
        }
        # Write updated content
        Set-Content -LiteralPath $path -Value $new -Encoding UTF8
      }

      # Build a compact preview by showing only lines that contained any "From"
      $oldLines = $old -split "`r?`n"
      $shown    = 0
      for ($i=0; $i -lt $oldLines.Count -and $shown -lt $maxPerFile; $i++) {
        $line = $oldLines[$i]
        $hit  = $false
        foreach ($rep in $Replacements) {
          if ($line.Contains($rep.From)) { $hit = $true; break }
        }
        if ($hit) {
          $updated = $line
          foreach ($rep in $Replacements) { $updated = $updated.Replace($rep.From, $rep.To) }
          $Report.Add([PSCustomObject]@{
            File   = $path
            Line   = $i + 1
            Before = $line.Trim()
            After  = $updated.Trim()
          })
          $shown++
        }
      }
    }
  }

# --- OUTPUT ---
if ($Changed -gt 0) {
  $Report | Sort-Object File, Line | Format-Table -Auto File, Line, Before, After
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $csv   = Join-Path $Root "path-migration-report_$stamp.csv"
  $Report | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
  Write-Host "`nChanged files: $Changed / Scanned: $Scanned" -ForegroundColor Green
  Write-Host "Saved CSV report to: $csv" -ForegroundColor Cyan
  if (-not $DryRun) { Write-Host "Backups saved as *.bak next to each modified file." -ForegroundColor Yellow }
} else {
  Write-Host "No changes needed. Scanned: $Scanned files." -ForegroundColor Gray
}
