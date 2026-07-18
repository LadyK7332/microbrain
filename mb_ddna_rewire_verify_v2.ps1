param(
  [string]$RepoRoot = (Get-Location).Path,
  [string]$MemDir = "Z:\Memory"
)

$ErrorActionPreference = "Continue"
$missing = $false

function Check-File($rel) {
  $path = Join-Path $RepoRoot $rel
  if (Test-Path $path) {
    Write-Host "PASS found $rel" -ForegroundColor Green
    return $true
  }
  Write-Host "FAIL missing $rel" -ForegroundColor Red
  $script:missing = $true
  return $false
}

function Check-Marker($rel, $marker) {
  $path = Join-Path $RepoRoot $rel
  if (!(Test-Path $path)) {
    Write-Host "FAIL $rel missing before marker $marker" -ForegroundColor Red
    $script:missing = $true
    return
  }
  $text = Get-Content -Raw -Encoding UTF8 $path
  if ($text.Contains($marker)) {
    Write-Host "PASS $rel contains $marker" -ForegroundColor Green
  } else {
    Write-Host "FAIL $rel missing marker $marker" -ForegroundColor Red
    $script:missing = $true
  }
}

$files = @(
  "microbrain\pdna\access.py",
  "microbrain\pdna\core.py",
  "microbrain\hormone.py",
  "microbrain\mind.py",
  "microbrain\neurons\pdna_state_neuron.py",
  "microbrain\neurons\reward_novelty_pulse_neuron.py",
  "microbrain\neurons\boredom_drive_neuron.py",
  "microbrain\neurons\attention_controller.py",
  "microbrain\neurons\thought_momentum_neuron.py",
  "microbrain\neurons\thought_turn_arbitration_neuron.py",
  "microbrain\ui\textual_bridge.py",
  "tests\test_pdna_ddna_profile_v2.py",
  "docs\demi_ddna_metabolic_rewire_v1_20260703.md"
)

foreach ($f in $files) { Check-File $f | Out-Null }

Check-Marker "microbrain\pdna\core.py" "extra_sections"
Check-Marker "microbrain\pdna\core.py" "ddna_mutators"
Check-Marker "microbrain\pdna\core.py" "reinforcement_model"
Check-Marker "microbrain\pdna\access.py" "drive:ddna_modulators"
Check-Marker "microbrain\pdna\access.py" "pdna:affect_model"
Check-Marker "microbrain\pdna\access.py" "pdna:wans"
Check-Marker "microbrain\mind.py" "publish_pdna_runtime_profile"
Check-Marker "microbrain\mind.py" "drive:ddna_modulators"
Check-Marker "microbrain\neurons\reward_novelty_pulse_neuron.py" "affect:reward_state"
Check-Marker "microbrain\neurons\reward_novelty_pulse_neuron.py" "affect:salience_state"
Check-Marker "microbrain\neurons\reward_novelty_pulse_neuron.py" "pdna:reinforcement_model"
Check-Marker "microbrain\neurons\boredom_drive_neuron.py" "drive:boredom_relief"
Check-Marker "microbrain\neurons\boredom_drive_neuron.py" "boredom_relief_gain"
Check-Marker "microbrain\neurons\boredom_drive_neuron.py" "drive:ddna_modulators"
Check-Marker "microbrain\neurons\attention_controller.py" "affect:salience_state"
Check-Marker "microbrain\neurons\attention_controller.py" "drive:ddna_modulators"
Check-Marker "microbrain\neurons\thought_momentum_neuron.py" "drive:ddna_modulators"
Check-Marker "microbrain\neurons\thought_turn_arbitration_neuron.py" "drive:ddna_modulators"
Check-Marker "microbrain\neurons\thought_turn_arbitration_neuron.py" "pdna:wans"
Check-Marker "microbrain\ui\textual_bridge.py" "affect:reward_state"
Check-Marker "microbrain\ui\textual_bridge.py" "affect:salience_state"

$profilePath = Join-Path $MemDir "pdna_profile.json"
if (Test-Path $profilePath) {
  try {
    $pdna = Get-Content -Raw -Encoding UTF8 $profilePath | ConvertFrom-Json
    $sections = @("affect_model", "reinforcement_model", "drive_thresholds", "ddna_mutators", "wans")
    foreach ($s in $sections) {
      if ($pdna.PSObject.Properties.Name -contains $s) {
        Write-Host "PASS profile has $s at $profilePath" -ForegroundColor Green
      } else {
        Write-Host "FAIL profile missing $s at $profilePath" -ForegroundColor Red
        $missing = $true
      }
    }
  } catch {
    Write-Host "FAIL could not parse $profilePath : $($_.Exception.Message)" -ForegroundColor Red
    $missing = $true
  }
} else {
  Write-Host "WARN pdna_profile.json not found at $profilePath; pass -MemDir if it lives elsewhere." -ForegroundColor Yellow
}

if ($missing) {
  Write-Host "`nDDNA rewire check found missing pieces." -ForegroundColor Red
  exit 1
}

Write-Host "`nDDNA rewire check passed: core patch markers are present." -ForegroundColor Green
