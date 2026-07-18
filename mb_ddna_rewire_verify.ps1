# Run from the MicroBrain repo root, usually: C:\aiproj\microbrain
# Checks that the DDNA metabolic rewire patch is present without requiring exact file hashes.

$ErrorActionPreference = 'Stop'

function Pass($msg) { Write-Host "PASS $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "FAIL $msg" -ForegroundColor Red; $script:failed = $true }
function Warn($msg) { Write-Host "WARN $msg" -ForegroundColor Yellow }

$failed = $false

$requiredFiles = @(
  'microbrain\pdna\access.py',
  'microbrain\pdna\core.py',
  'microbrain\hormone.py',
  'microbrain\mind.py',
  'microbrain\neurons\pdna_state_neuron.py',
  'microbrain\neurons\reward_novelty_pulse_neuron.py',
  'microbrain\neurons\boredom_drive_neuron.py',
  'microbrain\neurons\attention_controller.py',
  'microbrain\neurons\thought_momentum_neuron.py',
  'microbrain\neurons\thought_turn_arbitration_neuron.py',
  'microbrain\ui\textual_bridge.py',
  'tests\test_pdna_ddna_profile_v2.py',
  'docs\demi_ddna_metabolic_rewire_v1_20260703.md'
)

foreach ($file in $requiredFiles) {
  if (Test-Path $file) { Pass "found $file" } else { Fail "missing $file" }
}

$markerChecks = @(
  @{Path='microbrain\pdna\core.py'; Patterns=@('extra_sections','ddna_mutators','reinforcement_model')},
  @{Path='microbrain\pdna\access.py'; Patterns=@('drive:ddna_modulators','pdna:affect_model','pdna:wans')},
  @{Path='microbrain\mind.py'; Patterns=@('publish_pdna_runtime_profile','drive:ddna_modulators')},
  @{Path='microbrain\neurons\reward_novelty_pulse_neuron.py'; Patterns=@('affect:reward_state','affect:salience_state','pdna:reinforcement_model')},
  @{Path='microbrain\neurons\boredom_drive_neuron.py'; Patterns=@('drive:boredom_relief','boredom_relief_gain','drive:ddna_modulators')},
  @{Path='microbrain\neurons\attention_controller.py'; Patterns=@('affect:salience_state','drive:ddna_modulators')},
  @{Path='microbrain\neurons\thought_momentum_neuron.py'; Patterns=@('drive:ddna_modulators')},
  @{Path='microbrain\neurons\thought_turn_arbitration_neuron.py'; Patterns=@('drive:ddna_modulators','pdna:wans')},
  @{Path='microbrain\ui\textual_bridge.py'; Patterns=@('affect:reward_state','affect:salience_state')}
)

foreach ($check in $markerChecks) {
  if (-not (Test-Path $check.Path)) { continue }
  $text = Get-Content $check.Path -Raw
  foreach ($pattern in $check.Patterns) {
    if ($text.Contains($pattern)) { Pass "$($check.Path) contains $pattern" } else { Fail "$($check.Path) missing marker $pattern" }
  }
}

if (Test-Path 'pdna_profile.json') {
  try {
    $pdna = Get-Content 'pdna_profile.json' -Raw | ConvertFrom-Json
    foreach ($field in @('affect_model','reinforcement_model','drive_thresholds','ddna_mutators','wans')) {
      if ($pdna.PSObject.Properties.Name -contains $field) { Pass "pdna_profile.json has $field" } else { Warn "pdna_profile.json missing $field" }
    }
  } catch {
    Fail "pdna_profile.json could not be parsed as JSON: $($_.Exception.Message)"
  }
} else {
  Warn 'pdna_profile.json not found in this folder; profile may live elsewhere.'
}

Write-Host ''
if ($failed) {
  Write-Host 'DDNA rewire check found missing pieces.' -ForegroundColor Red
  exit 1
} else {
  Write-Host 'DDNA rewire check passed: core patch markers are present.' -ForegroundColor Green
  exit 0
}
