param(
    [int]$Port = 8501,
    [switch]$NoBrowser,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$StreamlitExe = Join-Path $VenvDir "Scripts\streamlit.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating Python virtual environment in $VenvDir"
    if ($PythonPath -and (Test-Path $PythonPath)) {
        & $PythonPath -m venv $VenvDir
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.11 -m venv $VenvDir
    } else {
        python -m venv $VenvDir
    }
}

Write-Host "Upgrading pip"
& $PythonExe -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt"
& $PythonExe -m pip install -r requirements.txt

Write-Host "Checking required model files"
$RequiredFiles = @(
    "models\arabidopsis_single_strict2601_common6751\selected_model_and_manifest.joblib",
    "models\rice_single_strict399_Tos17N4_common6751\model.joblib",
    "models\joint_arabidopsis_rice_common6751\model.joblib",
    "models\deployable_feature_profiles\sequence_plm_go_ppi\model.joblib",
    "data\processed_features\common6751_feature_names.tsv"
)
foreach ($File in $RequiredFiles) {
    if (-not (Test-Path (Join-Path $ProjectRoot $File))) {
        throw "Missing required file: $File"
    }
}

$Args = @("run", "webapp\app.py", "--server.port", "$Port", "--server.address", "0.0.0.0")
if ($NoBrowser) {
    $Args += @("--server.headless", "true")
}

Write-Host "Starting Streamlit on http://localhost:$Port"
& $StreamlitExe @Args
