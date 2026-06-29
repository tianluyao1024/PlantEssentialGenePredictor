$ErrorActionPreference = "SilentlyContinue"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$pidFile = Join-Path $root "webapp_data\streamlit_server.pid"

if (Test-Path $pidFile) {
    $pidValue = Get-Content $pidFile
    $process = Get-Process -Id $pidValue
    if ($process) {
        Stop-Process -Id $pidValue -Force
        Write-Host "Stopped PlantEssentialGenePredictor server PID $pidValue"
    }
    Remove-Item $pidFile -Force
} else {
    Write-Host "No PID file found. If needed, stop python/streamlit manually."
}
