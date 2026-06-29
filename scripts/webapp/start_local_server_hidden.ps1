$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$python = "D:\Python\Python311\python.exe"
$logDir = Join-Path $root "webapp_data\logs"
$pidFile = Join-Path $root "webapp_data\streamlit_server.pid"
$stdout = Join-Path $logDir "streamlit_stdout.log"
$stderr = Join-Path $logDir "streamlit_stderr.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (Test-Path $pidFile) {
    $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        Write-Host "Server already appears to be running with PID $oldPid"
        Write-Host "Open: http://192.168.1.100:8501"
        exit 0
    }
}

$arguments = @(
    "-m", "streamlit", "run", "webapp/app.py",
    "--server.address", "0.0.0.0",
    "--server.port", "8501",
    "--server.headless", "true",
    "--server.maxUploadSize", "4096"
)

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $pidFile -Value $process.Id
Write-Host "Started PlantEssentialGenePredictor with PID $($process.Id)"
Write-Host "Local URL: http://localhost:8501"
Write-Host "LAN URL:   http://192.168.1.100:8501"
