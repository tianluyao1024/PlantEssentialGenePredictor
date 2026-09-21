$ErrorActionPreference = 'Stop'
$projectRoot = 'E:\PlantEssentialGenePredictor'
$pythonExe = 'D:\Python\Python311\python.exe'
$logDir = Join-Path $projectRoot 'webapp_data\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$listener = Get-NetTCPConnection -State Listen -LocalPort 8501 -ErrorAction SilentlyContinue
if ($listener) { throw 'Port 8501 is already occupied; inspect its owner before switching.' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$process = Start-Process -FilePath $pythonExe -ArgumentList @('-m','streamlit','run','webapp/nar_app.py','--server.address','127.0.0.1','--server.port','8501','--server.headless','true','--server.maxUploadSize','50','--browser.gatherUsageStats','false') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir "nar_public_$stamp.out.log") -RedirectStandardError (Join-Path $logDir "nar_public_$stamp.err.log")
$process.Id | Set-Content (Join-Path $projectRoot 'webapp_data\nar_public.pid')
Write-Output "Started PlantEGP public beta, PID $($process.Id), local HTTP 127.0.0.1:8501. TLS remains managed by NATAPP."
