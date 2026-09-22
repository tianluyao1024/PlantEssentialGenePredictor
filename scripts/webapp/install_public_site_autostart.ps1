$ErrorActionPreference = 'Stop'

$projectRoot = 'E:\PlantEssentialGenePredictor'
$watchdog = Join-Path $projectRoot 'scripts\webapp\ensure_public_sites.ps1'
$powershell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

if (-not (Test-Path -LiteralPath $watchdog)) {
    throw "Missing watchdog script: $watchdog"
}

$action = New-ScheduledTaskAction -Execute $powershell -Argument (
    '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $watchdog + '"'
)
$startup = New-ScheduledTaskTrigger -AtStartup
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName 'PlantResearch-Websites-EnsureRunning' `
    -Description 'At startup and every five minutes, starts only missing PlantEGP, ProtoEssAtlas, and their existing NATAPP processes.' `
    -Action $action -Trigger @($startup, $repeat) -Settings $settings -Principal $principal -Force -ErrorAction Stop | Out-Null

Start-ScheduledTask -TaskName 'PlantResearch-Websites-EnsureRunning' -ErrorAction Stop
Write-Output 'Installed and started PlantResearch-Websites-EnsureRunning.'
