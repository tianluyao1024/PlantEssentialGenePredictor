$ErrorActionPreference = 'Stop'
$projectRoot = 'E:\PlantEssentialGenePredictor'
$pythonExe = 'D:\Python\Python311\python.exe'
$scriptPath = Join-Path $projectRoot 'scripts\webapp\cleanup_nar_jobs.py'
$taskName = 'PlantEGP-NAR-Cleanup'

if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Python not found: $pythonExe" }
if (-not (Test-Path -LiteralPath $scriptPath)) { throw "Cleanup script not found: $scriptPath" }

$action = New-ScheduledTaskAction -Execute $pythonExe -Argument ('"' + $scriptPath + '"') -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description 'Hourly cleanup of terminal PlantEGP NAR private jobs.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
