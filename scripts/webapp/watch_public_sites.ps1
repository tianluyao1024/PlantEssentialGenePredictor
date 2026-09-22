$ErrorActionPreference = 'Continue'

$root = 'E:\PlantEssentialGenePredictor'
$ensure = Join-Path $root 'scripts\webapp\ensure_public_sites.ps1'
$logDir = Join-Path $root 'webapp_data\logs'
$mutex = New-Object System.Threading.Mutex($false, 'Local\PlantResearchPublicSitesWatchdog')

if (-not $mutex.WaitOne(0)) { exit 0 }
try {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir 'public_sites_watchdog.log'
    while ($true) {
        try {
            & $ensure
            "$(Get-Date -Format s) checked public-site processes" | Add-Content -LiteralPath $log
        } catch {
            "$(Get-Date -Format s) watchdog error: $($_.Exception.Message)" | Add-Content -LiteralPath $log
        }
        Start-Sleep -Seconds 300
    }
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
