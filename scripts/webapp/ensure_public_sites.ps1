$ErrorActionPreference = 'Stop'

# This watchdog only starts missing processes.  It never stops or reconfigures
# either site, NATAPP, DNS, certificates, or existing tunnels.
$plantRoot = 'E:\PlantEssentialGenePredictor'
$plantStarter = Join-Path $plantRoot 'scripts\webapp\start_nar_public.ps1'
$protoRoot = 'C:\Users\tly\Desktop\植物\水稻日本\plasmodium_atlas_web'
$protoStarter = Join-Path $protoRoot 'start_site_background.ps1'
$protoTunnelStarter = 'E:\natapp_protoessatlas\start_protoessatlas_tunnel.ps1'
$protoTunnelConfig = 'E:\natapp_protoessatlas\config.ini'
$powershell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

function Test-ListeningPort([int]$Port) {
    return [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

if (-not (Test-ListeningPort 8501)) {
    & $plantStarter
}

if (-not (Test-ListeningPort 8511)) {
    Start-Process -FilePath $powershell -ArgumentList @(
        '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
        '-File', $protoStarter
    ) -WindowStyle Hidden
}

$natappService = Get-Service -Name 'natapp' -ErrorAction SilentlyContinue
if ($natappService -and $natappService.Status -ne 'Running') {
    try {
        Start-Service -Name 'natapp'
    } catch {
        # A normal user session cannot always start a Windows service.  The
        # automatic NATAPP service will retry at boot; still restore the
        # independent ProtoEss tunnel below whenever its process is absent.
    }
}

$protoTunnelRunning = Get-CimInstance Win32_Process -Filter "Name = 'natapp.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*${protoTunnelConfig}*" }
if (-not $protoTunnelRunning) {
    Start-Process -FilePath $powershell -ArgumentList @(
        '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
        '-File', $protoTunnelStarter
    ) -WindowStyle Hidden
}
