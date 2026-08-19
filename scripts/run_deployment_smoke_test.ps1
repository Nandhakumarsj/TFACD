param(
  [string]$CertDir = "artifacts/certs",
  [int]$NumSuperNodes = 2,
  [string]$SuperlinkConnectionName = "local-tls",
  [int]$ControlApiPort = 9093,
  [int]$FleetApiPort = 9092,
  [int]$ReadinessTimeoutSeconds = 30
)

# Real multi-process deployment-mode verification: server-authenticated TLS +
# SuperNode public-key auth (NOT client-cert mTLS - see security/certificates.py)
# over actual network sockets, not Flower's local-simulation engine used by
# every other run in this repo. $ErrorActionPreference alone does not kill
# child processes on failure, hence the explicit trap + Stop-AllProcesses below.

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
Set-Location $repoRoot
$venvScripts = Join-Path $repoRoot ".venv/Scripts"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = "$venvScripts;$env:PATH"

$startedProcesses = New-Object System.Collections.ArrayList

function Stop-AllProcesses {
    foreach ($p in $startedProcesses) {
        if ($p -and -not $p.HasExited) {
            Write-Host "Stopping process PID $($p.Id) ($($p.ProcessName))"
            try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
}

trap {
    Write-Host "ERROR: $_"
    Stop-AllProcesses
    exit 1
}

# 1. Generate certs if missing
$caPath = Join-Path $repoRoot "$CertDir/ca.pem"
if (-not (Test-Path $caPath)) {
    Write-Host "Generating certs..."
    & "$venvScripts/python.exe" scripts/generate_deployment_certs.py --output-dir $CertDir --num-supernodes $NumSuperNodes
    if ($LASTEXITCODE -ne 0) { throw "cert generation failed" }
}
$caPathAbs = (Resolve-Path $caPath).Path
$serverCertAbs = (Resolve-Path (Join-Path $repoRoot "$CertDir/server.pem")).Path
$serverKeyAbs = (Resolve-Path (Join-Path $repoRoot "$CertDir/server_key.pem")).Path

# 2. Ensure a [superlink.<name>] connection entry exists in ~/.flwr/config.toml -
# there is no `flwr config` subcommand to create one, only `list`, so this is a
# direct, additive (never overwritten if already present) file append.
$flwrConfigPath = Join-Path $env:USERPROFILE ".flwr/config.toml"
$configContent = if (Test-Path $flwrConfigPath) { Get-Content $flwrConfigPath -Raw } else { "" }
$sectionHeader = "[superlink.$SuperlinkConnectionName]"
if ($configContent -notmatch [regex]::Escape($sectionHeader)) {
    Write-Host "Registering SuperLink connection '$SuperlinkConnectionName' in $flwrConfigPath"
    $caTomlPath = $caPathAbs -replace '\\', '/'
    $entry = "`n$sectionHeader`naddress = `"127.0.0.1:$ControlApiPort`"`nroot-certificates = `"$caTomlPath`"`n"
    Add-Content -Path $flwrConfigPath -Value $entry
}

# 3. Start SuperLink (Control API on $ControlApiPort, Fleet API on $FleetApiPort - defaults)
Write-Host "Starting flower-superlink..."
$superlinkArgs = @("--ssl-certfile", $serverCertAbs, "--ssl-keyfile", $serverKeyAbs, "--ssl-ca-certfile", $caPathAbs, "--enable-supernode-auth")
$superlinkProc = Start-Process -FilePath "$venvScripts/flower-superlink.exe" -ArgumentList $superlinkArgs -PassThru -NoNewWindow `
    -RedirectStandardOutput "$CertDir/superlink.out.log" -RedirectStandardError "$CertDir/superlink.err.log"
[void]$startedProcesses.Add($superlinkProc)

# 4. Readiness check - poll the Control API port, don't sleep-and-hope
Write-Host "Waiting for Control API on port $ControlApiPort..."
$ready = $false
$deadline = (Get-Date).AddSeconds($ReadinessTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", $ControlApiPort)
        if ($tcp.Connected) { $ready = $true; $tcp.Close(); break }
        $tcp.Close()
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $ready) { throw "SuperLink Control API did not become ready within $ReadinessTimeoutSeconds seconds" }
Write-Host "Control API ready."

# 5. Register each SuperNode against the running SuperLink -> node_id
$nodeIds = New-Object System.Collections.ArrayList
for ($i = 0; $i -lt $NumSuperNodes; $i++) {
    $pubKeyPath = (Resolve-Path "$CertDir/supernodes/supernode-$i`_auth.pub").Path
    Write-Host "Registering supernode-$i..."
    $registerOutput = & "$venvScripts/flwr.exe" supernode register $pubKeyPath $SuperlinkConnectionName --format json 2>&1 | Out-String
    Write-Host $registerOutput
    $parsed = $registerOutput | ConvertFrom-Json
    [void]$nodeIds.Add($parsed.node_id)
}

# 6. Start each SuperNode process (Fleet API, a different port from Control API).
# Each SuperNode also runs its own local ClientAppIo API (default 0.0.0.0:9094) -
# running >1 SuperNode on one machine needs a distinct port per node, confirmed
# empirically: without this, the 2nd node fails silently with a gRPC bind error
# and only 1/2 nodes ever connect, no matter how long flwr run waits.
for ($i = 0; $i -lt $NumSuperNodes; $i++) {
    $privKeyPath = (Resolve-Path "$CertDir/supernodes/supernode-$i`_auth").Path
    $clientAppIoPort = 9094 + $i
    Write-Host "Starting flower-supernode $i (ClientAppIo on port $clientAppIoPort)..."
    $supernodeArgs = @(
        "--auth-supernode-private-key", $privKeyPath, "--root-certificates", $caPathAbs,
        "--superlink", "127.0.0.1:$FleetApiPort", "--node-config", "partition-id=$i",
        "--clientappio-api-address", "127.0.0.1:$clientAppIoPort"
    )
    $proc = Start-Process -FilePath "$venvScripts/flower-supernode.exe" -ArgumentList $supernodeArgs -PassThru -NoNewWindow `
        -RedirectStandardOutput "$CertDir/supernode-$i.out.log" -RedirectStandardError "$CertDir/supernode-$i.err.log"
    [void]$startedProcesses.Add($proc)
}
Start-Sleep -Seconds 5  # let supernodes finish connecting before submitting a run

# 7. `flwr federation create`/`add-supernode` are for Flower's multi-tenant/cloud
# control plane ("SuperLink does not support federation management" on a plain
# on-prem SuperLink, confirmed empirically). `--federation` is a DIFFERENT thing
# entirely - a hosted-cloud federation id (`@account/name`), not a connection
# selector - confirmed by reading flwr/cli/run/run.py: check_federation_format
# only runs when --federation is explicitly passed; every prior run this session
# omitted it and relied on `default = "local-simulation"` in ~/.flwr/config.toml.
# The SuperLink connection to use is SUPERLINK, a POSITIONAL argument
# (`flwr run [APP] [SUPERLINK]`), confirmed against `flwr run --help`.
Write-Host "Submitting flwr run over TLS against '$SuperlinkConnectionName'..."
& "$venvScripts/flwr.exe" run . $SuperlinkConnectionName --stream --run-config "num-server-rounds=1"
if ($LASTEXITCODE -ne 0) { throw "flwr run over deployment-mode TLS failed" }

Write-Host "Deployment smoke test completed successfully."
Stop-AllProcesses
