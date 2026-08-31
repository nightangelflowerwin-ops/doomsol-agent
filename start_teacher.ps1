[CmdletBinding()]
param(
    [int]$Port = 8150,
    [string]$Output,
    [ValidateRange(1, 21600)][int]$Seconds = 300,
    [ValidateRange(1, 2147483647)][int]$Seed = 1701,
    [ValidateRange(1, 35)][int]$CaptureEvery = 5,
    [ValidateRange(1, 1000)][int]$ChunkSize = 50,
    [switch]$NoBrowser,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$build = Join-Path $repo 'build-teacher3'
$server = Join-Path $repo 'wasm\teacher_server.py'
$python = Join-Path $repo '..\tools\emsdk\python\3.13.3_64bit\python.exe'
$required = @('index.html', 'index.js', 'index.wasm', 'index.data')

foreach ($name in $required) {
    $path = Join-Path $build $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Teacher build is incomplete: missing $path"
    }
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Bundled EMSDK Python was not found: $python"
}

if (-not $Output) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $Output = Join-Path $repo "..\doomv9\models\dwasm_teacher\teacher-$stamp.jsonl"
}
$Output = [System.IO.Path]::GetFullPath($Output)
$outputDirectory = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$existing = $null
try {
    $existing = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" `
        -TimeoutSec 1 -ErrorAction Stop
} catch {}
if ($existing) {
    throw "Port $Port is already in use. Run again with -Port 8151 or stop the existing server."
}

$serverToken = [guid]::NewGuid().ToString('N')

$arguments = @(
    $server,
    '--directory', $build,
    '--output', $Output,
    '--port', $Port,
    '--token', $serverToken
)
if ($Resume) { $arguments += '--resume' }

$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru
$url = "http://127.0.0.1:$Port/index.html?seed=$Seed&seconds=$Seconds&captureEvery=$CaptureEvery&chunkSize=$ChunkSize#agent-teacher"

$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if ($process.HasExited) {
        throw "Teacher server exited during startup with code $($process.ExitCode)."
    }
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" `
            -TimeoutSec 1
        if ($response.ok -and $response.token -eq $serverToken -and
            $response.pid -eq $process.Id) { $ready = $true; break }
    } catch {
        Start-Sleep -Milliseconds 100
    }
}
if (-not $ready) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    throw "Teacher server did not become ready on port $Port."
}

Write-Host "Dwasm teacher server ready"
Write-Host "URL: $url"
Write-Host "Recording: $Output"
Write-Host "Server PID: $($process.Id)"

if (-not $NoBrowser) {
    Start-Process $url
}

[pscustomobject]@{
    Url = $url
    Output = $Output
    ProcessId = $process.Id
    Seed = $Seed
    Seconds = $Seconds
    CaptureEvery = $CaptureEvery
}
