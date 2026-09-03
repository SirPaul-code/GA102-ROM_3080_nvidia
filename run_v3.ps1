param(
    [string]$V3Args = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Reuse the main bootstrap for CUDA / Visual Studio selection and build.
$bootstrap = @{
    BenchArgs = "--info"
}
if ($Clean) {
    $bootstrap.Clean = $true
}

& (Join-Path $root "run.ps1") @bootstrap
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exeCandidates = @(
    (Join-Path $root "build\Release\ga102-rom-v3.exe"),
    (Join-Path $root "build\ga102-rom-v3.exe")
)
$exe = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { throw "ga102-rom-v3.exe was not built" }

$argv = @()
if ($V3Args.Trim().Length -gt 0) {
    $argv = [regex]::Matches($V3Args, '(?:[^\s\"]+|\"[^\"]*\")+') | ForEach-Object {
        $_.Value.Trim('"')
    }
}

Write-Host "`n=== Running GA102-ROM V3 ==="
Write-Host ("{0} {1}" -f $exe, ($argv -join ' '))
& $exe @argv
exit $LASTEXITCODE
