param(
    [string]$V2Args = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Reuse the main bootstrap so CUDA/VS selection stays in one place.
# Use a hashtable splat here. Passing '-BenchArgs' inside an array causes
# PowerShell to bind it positionally as the value of BenchArgs.
$bootstrap = @{
    BenchArgs = "--info"
}
if ($Clean) {
    $bootstrap.Clean = $true
}

& (Join-Path $root "run.ps1") @bootstrap
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exeCandidates = @(
    (Join-Path $root "build\Release\ga102-rom-v2.exe"),
    (Join-Path $root "build\ga102-rom-v2.exe")
)
$exe = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { throw "ga102-rom-v2.exe was not built" }

$argv = @()
if ($V2Args.Trim().Length -gt 0) {
    $argv = [regex]::Matches($V2Args, '(?:[^\s\"]+|\"[^\"]*\")+') | ForEach-Object {
        $_.Value.Trim('"')
    }
}

Write-Host "`n=== Running GA102-ROM V2 ==="
Write-Host ("{0} {1}" -f $exe, ($argv -join ' '))
& $exe @argv
exit $LASTEXITCODE
