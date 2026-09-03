param(
    [string]$BenchArgs = "--all",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "=== GA102-ROM bootstrap ==="

function Require-Command([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Missing required command: $Name" }
    return $cmd.Source
}

$cmake = Require-Command "cmake"
$nvcc  = Require-Command "nvcc"
$nvsmi = Require-Command "nvidia-smi"

Write-Host "cmake: $cmake"
Write-Host "nvcc : $nvcc"

Write-Host "`nDetected NVIDIA GPU(s):"
& $nvsmi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed" }

$gpuInfo = (& $nvsmi --query-gpu=name,compute_cap --format=csv,noheader | Out-String)
if ($gpuInfo -notmatch "RTX 3080") {
    Write-Warning "No RTX 3080 detected. GA102-ROM is tuned for RTX 3080 / SM86; the binary will reject incompatible devices unless --allow-other-gpu is supplied."
}

$build = Join-Path $root "build"
if ($Clean -and (Test-Path $build)) {
    Remove-Item -Recurse -Force $build
}

cmake -S . -B $build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

cmake --build $build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

$exeCandidates = @(
    (Join-Path $build "Release\ga102-rom.exe"),
    (Join-Path $build "ga102-rom.exe")
)
$exe = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { throw "Built executable not found" }

Write-Host "`n=== Running GA102-ROM ==="
Write-Host "$exe $BenchArgs"
$argv = @()
if ($BenchArgs.Trim().Length -gt 0) {
    $argv = [System.Management.Automation.PSParser]::Tokenize($BenchArgs, [ref]$null) |
        Where-Object { $_.Type -eq 'CommandArgument' -or $_.Type -eq 'Command' } |
        ForEach-Object { $_.Content }
}
& $exe @argv
exit $LASTEXITCODE
