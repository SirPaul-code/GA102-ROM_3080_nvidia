param(
    [string]$BenchArgs = "--all",
    [switch]$Clean,
    [switch]$NoAutoInstall
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

function Get-CudaVersion([string]$NvccPath) {
    $text = (& $NvccPath --version 2>&1 | Out-String)
    if ($text -match 'release\s+(\d+)\.(\d+)') {
        return [version]("{0}.{1}" -f $Matches[1], $Matches[2])
    }
    return $null
}

function Find-CudaToolkits {
    $items = @()

    $cudaRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $candidate = Join-Path $_.FullName "bin\nvcc.exe"
            if (Test-Path $candidate) {
                $v = Get-CudaVersion $candidate
                if ($v) {
                    $items += [pscustomobject]@{
                        Version = $v
                        Nvcc = $candidate
                        Root = $_.FullName
                    }
                }
            }
        }
    }

    $pathNvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($pathNvcc) {
        $candidate = $pathNvcc.Source
        $v = Get-CudaVersion $candidate
        if ($v -and -not ($items | Where-Object { $_.Nvcc -eq $candidate })) {
            $items += [pscustomobject]@{
                Version = $v
                Nvcc = $candidate
                Root = Split-Path -Parent (Split-Path -Parent $candidate)
            }
        }
    }

    return @($items | Sort-Object Version -Descending)
}

function Get-VisualStudioInfo {
    $vswhereCandidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"),
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    )
    $vswhere = $vswhereCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $vswhere) {
        throw "Visual Studio was detected by CMake previously, but vswhere.exe is missing. Install Visual Studio/Build Tools with Desktop development with C++."
    }

    $installPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
    $installVersion = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion | Select-Object -First 1)
    if (-not $installPath -or -not $installVersion) {
        throw "No Visual Studio installation with the MSVC x64 toolchain was found. Install Desktop development with C++."
    }

    $major = [int]($installVersion.Split('.')[0])
    $generator = switch ($major) {
        18 { "Visual Studio 18 2026" }
        17 { "Visual Studio 17 2022" }
        16 { "Visual Studio 16 2019" }
        default { throw "Unsupported Visual Studio major version $major ($installVersion)." }
    }

    return [pscustomobject]@{
        Major = $major
        Version = $installVersion
        Path = $installPath
        Generator = $generator
    }
}

function Get-MinCudaForVS([int]$VsMajor) {
    switch ($VsMajor) {
        18 { return [version]"13.2" }
        17 { return [version]"11.8" }
        16 { return [version]"11.2" }
        default { throw "No CUDA compatibility rule for Visual Studio $VsMajor." }
    }
}

# Split a command-line fragment without PowerShell interpreting --foo as operators.
# Supports ordinary whitespace-separated args plus single/double-quoted values.
function Split-BenchArgs([string]$Text) {
    if ([string]::IsNullOrWhiteSpace($Text)) { return @() }

    $matches = [regex]::Matches($Text, '"(?:\\.|[^"\\])*"|''(?:''''|[^''])*''|\S+')
    $result = @()
    foreach ($m in $matches) {
        $v = $m.Value
        if ($v.Length -ge 2 -and (($v[0] -eq '"' -and $v[$v.Length - 1] -eq '"') -or ($v[0] -eq "'" -and $v[$v.Length - 1] -eq "'"))) {
            $v = $v.Substring(1, $v.Length - 2)
        }
        $result += $v
    }
    return $result
}

$cmake = Require-Command "cmake"
$nvsmi = Require-Command "nvidia-smi"
$vs = Get-VisualStudioInfo
$minCuda = Get-MinCudaForVS $vs.Major

Write-Host "cmake : $cmake"
Write-Host "Visual Studio: $($vs.Version)"
Write-Host "Generator    : $($vs.Generator)"
Write-Host "Minimum CUDA : $minCuda"

Write-Host "`nDetected NVIDIA GPU(s):"
& $nvsmi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed" }

$gpuInfo = (& $nvsmi --query-gpu=name,compute_cap --format=csv,noheader | Out-String)
if ($gpuInfo -notmatch "RTX 3080") {
    Write-Warning "No RTX 3080 detected. GA102-ROM is tuned for RTX 3080 / SM86; the binary will reject incompatible devices unless --allow-other-gpu is supplied."
}

$toolkits = @(Find-CudaToolkits)
if ($toolkits.Count -gt 0) {
    Write-Host "`nInstalled CUDA toolkits:"
    foreach ($tk in $toolkits) {
        $compat = if ($tk.Version -ge $minCuda) { "compatible" } else { "too old for VS $($vs.Major)" }
        Write-Host ("  CUDA {0,-6} {1} [{2}]" -f $tk.Version, $tk.Root, $compat)
    }
}

$cuda = $toolkits | Where-Object { $_.Version -ge $minCuda } | Select-Object -First 1

if (-not $cuda) {
    Write-Host ""
    Write-Warning "No CUDA Toolkit compatible with Visual Studio $($vs.Major) was found."
    Write-Host "Your old CUDA 11.2 cannot compile with Visual Studio 2026/MSVC 19.5x. GA102 itself is fine."

    if ($NoAutoInstall) {
        throw "Install CUDA Toolkit 13.2+ (for VS 2026), then rerun .\run.ps1."
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Install CUDA Toolkit 13.2+ from NVIDIA. WinGet is unavailable, so GA102-ROM cannot bootstrap it automatically."
    }

    Write-Host "`nInstalling/upgrading NVIDIA CUDA Toolkit through WinGet..."
    Write-Host "This may request administrator elevation. CUDA 13.2+ is required for Visual Studio 2026."
    & $winget.Source install --exact --id Nvidia.CUDA --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        & $winget.Source upgrade --exact --id Nvidia.CUDA --accept-source-agreements --accept-package-agreements
    }

    $toolkits = @(Find-CudaToolkits)
    $cuda = $toolkits | Where-Object { $_.Version -ge $minCuda } | Select-Object -First 1
    if (-not $cuda) {
        throw "CUDA installation finished but no compatible Toolkit was found yet. If the installer requested a reboot, reboot once and rerun .\run.ps1. Otherwise install CUDA 13.2+ manually."
    }
}

$nvcc = $cuda.Nvcc
$cudaRoot = $cuda.Root
$env:CUDA_PATH = $cudaRoot
$env:PATH = (Join-Path $cudaRoot "bin") + ";" + $env:PATH

Write-Host "`nSelected CUDA: $($cuda.Version)"
Write-Host "nvcc         : $nvcc"
Write-Host "CUDA_PATH    : $cudaRoot"

$build = Join-Path $root "build"
if ($Clean -and (Test-Path $build)) {
    Remove-Item -Recurse -Force $build
}

$cache = Join-Path $build "CMakeCache.txt"
if (Test-Path $cache) {
    $cacheText = Get-Content $cache -Raw -ErrorAction SilentlyContinue
    if ($cacheText -and (($cacheText -notmatch [regex]::Escape($cudaRoot)) -or ($cacheText -notmatch [regex]::Escape($vs.Generator)))) {
        Write-Host "Removing stale CMake cache from an older CUDA/Visual Studio pairing..."
        Remove-Item -Recurse -Force $build
    }
}

$toolset = "cuda=$cudaRoot"
& $cmake -S . -B $build -G $vs.Generator -T $toolset -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER="$nvcc"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with CUDA $($cuda.Version) + $($vs.Generator). Run .\run.ps1 -Clean once if this directory was configured with an older toolchain."
}

& $cmake --build $build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

$exeCandidates = @(
    (Join-Path $build "Release\ga102-rom.exe"),
    (Join-Path $build "ga102-rom.exe")
)
$exe = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) { throw "Built executable not found" }

Write-Host "`n=== Running GA102-ROM ==="
$argv = @(Split-BenchArgs $BenchArgs)
Write-Host ("{0} {1}" -f $exe, ($argv -join ' '))
& $exe @argv
exit $LASTEXITCODE
