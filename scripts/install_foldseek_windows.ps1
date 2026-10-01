[CmdletBinding()]
param(
    [ValidateSet("gpu", "cpu")]
    [string]$Device = "gpu",
    [string]$Distribution = "Ubuntu",
    [string]$CondaEnvironment = "ab-ag-benchmark-gpu",
    [string]$CudaArchitectures = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL2 is required. Install WSL with Ubuntu before running this script."
}

$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslRepository = (& wsl.exe -d $Distribution wslpath -a $repository).Trim()
if ($LASTEXITCODE -ne 0 -or -not $wslRepository) {
    throw "Could not translate the repository path for WSL distribution '$Distribution'."
}
if ($wslRepository.Contains("'")) {
    throw "Repository paths containing a single quote are not supported by this launcher."
}

if ($Device -eq "gpu") {
    & wsl.exe -d $Distribution nvidia-smi -L
    if ($LASTEXITCODE -ne 0) {
        throw "The NVIDIA GPU is not visible inside WSL2. Update the Windows NVIDIA driver and verify WSL GPU access."
    }
}

$architecturePrefix = ""
if ($CudaArchitectures) {
    if ($CudaArchitectures.Contains("'")) {
        throw "CudaArchitectures must not contain a single quote."
    }
    $architecturePrefix = "env FOLDSEEK_CUDA_ARCHITECTURES='$CudaArchitectures' "
}

$linuxCommand = "cd '$wslRepository' && conda run --no-capture-output -n '$CondaEnvironment' ${architecturePrefix}bash scripts/install_foldseek_linux.sh --$Device"
& wsl.exe -d $Distribution bash -lic $linuxCommand
if ($LASTEXITCODE -ne 0) {
    throw "Foldseek installation failed inside WSL2."
}

Write-Host "Foldseek installation completed in WSL distribution '$Distribution' using Conda environment '$CondaEnvironment'."
