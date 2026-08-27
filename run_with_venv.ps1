param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ScriptPath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs,
    [switch]$UseLock
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
    Write-Host "[INFO] .venv not found. Bootstrapping environment first ..."
    & ".\\setup_venv.ps1" -UseLock:$UseLock
}

if (-not (Test-Path $ScriptPath)) {
    throw "Script not found: $ScriptPath"
}

$pyExe = ".\\.venv\\Scripts\\python.exe"
Write-Host "[INFO] Running script: $ScriptPath"
& $pyExe $ScriptPath @ScriptArgs
