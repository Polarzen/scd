param(
    [string]$PythonVersion = "3.13",
    [switch]$UseLock
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function New-ProjectVenv {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py "-$PythonVersion" -m venv ".venv"
            return
        }
        catch {
            Write-Warning "py -$PythonVersion failed. Falling back to py -3."
            & py -3 -m venv ".venv"
            return
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv ".venv"
        return
    }

    throw "No Python launcher found. Install Python 3.13+ and ensure py or python is in PATH."
}

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
    Write-Host "[INFO] Creating .venv ..."
    New-ProjectVenv
}
else {
    Write-Host "[INFO] Reusing existing .venv"
}

$pyExe = ".\\.venv\\Scripts\\python.exe"
& $pyExe -m pip install --upgrade pip

$reqFile = if ($UseLock -and (Test-Path "requirements-lock.txt")) {
    "requirements-lock.txt"
}
else {
    "requirements.txt"
}

Write-Host "[INFO] Installing dependencies from $reqFile ..."
& $pyExe -m pip install -r $reqFile

Write-Host ""
Write-Host "[OK] Environment is ready."
Write-Host "[RUN] .\\.venv\\Scripts\\python.exe <your_script.py>"
Write-Host "[RUN] .\\run_with_venv.ps1 scd_logistic_model.py"
Write-Host "[RUN] .\\run_with_venv.ps1 visualize_dynamic_risk.py"
