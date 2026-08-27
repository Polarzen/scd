# Portable Python environment guide

Do not copy .venv between computers.
Always recreate .venv on the target computer from requirements files.

## First time setup on a new computer

1. Open PowerShell in this project folder.
2. Run: .\\setup_venv.ps1

If you want fully locked versions, run:

.\\setup_venv.ps1 -UseLock

## Run scripts

.\\run_with_venv.ps1 心电图分析.py
.\\run_with_venv.ps1 scd_logistic_model.py
.\\run_with_venv.ps1 visualize_dynamic_risk.py

## Notes

- requirements.txt contains core direct dependencies.
- requirements-lock.txt contains the full tested lock list.
- .vscode/settings.json points VS Code to .venv automatically.
- For scd_logistic_model.py, place scd_dataset.csv in the project root,
  or edit DATA_PATH in the script.
