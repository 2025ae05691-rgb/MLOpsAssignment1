python3 -m venv .venv
.source .venv/bin/activate
pwsh ./run_phase1.ps1

python ./src/train_with_mlflow.py

