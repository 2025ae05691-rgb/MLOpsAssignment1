
$ErrorActionPreference = "Stop"

$steps = @(
    "src\download_data.py",
    "src\eda.py",
    "src\train.py",
    "src\verify_model.py"
)

foreach ($step in $steps) {
    Write-Host "`n========================================"
    Write-Host "Running $step"
    Write-Host "========================================"

    python $step

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed at $step. Stopping pipeline."
        exit 1
    }
}

Write-Host "`nPhase 1 completed successfully!"
