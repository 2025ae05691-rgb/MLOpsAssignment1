$ErrorActionPreference = "Stop"

$baseUrl = "http://localhost:8000"

Write-Host "`n=== Phase 5: Endpoint Verification ===" -ForegroundColor Cyan

# Confirm the Kubernetes Service exists.
kubectl get service heart-disease-api
if ($LASTEXITCODE -ne 0) {
    throw "Service not found. Run deploy_phase5.ps1 first."
}

# Test the health endpoint.
Write-Host "`nTesting GET /health ..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod `
        -Uri "$baseUrl/health" `
        -Method Get `
        -TimeoutSec 15

    $health | ConvertTo-Json -Depth 10
    Write-Host "Health endpoint succeeded." -ForegroundColor Green
}
catch {
    Write-Warning "localhost:8000 is not responding."
    Write-Host "Check the Service address with: kubectl get service heart-disease-api"
    Write-Host "Check the Pod with: kubectl get pods"
    Write-Host "For temporary debugging, run: kubectl port-forward service/heart-disease-api 8000:8000"
    throw
}

# Test the OpenAPI schema.
Write-Host "`nTesting GET /openapi.json ..." -ForegroundColor Yellow
$schema = Invoke-RestMethod `
    -Uri "$baseUrl/openapi.json" `
    -Method Get `
    -TimeoutSec 15

Write-Host "Available API paths:" -ForegroundColor Green
$schema.paths.PSObject.Properties.Name

Write-Host "`nOpen the interactive API documentation at:"
Write-Host "$baseUrl/docs" -ForegroundColor Cyan

Write-Host "`nBasic endpoint verification completed." -ForegroundColor Green
Write-Host "Use /docs to submit a prediction request using your API's actual schema."