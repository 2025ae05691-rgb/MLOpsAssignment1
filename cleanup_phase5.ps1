$ErrorActionPreference = "Stop"

Write-Host "Removing Phase 5 Kubernetes resources..." -ForegroundColor Yellow

kubectl delete -f .\k8s\ --ignore-not-found=true

Write-Host "`nRemaining resources:" -ForegroundColor Cyan
kubectl get deployments
kubectl get services

Write-Host "`nPhase 5 resources removed." -ForegroundColor Green