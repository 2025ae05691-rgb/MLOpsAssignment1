$ErrorActionPreference = "Stop"

Write-Host "`n=== Phase 5: Kubernetes Deployment ===" -ForegroundColor Cyan

# 1. Confirm the Docker image exists.
$imageOutput = docker images -q heart-disease-api:1.0
if (-not $imageOutput) {
    throw "Docker image heart-disease-api:1.0 not found. Build the image before deploying."
}
Write-Host "Docker image found." -ForegroundColor Green

# 2. Confirm the Kubernetes context.
$context = kubectl config current-context
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read the Kubernetes context."
}

if ($context.Trim() -ne "docker-desktop") {
    Write-Warning "Current Kubernetes context is '$context', not 'docker-desktop'."
    Write-Warning "Confirm this is the cluster where your image is available."
}

# 3. Check Kubernetes connectivity.
kubectl get nodes
if ($LASTEXITCODE -ne 0) {
    throw "Kubernetes is not responding. Check Docker Desktop Kubernetes and retry."
}

# 4. Apply the Deployment and Service.
kubectl apply -f .\k8s\deployment.yaml
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply deployment.yaml."
}

kubectl apply -f .\k8s\service.yaml
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply service.yaml."
}

# 5. Wait for the Deployment.
kubectl rollout status deployment/heart-disease-api --timeout=180s
if ($LASTEXITCODE -ne 0) {
    kubectl describe deployment heart-disease-api
    kubectl logs deployment/heart-disease-api --all-containers=true
    throw "Deployment did not become ready. Review the diagnostics above."
}

# 6. Display deployment resources.
Write-Host "`n=== Deployments ===" -ForegroundColor Cyan
kubectl get deployments

Write-Host "`n=== Pods ===" -ForegroundColor Cyan
kubectl get pods -o wide

Write-Host "`n=== Services ===" -ForegroundColor Cyan
kubectl get services

Write-Host "`nDeployment completed. Verify the endpoints next." -ForegroundColor Green