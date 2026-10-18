# MLOpsAssignment1
 Build a machine learning classifier to predict the risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.

 Repo: https://github.com/2025ae05691-rgb/MLOpsAssignment1

 Clone repo: git clone -b feature/MLOps https://github.com/2025ae05691-rgb/MLOpsAssignment1.git

 Pull:
 git fetch origin
git pull --ff-only origin feature/MLOps

 git commit -m "Initial commit for Heart Disease MLOps project"    

 # Phase 1 - Build a machine learning model.

1. Create virtual environment:

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip,.
pip install -r requirements.txt
.\run_phase1.ps1

2. Phase 1 Deliverables:
a) Python virtual environment and dependencies verifiedDataset downloaded and shape checked
b) EDA plots and missing-value analysis generated
c) Two classification models compared using cross-validation
d) Both models tuned and final model selected using training data
e) Held-out test metrics and evaluation plots saved
f) Fitted model pipeline saved and reloaded successfully

# Phase 2 - Track experiments and make the model reproducible
1. .\src\train_with_mlflow.py
2. mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
3. python -m pip freeze > requirements-lock.txt
4. python -m pip check
5. python src\verify_model.py (or)  .\.venv\Scripts\python.exe src\verify_model.py

Phase 2 deliverables:

a) MLflow installed and version verified
b) Tracked training completed without errors
c) Two model runs visible in MLflow
d)  Parameters and cross-validation metrics logged
e) Held-out test metrics and plots logged as artifacts
f)  Both MLflow runs contain model artifacts
g) Selected joblib pipeline loads successfully
h) requirements-lock.txt created and pip check passes

# Phase 3 CI/CD and Automated Testing
1. __init__.py
2. tests/test_pipeline.py
3. Run the tests: python -m pytest tests -v
4. New-Item -ItemType Directory -Force .github\workflows
5. Create this file  .github/workflows/ci.yml

# Phase 4 - Windows, Docker and Kubernetes
Step 1:
wsl --version
wsl --status

Docker Desktop: https://docs.docker.com/desktop/setup/install/windows-install

docker --version
docker info
docker run --rm hello-world

Step 2:
In Docker Desktop, enable Kubernetes under Settings → Kubernete

kubectl version --client
kubectl config current-context
kubectl get nodes
kubectl get pods -A

Step 3: Fast API

python -m pip install fastapi "uvicorn[standard]"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

Launch Fast API: http://127.0.0.1:8000/docs

Step 4: Create docker image and docker ignore
The Docker image installs only the API runtime dependencies from `requirements-runtime.txt`; the full `requirements.txt` remains for development, training, and CI.

Step 5: Build image , push 
docker build -t heart-disease-api:1.0 .

Step 6: docker run --rm --name heart-disease-api -p 8000:8000 heart-disease-api:1.0

Test API which hosted container - http://localhost:8000/docs

Step 7: Kubernetes set up 
New-Item -ItemType Directory -Force .\k8s
It requried K8s/deployment.yml and k8s/service.yml

Push the image to dockerhub:

docker login -u santoshsrinivasg
docker tag heart-disease-api:1.0 santoshsrinivasg/heart-disease-api:1.0
docker push santoshsrinivasg/heart-disease-api:1.0

Kubernetes rollout:

kubectl apply -f .\k8s\deployment.yml
kubectl rollout status deployment/heart-disease-api

verify: 
kubectl get deployments
kubectl get pods
kubectl get services

Host service on local : 
kubectl port-forward service/heart-disease-api 8000:8000
http://127.0.0.1:8000/docs

Below are Record the following for your report:
- Docker version and successful image build.
- Kubernetes node status.
- Deployment and Service YAML files.
- Successful rollout and pod status.
- API health-check response.
- Sample prediction response.
- Container or Kubernetes logs.
- Python and library versions used to build the model and API.

Useful commands:

Kubernetes node not ready	kubectl get nodes
Pod not starting	        kubectl describe pod -l app=heart-disease-api
Application error	        kubectl logs deployment/heart-disease-api
Service configuration	    kubectl describe service heart-disease-api
Recent cluster events	    kubectl get events --sort-by=.metadata.creationTimestamp
Deployment status	        kubectl rollout status deployment/heart-disease-api

Phase 6: Monitoring

python -m pip install prometheus-client -- Added to requirements.txt

docker build -t santoshsrinivasg/heart-disease-api:1.1 .
docker push santoshsrinivasg/heart-disease-api:1.1

Add new version(santoshsrinivasg/heart-disease-api:1.1) to deployment.yml

kubectl apply -f .\k8s\deployment.yml
kubectl set image deployment/heart-disease-api api=santoshsrinivasg/heart-disease-api:1.1
kubectl rollout status deployment/heart-disease-api
kubectl get deployment heart-disease-api -o jsonpath="{.spec.template.spec.containers[0].image}"
kubectl port-forward service/heart-disease-api 8002:8000

Additional endpoints:
- `GET /model-info` returns the API version, model type, feature names/count, and model classes.
- `GET /metrics/summary` returns request/error counts and average request latency as JSON.
- `GET /metrics` exposes raw Prometheus metrics.
- `POST /predict/batch` accepts a non-empty `patients` array and returns one prediction per patient, in input order.

Test locally after port-forwarding:
http://127.0.0.1:8002/model-info
http://127.0.0.1:8002/metrics/summary
http://127.0.0.1:8002/metrics

Batch prediction request shape:
```json
{
  "patients": [
    {
      "age": 55,
      "sex": 1,
      "cp": 2,
      "trestbps": 130,
      "chol": 220,
      "fbs": 0,
      "restecg": 1,
      "thalach": 150,
      "exang": 0,
      "oldpeak": 1.0,
      "slope": 2,
      "ca": 0,
      "thal": 3
    }
  ]
}
```

To deploy API code changes, build and push a new image tag, update the image in `k8s/deployment.yml`, and apply the manifest:
docker build -t santoshsrinivasg/heart-disease-api:1.2 .
docker push santoshsrinivasg/heart-disease-api:1.2
kubectl apply -f .\k8s\deployment.yml
kubectl rollout status deployment/heart-disease-api

# Evidence collection steps
Step 1: Unit test cases and code quality check
.\.venv\Scripts\ruff.exe check src tests --select E,F --ignore E402
.\.venv\Scripts\python.exe -m pytest tests -v

