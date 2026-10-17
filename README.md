# MLOpsAssignment1
 Build a machine learning classifier to predict the risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.

 Clone repo: git clone -b feature/MLOps https://github.com/2025ae05691-rgb/MLOpsAssignment1.git

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


