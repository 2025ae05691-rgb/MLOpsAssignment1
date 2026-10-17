
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)

from train import (
    build_pipeline,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "heart_disease_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TRACKING_DB = ROOT / "mlflow.db"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Store MLflow metadata in a local SQLite database.
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
mlflow.set_experiment("heart-disease-classification")

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df["target"].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected binary target labels 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model_configs = {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "grid": {
                "classifier__C": [0.1, 1.0, 10.0],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_leaf": [1, 3],
            },
        },
    }

    dataset_hash = sha256_file(DATA_FILE)
    candidates = []

    for model_name, config in model_configs.items():
        print(f"\nTraining and tracking: {model_name}")

        with mlflow.start_run(run_name=model_name):
            mlflow.set_tags({
                "model_family": model_name,
                "dataset": DATA_FILE.name,
                "dataset_sha256": dataset_hash,
                "stage": "phase_2_training",
            })

            mlflow.log_params({
                "random_state": RANDOM_STATE,
                "test_size": 0.20,
                "cv_folds": 5,
                "selection_metric": "roc_auc",
                "class_weight": "balanced",
                "model_family": model_name,
            })

            # Record baseline cross-validation results.
            baseline = build_pipeline(config["estimator"])

            baseline_scores = cross_validate(
                baseline,
                X_train,
                y_train,
                cv=cv,
                scoring=SCORING,
                n_jobs=-1,
            )

            for metric in SCORING:
                values = baseline_scores[f"test_{metric}"]
                mlflow.log_metric(
                    f"baseline_cv_{metric}_mean",
                    float(values.mean()),
                )
                mlflow.log_metric(
                    f"baseline_cv_{metric}_std",
                    float(values.std()),
                )

            # Tune only on the training partition.
            search = GridSearchCV(
                estimator=build_pipeline(config["estimator"]),
                param_grid=config["grid"],
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_params = {
                key: str(value)
                for key, value in search.best_params_.items()
            }
            mlflow.log_params(best_params)
            mlflow.log_metric(
                "best_cv_roc_auc",
                float(search.best_score_),
            )

            # Evaluate on the untouched test partition.
            predictions = best_model.predict(X_test)
            probabilities = best_model.predict_proba(X_test)[:, 1]

            metrics = {
                "test_accuracy": float(
                    accuracy_score(y_test, predictions)
                ),
                "test_precision": float(
                    precision_score(
                        y_test, predictions, zero_division=0
                    )
                ),
                "test_recall": float(
                    recall_score(
                        y_test, predictions, zero_division=0
                    )
                ),
                "test_roc_auc": float(
                    roc_auc_score(y_test, probabilities)
                ),
            }

            mlflow.log_metrics(metrics)

            # Save run-specific artifacts.
            run_report = REPORT_DIR / f"{model_name}_metrics.json"
            run_report.write_text(
                json.dumps({
                    "model": model_name,
                    "best_parameters": search.best_params_,
                    "best_cv_roc_auc": float(search.best_score_),
                    **metrics,
                }, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(run_report))

            report_file = REPORT_DIR / f"{model_name}_classification_report.txt"
            report_file.write_text(
                classification_report(
                    y_test, predictions, zero_division=0
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_file))

            cm = confusion_matrix(
                y_test, predictions, labels=[0, 1]
            )
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"],
            )
            plt.title(f"Confusion Matrix - {model_name}")
            plt.tight_layout()

            cm_file = REPORT_DIR / f"{model_name}_confusion_matrix.png"
            plt.savefig(cm_file, dpi=160)
            plt.close("all")
            mlflow.log_artifact(str(cm_file))

            RocCurveDisplay.from_predictions(y_test, probabilities)
            plt.title(f"ROC Curve - {model_name}")
            plt.tight_layout()

            roc_file = REPORT_DIR / f"{model_name}_roc_curve.png"
            plt.savefig(roc_file, dpi=160)
            plt.close("all")
            mlflow.log_artifact(str(roc_file))

            # Log the complete fitted preprocessing/model pipeline.
            # sklearn pipelines include numpy.dtype; skops requires it to be trusted.
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            candidates.append({
                "name": model_name,
                "cv_auc": float(search.best_score_),
                "model": best_model,
                "metrics": metrics,
                "params": search.best_params_,
            })

            print("Best parameters:", search.best_params_)
            print("Best CV ROC-AUC:", round(search.best_score_, 4))
            print("Test metrics:", metrics)

    # Select the model using training CV ROC-AUC, not test results.
    winner = max(candidates, key=lambda item: item["cv_auc"])

    final_model_path = MODEL_DIR / "heart_disease_model.joblib"
    joblib.dump(winner["model"], final_model_path)

    final_summary = {
        "selected_model": winner["name"],
        "selection_metric": "5-fold training CV ROC-AUC",
        "best_cv_roc_auc": winner["cv_auc"],
        "best_parameters": winner["params"],
        **winner["metrics"],
        "dataset_sha256": dataset_hash,
    }

    summary_path = REPORT_DIR / "mlflow_final_summary.json"
    summary_path.write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )

    print("\nSelected model:", winner["name"])
    print("Saved reusable model:", final_model_path)
    print("Final summary:", summary_path)
    print("\nMLflow tracking database:", TRACKING_DB)


if __name__ == "__main__":
    main()
