
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "heart_disease_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg","exang", "slope", "ca", "thal"
]

RANDOM_STATE = 42


def build_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", model),
    ])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    expected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_columns = set(expected_features + ["target"]) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    X = df[expected_features]
    y = df["target"].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Target must contain both classes 0 and 1.")

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

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    # Initial cross-validation comparison on training data.
    comparison = {}
    pipelines = {}

    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        pipelines[name] = pipeline

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        comparison[name] = {
            metric: {
                "mean": float(cv_results[f"test_{metric}"].mean()),
                "std": float(cv_results[f"test_{metric}"].std()),
            }
            for metric in scoring
        }

        print(f"\n{name} cross-validation:")
        for metric, values in comparison[name].items():
            print(
                f"  {metric}: {values['mean']:.3f} "
                f"+/- {values['std']:.3f}"
            )

    with open(REPORT_DIR / "cv_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Tune both models using training data only.
    parameter_grids = {
        "logistic_regression": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_leaf": [1, 3],
        },
    }

    searches = {}

    for name, pipeline in pipelines.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grids[name],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train, y_train)
        searches[name] = search

        print(f"\n{name} tuning:")
        print("  Best parameters:", search.best_params_)
        print(f"  Best CV ROC-AUC: {search.best_score_:.3f}")

    # Select the model with the highest training CV ROC-AUC.
    best_name = max(
        searches,
        key=lambda name: searches[name].best_score_,
    )
    best_search = searches[best_name]
    best_model = best_search.best_estimator_

    # Evaluate the selected model once on the held-out test set.
    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "selected_model": best_name,
        "best_parameters": best_search.best_params_,
        "best_cv_roc_auc": float(best_search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, y_probability)
        ),
    }

    with open(REPORT_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSelected model:", best_name)
    print("\nHeld-out test metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save the full fitted preprocessing + model pipeline.
    model_path = MODEL_DIR / "heart_disease_model.joblib"
    joblib.dump(best_model, model_path)
    print("\nSaved model:", model_path)

    # Confusion matrix.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix - Selected Model")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close("all")

    # ROC curve.
    RocCurveDisplay.from_predictions(y_test, y_probability)
    plt.title("ROC Curve - Selected Model")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_curve.png", dpi=160)
    plt.close("all")

    print("Saved evaluation plots in:", REPORT_DIR)


if __name__ == "__main__":
    main()
