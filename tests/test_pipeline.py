
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.train import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    build_pipeline,
)


def sample_data():
    """Create synthetic patient records for testing."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 50, 55, 60, 65],
        "trestbps": [110, 120, 125, 130, 135, 140, 145, 150],
        "chol": [180, 190, 200, 210, 220, 230, 240, 250],
        "thalach": [180, 175, 170, 165, 150, 145, 140, 135],
        "oldpeak": [0.0, 0.1, 0.2, 0.3, 1.0, 1.2, 1.5, 2.0],
        "sex": [0, 1, 0, 1, 0, 1, 0, 1],
        "cp": [1, 2, 1, 2, 3, 4, 3, 4],
        "fbs": [0, 0, 0, 1, 0, 1, 0, 1],
        "restecg": [0, 1, 0, 1, 0, 1, 0, 1],
        "exang": [0, 0, 0, 0, 1, 1, 1, 1],
        "slope": [1, 1, 2, 2, 1, 2, 3, 3],
        "ca": [0, 0, 1, 0, 1, 2, 2, 3],
        "thal": [3, 3, 6, 3, 7, 7, 6, 7],
    })


def test_pipeline_returns_binary_predictions():
    X = sample_data()
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = build_pipeline(
        LogisticRegression(max_iter=1000)
    )
    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_pipeline_returns_valid_probabilities():
    X = sample_data()
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = build_pipeline(
        LogisticRegression(max_iter=1000)
    )
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_pipeline_handles_missing_values():
    X = sample_data()
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    X.loc[0, "age"] = np.nan
    X.loc[1, "cp"] = np.nan

    model = build_pipeline(
        LogisticRegression(max_iter=1000)
    )
    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_pipeline_uses_expected_features():
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    assert len(expected) == 13
    assert "age" in expected
    assert "cp" in expected
    assert "thal" in expected
