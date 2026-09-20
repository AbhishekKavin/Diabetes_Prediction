import pytest
import numpy as np
from pydantic import ValidationError
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_diabetes
import mlflow

from diabetes_api.schemas import PatientFeatures
from diabetes_api.preprocessing import EfronScaler

@pytest.fixture
def trained_pipeline():
    """Pulls the model we trained"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    return mlflow.sklearn.load_model("models:/diabetes_progression_model@champion")

@pytest.fixture
def valid_data():
    return {
        "age": 45, "sex": 1, "bmi": 26.5, "bp": 100.0,
        "s1": 150.0, "s2": 100.0, "s3": 50.0, "s4": 3.0,
        "s5": 4.5, "s6": 100.0
    }

def test_schema_accepts_valid_clinical_data(valid_data):
    patient = PatientFeatures(**valid_data)
    assert patient.bmi == 26.5

@pytest.mark.parametrize("field, invalid_value", [
    ("age", 150),       # > 120
    ("age", -5),        # < 0
    ("sex", 3),         # Not Literal[1, 2]
    ("bmi", 5.0),       # < 10.0
    ("bp", 300.0),      # > 200.0
    ("s1", 450.0),      # > 400.0
    ("s2", 10.0),       # < 20.0
    ("s3", 200.0),      # > 150.0
    ("s4", 0.5),        # < 1.0
    ("s5", 8.0),        # > 7.0
    ("s6", 30.0),       # < 40.0
])

def test_schema_rejects_physiologically_impossible_data(valid_data, field, invalid_value):
    """Proves bounds fire correctly for each specific field independently."""
    valid_data[field] = invalid_value
    with pytest.raises(ValidationError) as exc_info:
        PatientFeatures(**valid_data)

    assert field in str(exc_info.value)

def test_schema_rejects_missing_features():
    with pytest.raises(ValidationError):
        PatientFeatures(age=45, bmi=25.0)

def test_model_output_sanity(trained_pipeline, valid_data):
    """Proves the model does not silently return negative or absurdly high predictions."""
    patient = PatientFeatures(**valid_data)
    X_new = patient.to_numpy()

    prediction = trained_pipeline.predict(X_new)[0]
    assert 0 < prediction < 500, f"Model generated absurd prediction: {prediction}"


