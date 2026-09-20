from fastapi.testclient import TestClient
from diabetes_api.app import app
import pytest

@pytest.fixture
def client():
    """
    This fixture creates the test client and triggers the app lifespan.
    It yields the client to the test, and cleans up when the test finishes.
    """
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "up", "model_loaded": True}

def test_predict_returns_valid_response(client):
    valid_payload = {
        "age": 45, "sex": 1, "bmi": 26.5, "bp": 100.0,
        "s1": 150.0, "s2": 100.0, "s3": 50.0, "s4": 3.0,
        "s5": 4.5, "s6": 100.0
    }
    
    response = client.post("/predict", json = valid_payload)

    assert response.status_code == 200
    data = response.json()
    assert "disease_progression_score" in data
    assert data["reference_range"] == "25 to 346"

def test_predict_rejects_invalid_input(client):
    invalid_payload = {
        "age": 150,  # Violates ge=0, le=120
        "sex": 1, "bmi": 26.5, "bp": 100.0,
        "s1": 150.0, "s2": 100.0, "s3": 50.0, "s4": 3.0,
        "s5": 4.5, "s6": 100.0
    }
    response = client.post("/predict", json = invalid_payload)

    assert response.status_code == 422
    errors = response.json()['detail']
    assert errors[0]["loc"] == ['body','age']
    assert errors[0]['type'] == 'less_than_equal'