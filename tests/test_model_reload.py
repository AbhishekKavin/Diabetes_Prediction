import mlflow
from sklearn.datasets import load_diabetes

def test_model_reloads_and_predicts():
    """
    Proves that the skops-serialized custom transformer survives a cold load
    in a fresh process, using the explicit champion alias.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model_uri = "models:/diabetes_progression_model@champion"
    loaded_pipeline = mlflow.sklearn.load_model(model_uri)

    # Assert model loaded successfully
    assert loaded_pipeline is not None

    # Prove the loaded model can predict on raw data without errors
    X, y = load_diabetes(scaled=False, return_X_y=True)
    predictions = loaded_pipeline.predict(X[:3])

    assert len(predictions) == 3, "Prediction count must match input count"
    for pred in predictions:
        assert 0 < pred < 500, f"Prediction {pred} out of expected physiological range"
