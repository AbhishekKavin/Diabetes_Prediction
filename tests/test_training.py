import mlflow
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from diabetes_api.train import CONFIG

def test_training_clears_required_threshold():
    """
    End-to-end smoke test asserting the selected architecture 
    programmatically clears the R2 >= 0.35 and MAE <= 45 baseline.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    pipeline = mlflow.sklearn.load_model(f"models:/diabetes_progression_model@champion")

    X, y = load_diabetes(scaled = False, return_X_y = True)
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_seed"]
        )

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    assert r2 >= 0.35, f"Regression: R2 {r2:.4f} fell below 0.35 threshold"
    assert mae <= 45.0, f"Regression: MAE {mae:.4f} exceeded 45.0 threshold"

