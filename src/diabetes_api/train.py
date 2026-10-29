import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from diabetes_api.preprocessing import EfronScaler

CONFIG = {
    "random_seed": 42,
    "test_size": 0.3,
    "cv_folds": 5,
    "experiment_name": "diabetes_progression_search"
}

# Define Models and their tuning grids
MODELS = {
    "Linear Regression": (
        LinearRegression(),
        {
            "model__fit_intercept": [True, False]
        }
    ),
    "Ridge_Regression": (
        Ridge(random_state = CONFIG["random_seed"]),
        {
            "model__alpha": [0.1, 1.0, 10.0],
        }
    ),
    "RandomForestRegressor": (
        RandomForestRegressor(random_state = CONFIG["random_seed"]),
        {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [5, 10, None]
        }
    )
}

def main():
    # Load the data
    X, y = load_diabetes(scaled = False, return_X_y = True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = CONFIG["test_size"],
        random_state = CONFIG["random_seed"]
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(CONFIG["experiment_name"])
    print(f"Starting Two-Stage Experiment: {CONFIG['experiment_name']}\n")

    # Baseline Model Comparison
    print("Stage 1: Baseline Model Comparison\n")
    best_cv_score = -float("inf")
    winning_model_name = ""

    for name, (estimator, _) in MODELS.items():
        pipeline = Pipeline([('scaler', EfronScaler()), ('model', estimator)])

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv = CONFIG["cv_folds"], scoring = 'neg_mean_absolute_error'
        )
        avg_score = np.mean(scores)
        print(f"Baseline [{name}] CV Neg MAE: {avg_score:.4f}")

        if avg_score > best_cv_score:
            best_cv_score = avg_score
            winning_model_name = name

    print(f"\nWinner selected for tuning: {winning_model_name}\n")

    # Tune winner and evaluate on test set
    print("Stage 2: Hyperparameter Tuning and Test Evaluation\n")
    with mlflow.start_run(run_name = f"Tuned_{winning_model_name}"):
        winner_estimator, winner_grid = MODELS[winning_model_name]
        pipeline = Pipeline([("scaler", EfronScaler()), ("model", winner_estimator)])

        # Exhaustive Grid Search
        search = GridSearchCV(
            pipeline,
            param_grid = winner_grid,
            cv = CONFIG["cv_folds"],
            scoring = "neg_mean_absolute_error",
            n_jobs = -1
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_

        # Evaluate on test set
        y_pred = best_pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Log metrics and model
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({"r2": r2, "mae": mae})
        mlflow.sklearn.log_model(
            best_pipeline, "model_artifact",
            skops_trusted_types = ["diabetes_api.preprocessing.EfronScaler"]
            )

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model_artifact"
        registered_name = "diabetes_progression_model"

        mv = mlflow.register_model(model_uri, registered_name)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(registered_name, "champion", mv.version)

        print(
            "\nArtifact logged and aliased as @champion.\n"
            "Run `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db "
            "--host 127.0.0.1` to view."
        )

if __name__ == "__main__":
    main()