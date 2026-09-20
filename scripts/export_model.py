import mlflow
from mlflow.tracking import MlflowClient

def export():
    print("Exporting the champion model....")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.artifacts.download_artifacts(
        artifact_uri = "models:/diabetes_progression_model@champion",
        dst_path = "./model_package"
    )
    print("Export Complete")

if __name__ == "__main__":
    export()