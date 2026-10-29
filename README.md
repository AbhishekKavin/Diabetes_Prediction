# Production-Grade Diabetes Progression API

A robust, end-to-end MLOps pipeline and serving architecture for predicting diabetes disease progression. This project transforms a exploratory machine learning model into a secure, containerized, CI/CD-gated microservice following strict software engineering best practices.

---

## 🛠️ Architecture & Tech Stack

- **Machine Learning**: Scikit-learn, Custom Transformers (`EfronScaler`), MLflow (Tracking & Model Registry with `@champion` aliasing).
- **API Framework**: FastAPI, Pydantic (Strict schema validation, custom `.to_numpy()` mapping for zero feature misalignment).
- **Testing & Quality**: Pytest (17 comprehensive tests covering contracts, model reloading, and sanity checks), Ruff (Linting & Formatting).
- **Containerization**: Multi-stage Docker build using `uv` for lightning-fast, reproducible dependency management and a non-root security user.
- **CI/CD**: GitHub Actions (Linting $\rightarrow$ Dynamic Test Generation $\rightarrow$ Container Build validation gate).

---

## 📁 Project Structure

```text
diabetes_api/
├── .github/
│   └── workflows/
│       └── pr.yml          # CI pipeline merge gate
├── model_package/          # Standalone packed artifact for production deployment
├── notebooks/              # Exploratory data analysis (excluded from production linting)
├── scripts/
│   └── export_model.py     # Registry-to-artifact export utility
├── src/
│   └── diabetes_api/
│       ├── __init__.py
│       ├── app.py          # FastAPI application & lifespan loader
│       ├── preprocessing.py# Leak-free custom scikit-learn transformers
│       ├── schemas.py      # Pydantic data contracts & numpy alignment mapping
│       └── train.py        # Reproducible training pipeline & MLflow registration
├── tests/
│   ├── test_contracts.py   # Parameterized input validation bounds tests
│   ├── test_model_reload.py# Cold-load skops custom type verification
│   ├── test_preprocessing.py# Mathematical precision verification for scalers
│   └── test_training.py    # Cross-validation & regression threshold checks
├── .dockerignore
├── .gitignore
├── .python-version
├── build.bat               # Atomic local export and docker build automation script
├── Dockerfile              # Multi-stage production container build
├── pyproject.toml          # Centralized project config and Ruff settings
├── README.md
└── uv.lock                 # Pinned dependency lockfile
```

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager

### 1. Install Dependencies
Clone the repository and install exact dependencies using the lockfile:
```cmd
uv sync --frozen
```

### 2. Train and Register the Model
Run the training pipeline to generate the local SQLite tracking database (`mlflow.db`) and register the `@champion` model alias:
```cmd
uv run python src/diabetes_api/train.py
```

### 3. Run the Test Suite
Verify model performance, contract rejection boundaries, and cold-load serialization:
```cmd
uv run pytest tests/
```

### 4. Start the Local API Server
Launch the FastAPI service locally:
```cmd
uv run uvicorn diabetes_api.app:app --reload --port 8000
```
- Access interactive Swagger documentation at: `http://localhost:8000/docs`
- Check service health at: `http://localhost:8000/health`

---

## 🐳 Containerization & Deployment

To avoid shipping a local SQLite development database inside production containers, the workflow decouples the registry from the runtime image by packing the artifact beforehand.

### Automated Build Script (Windows)
Run the atomic build script, which automatically exports the current `@champion` model from MLflow and builds the container image:
```cmd
build.bat
```

### Run the Container
Run the isolated container on a machine with zero Python installations required:
```cmd
docker run -p 8000:8000 diabetes-api
```

---

## 🛡️ CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/pr.yml`) acts as a strict merge gate for all pull requests:
1. **Linting (Ruff)**: Fails fast on syntax errors, unused imports, or style violations.
2. **Test Generation & Execution**: Dynamically executes `train.py` on a clean checkout to seed the test environment, followed by `pytest` across all 17 test suites.
3. **Container Verification**: Exports the model package and runs `docker build` to guarantee base image stability and prevent environment drift before code merges to `master`.