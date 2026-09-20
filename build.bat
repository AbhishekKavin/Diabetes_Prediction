@echo off
echo --- Step 1: Exporting @champion model from registry ---
uv run python scripts\export_model.py
if %errorlevel% neq 0 (
    echo Error: Model export failed. Aborting build.
    exit /b %errorlevel%
)

echo.
echo --- Step 2: Building Docker image ---
docker build -t diabetes-api .