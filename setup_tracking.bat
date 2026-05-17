@echo off
chcp 65001 >nul
echo ============================================================
echo   Property Classifier - Tracking Setup
echo ============================================================
echo.

echo Opening wandb_key.txt in Notepad...
echo Paste your WandB key into the file, save, and close Notepad.
echo Get your key from: https://wandb.ai/authorize
echo.
notepad wandb_key.txt
echo.

echo Testing WandB key...
python check_wandb.py
echo.

echo Setting up MLflow local tracking...
python -c "import os; os.makedirs('mlruns', exist_ok=True); print('MLflow ready.')"
echo.
echo To view MLflow dashboard, run in a new terminal:
echo   python -m mlflow ui --port 5000
echo   Then open: http://localhost:5000
echo.

echo Installing/updating dependencies...
python -m pip install setuptools --upgrade --quiet
python -m pip install wandb --upgrade --quiet
python -m pip install mlflow --upgrade --quiet
echo Done.
echo.

echo ============================================================
echo   Setup complete. Run: python main.py
echo ============================================================
pause
