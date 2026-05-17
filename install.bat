@echo off
chcp 65001 >nul
echo ============================================================
echo   Property Classifier  -  Windows Installer
echo ============================================================
echo.
where python >nul 2>&1
if %errorlevel% neq 0 ( echo ERROR: Python not found & pause & exit /b 1 )

echo [1/11] NumPy 1.26.4 (pinned for TensorFlow compatibility)
python -m pip install "numpy==1.26.4" --force-reinstall --quiet
if %errorlevel% neq 0 ( echo FAILED: numpy & pause & exit /b 1 )

echo [2/11] setuptools (required by mlflow)
python -m pip install setuptools --upgrade --quiet
if %errorlevel% neq 0 ( echo FAILED: setuptools & pause & exit /b 1 )

echo [3/11] h5py (model checkpoints)
python -m pip install "h5py==3.11.0" --quiet
if %errorlevel% neq 0 ( echo FAILED: h5py & pause & exit /b 1 )

echo [4/11] TensorFlow 2.15.0
python -m pip install tensorflow==2.15.0 --quiet
if %errorlevel% neq 0 ( echo FAILED: tensorflow & pause & exit /b 1 )

echo [5/11] FastAPI + Uvicorn
python -m pip install fastapi==0.110.0 "uvicorn[standard]==0.29.0" python-multipart==0.0.9 --quiet
if %errorlevel% neq 0 ( echo FAILED: fastapi & pause & exit /b 1 )

echo [6/11] Pillow + aiofiles
python -m pip install pillow==10.3.0 aiofiles==23.2.1 --quiet
if %errorlevel% neq 0 ( echo FAILED: pillow & pause & exit /b 1 )

echo [7/11] scikit-learn + matplotlib + tqdm
python -m pip install scikit-learn==1.4.2 matplotlib==3.8.4 tqdm==4.66.2 --quiet
if %errorlevel% neq 0 ( echo FAILED: sklearn & pause & exit /b 1 )

echo [8/11] WandB
python -m pip install wandb==0.17.0 --quiet
if %errorlevel% neq 0 ( echo FAILED: wandb & pause & exit /b 1 )

echo [9/11] MLflow + PyArrow
python -m pip install mlflow==2.13.0 "pyarrow>=14.0.0,<16.0.0" --quiet
if %errorlevel% neq 0 ( echo FAILED: mlflow & pause & exit /b 1 )

echo [10/11] Prometheus + Requests
python -m pip install prometheus-client==0.20.0 requests==2.31.0 --quiet
if %errorlevel% neq 0 ( echo FAILED: prometheus & pause & exit /b 1 )

echo [11/11] Removing conflicting packages
python -m pip uninstall roboflow supervision opencv-python opencv-python-headless -y >nul 2>&1
echo Conflicts removed.

echo.
echo ============================================================
echo   Verifying...
echo ============================================================
python -c "import numpy; print('  numpy      :', numpy.__version__)"
python -c "import tensorflow; print('  tensorflow :', tensorflow.__version__)"
python -c "import wandb; print('  wandb      :', wandb.__version__)"
python -c "import mlflow; print('  mlflow     :', mlflow.__version__)"
python -c "import fastapi; print('  fastapi    :', fastapi.__version__)"
python -c "import h5py; print('  h5py       :', h5py.__version__)"
python -c "import pkg_resources; print('  setuptools : OK')"
echo.
python -m pip check
echo.
echo ============================================================
echo   Done! Next steps:
echo   1. .\setup_tracking.bat   (WandB key setup)
echo   2. python main.py         (train and serve)
echo ============================================================
pause
