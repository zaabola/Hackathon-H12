@echo off
echo.
echo ==========================================
echo   Gabes AI Dashboard — Backend Launcher
echo ==========================================
echo.

:: Use Python 3.12 for GPU torch support
set PYTHON=py -3.12

echo [1/3] Verifying Python ^& packages...
%PYTHON% -c "import django; print('Django OK')" 2>nul || (
    echo [ERROR] Django not installed for Python 3.12
    echo Run: py -3.12 -m pip install -r requirements.txt
    pause && exit /b 1
)

%PYTHON% -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>nul || (
    echo [WARN] PyTorch not available - AI will use demo/stub mode
)

echo.
echo [2/3] Running migrations...
%PYTHON% manage.py migrate --no-input

echo.
echo [3/3] Starting Django development server on port 8000...
echo.
echo   Dashboard: http://localhost:3000
echo   API Root:  http://localhost:8000/api/
echo   Admin:     http://localhost:8000/django-admin/
echo.
echo Press Ctrl+C to stop.
echo.
%PYTHON% manage.py runserver 0.0.0.0:8000
