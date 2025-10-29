@echo off
setlocal

REM Set the project's root directory
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM Set the Python interpreter path if it's not in the system's PATH
REM set "PYTHON_EXE=C:\path\to\your\python.exe"
set "PYTHON_EXE=python"

echo [AURA PIPELINE] Starting End-to-End Workflow...
echo =================================================

echo.
echo [STEP 1/6] Generating synthetic raw data...
%PYTHON_EXE% scripts/generator_utils.py
if %errorlevel% neq 0 (echo [ERROR] Data generation failed. & exit /b %errorlevel%)

echo.
echo [STEP 2/6] Preprocessing data for models...
%PYTHON_EXE% scripts/preprocessing.py
if %errorlevel% neq 0 (echo [ERROR] Preprocessing failed. & exit /b %errorlevel%)

echo.
echo [STEP 3/6] Training Temporal Risk Model (Transformer)...
%PYTHON_EXE% scripts/train_transformer.py
if %errorlevel% neq 0 (echo [ERROR] Transformer training failed. & exit /b %errorlevel%)

echo.
echo [STEP 4/6] Training Network Risk Model (GNN)...
echo NOTE: This step assumes 'train_gnn.py' exists and functions correctly.
%PYTHON_EXE% scripts/train_gnn.py
if %errorlevel% neq 0 (echo [ERROR] GNN training failed. & exit /b %errorlevel%)

echo.
echo [STEP 5/6] Calculating final AURA Risk Scores...
%PYTHON_EXE% scripts/aura_risk_score.py
if %errorlevel% neq 0 (echo [ERROR] Final score calculation failed. & exit /b %errorlevel%)

echo.
echo [STEP 6/6] Running Explainability Engine for a high-risk borrower...
echo NOTE: This requires the GOOGLE_API_KEY environment variable to be set.
%PYTHON_EXE% scripts/explainability_engine.py
if %errorlevel% neq 0 (echo [ERROR] Explainability engine failed. & exit /b %errorlevel%)

echo.
echo =================================================
echo [AURA PIPELINE] Workflow Completed Successfully.
endlocal