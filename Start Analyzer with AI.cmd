@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 py -3.11 -m venv .venv
  if errorlevel 1 goto :failed
)
if not exist ".venv\models-ready" (
  echo Installing local AI support. First analysis also downloads model weights.
  ".venv\Scripts\python.exe" -m pip install -r requirements-models.txt
  if errorlevel 1 goto :failed
  echo ready>".venv\models-ready"
)
set ANALYZER_MODELS=linear,chronos,laya
set USE_TF=0
set HF_HOME=%~dp0data\models
".venv\Scripts\python.exe" -m streamlit run analyzer_app.py --server.address 127.0.0.1
if errorlevel 1 goto :failed
exit /b 0
:failed
echo AI setup failed. The lighter Start Analyzer.cmd remains available.
pause
exit /b 1
