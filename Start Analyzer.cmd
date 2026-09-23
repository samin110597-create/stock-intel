@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: preparing Python. Python 3.11 or 3.12 must be installed.
  py -3.12 -m venv .venv
  if errorlevel 1 py -3.11 -m venv .venv
  if errorlevel 1 goto :failed
)
if not exist ".venv\analyzer-ready" (
  ".venv\Scripts\python.exe" -m pip install -r requirements-analyzer.txt
  if errorlevel 1 goto :failed
  echo ready>".venv\analyzer-ready"
)
".venv\Scripts\python.exe" -m streamlit run analyzer_app.py --server.address 127.0.0.1
if errorlevel 1 goto :failed
exit /b 0
:failed
echo Setup failed. Read the message above. Install Python 3.12 from python.org if Python was not found.
pause
exit /b 1
