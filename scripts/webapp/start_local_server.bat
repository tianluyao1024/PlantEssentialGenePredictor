@echo off
setlocal

cd /d "%~dp0..\.."

set PYTHON_EXE=D:\Python\Python311\python.exe
set STREAMLIT_PORT=8501
set STREAMLIT_ADDRESS=0.0.0.0
set LOG_DIR=webapp_data\logs

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Starting PlantEssentialGenePredictor on http://%STREAMLIT_ADDRESS%:%STREAMLIT_PORT%
echo Local network users can usually open http://192.168.1.100:%STREAMLIT_PORT%
echo Logs:
echo   %LOG_DIR%\streamlit_stdout.log
echo   %LOG_DIR%\streamlit_stderr.log

"%PYTHON_EXE%" -m streamlit run webapp\app.py ^
  --server.address %STREAMLIT_ADDRESS% ^
  --server.port %STREAMLIT_PORT% ^
  --server.headless true ^
  --server.maxUploadSize 4096 ^
  1>> "%LOG_DIR%\streamlit_stdout.log" ^
  2>> "%LOG_DIR%\streamlit_stderr.log"

endlocal
