@echo off
TITLE GeneWeaver CRISPR Web Engine Launcher
COLOR 0B
echo ===============================================================================
echo   GENEWEAVER: GPU-ACCELERATED CRISPR SAFETY & EFFICIENCY ENGINE
echo   Launching Dual-Mode Clinical Web Portal (Simple & Expert Interfaces)...
echo ===============================================================================
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo [*] Checking Python runtime...
python -c "import streamlit, plotly, pandas; print('[+] All core dependencies verified.')"
if %ERRORLEVEL% NEQ 0 (
    echo [!] Missing Python dependencies. Installing via pip...
    pip install streamlit plotly pandas numpy biopython textual rich
)

echo.
echo [*] Launching GeneWeaver Web Dashboard at http://localhost:8501 ...
echo [*] Press Ctrl+C in this console window to gracefully terminate the server.
echo.

python -m streamlit run src\geneweaver\web\app.py --server.headless=false --theme.base="dark"
pause
