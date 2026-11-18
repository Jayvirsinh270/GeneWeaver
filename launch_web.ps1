<#
.SYNOPSIS
    GeneWeaver: 1-Click Desktop Web Launcher (PowerShell).
.DESCRIPTION
    Launches the GeneWeaver Dual-Mode Web Portal in your default browser.
#>

$Host.UI.RawUI.WindowTitle = "GeneWeaver CRISPR Web Engine Launcher"
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "  GENEWEAVER: GPU-ACCELERATED CRISPR SAFETY & EFFICIENCY ENGINE" -ForegroundColor Cyan
Write-Host "  Launching Dual-Mode Clinical Web Portal (Simple & Expert Interfaces)..." -ForegroundColor White
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "[*] Verifying runtime dependencies..." -ForegroundColor Gray
python -c "import streamlit, plotly, pandas; print('[+] All core dependencies verified.')"

Write-Host ""
Write-Host "[*] Launching Streamlit Web App on http://localhost:8501 ..." -ForegroundColor Green
Write-Host "[*] Press Ctrl+C to stop the web server." -ForegroundColor Yellow
Write-Host ""

python -m streamlit run src\geneweaver\web\app.py --server.headless=false --theme.base="dark"
