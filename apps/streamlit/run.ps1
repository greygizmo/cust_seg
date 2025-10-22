$ErrorActionPreference = 'Stop'
Write-Host 'Starting ICP Scoring Dashboard...' -ForegroundColor Cyan
Write-Host ''
Write-Host 'The dashboard will open in your default web browser at:'
Write-Host 'http://localhost:8501' -ForegroundColor Green
Write-Host ''
Write-Host 'Press Ctrl+C to stop the dashboard'
Write-Host ''
streamlit run "$(Join-Path $PSScriptRoot 'app.py')" --server.port 8501
