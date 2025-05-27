@echo off
echo Starting ICP Scoring Dashboard...
echo.
echo The dashboard will open in your default web browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.
streamlit run streamlit_icp_dashboard.py --server.port 8501 