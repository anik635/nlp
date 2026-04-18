@echo off
echo.
echo  =============================================
echo   SpeechFix -- ASR Grammar Correction Pipeline
echo  =============================================
echo.
echo  Starting Streamlit app...
echo  Open http://localhost:8501 in your browser
echo.
d:\project\.venv\Scripts\streamlit.exe run d:\project\speechfix\app.py --server.port 8501 --server.headless false
