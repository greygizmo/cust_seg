@echo off
setlocal
set PYTHONPATH=src;%PYTHONPATH%

if "%1"=="" goto help

if "%1"=="install" goto install
if "%1"=="test" goto test
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="type-check" goto type_check
if "%1"=="clean" goto clean
if "%1"=="weights" goto weights
if "%1"=="score" goto score
if "%1"=="playbooks" goto playbooks
if "%1"=="call-lists" goto call_lists
if "%1"=="pipeline" goto pipeline
if "%1"=="dashboard" goto dashboard

echo Unknown target: %1
goto help

:install
pip install -r requirements.txt
goto end

:test
pytest tests/
goto end

:lint
ruff check src tests
goto end

:format
ruff format src tests
goto end

:type_check
mypy src tests
goto end

:clean
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc
goto end

:weights
python -m icp.cli.generate_weights
goto end

:score
python -m icp.cli.score_accounts
goto end

:playbooks
python -m icp.cli.build_playbooks
goto end

:call_lists
python -m icp.cli.export_call_lists
goto end

:pipeline
call :score
if errorlevel 1 goto error
call :playbooks
if errorlevel 1 goto error
call :call_lists
if errorlevel 1 goto error
goto end

:dashboard
streamlit run apps/dashboard.py
goto end

:help
echo Available targets:
echo   install      Install dependencies
echo   test         Run tests
echo   lint         Run ruff check
echo   format       Run ruff format
echo   type-check   Run mypy
echo   clean        Remove pycache
echo   weights      Generate asset weights
echo   score        Score accounts
echo   playbooks    Build playbooks
echo   call-lists   Export call lists
echo   pipeline     Run score -^> playbooks -^> call-lists
echo   dashboard    Run Streamlit dashboard
goto end

:error
echo Failed with error code %errorlevel%
exit /b %errorlevel%

:end
endlocal
