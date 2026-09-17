@echo off
setlocal
cd /d "%~dp0"
title LIAMA Toolkit

rem ---------------------------------------------------------------
rem Find a Python interpreter: the "py" launcher first, then PATH.
rem ---------------------------------------------------------------
set "PY="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto check_libs

python --version >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto check_libs

goto no_python

rem ---------------------------------------------------------------
rem Report any missing libraries and exit non-zero if there are any.
rem ---------------------------------------------------------------
:check_libs
echo Comprobando librerias...
%PY% -c "import importlib.util as u,sys;mods=['PyQt6','matplotlib','numpy','scipy','pandas','sklearn','spectrochempy','openpyxl'];miss=[m for m in mods if u.find_spec(m) is None];print('Faltan: '+', '.join(miss)) if miss else print('Todo listo.');sys.exit(1 if miss else 0)"
if errorlevel 1 goto install
goto launch

:install
echo.
echo Hay que instalar las librerias que faltan.
echo Se ejecutara: pip install -r requirements.txt
echo La descarga es de varios cientos de MB y puede tardar unos minutos.
echo.
echo Presiona una tecla para instalar, o cerra esta ventana para cancelar.
pause >nul
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto pip_failed

:launch
echo Iniciando LIAMA Toolkit...
%PY% run.py
if errorlevel 1 goto app_failed
endlocal
exit /b 0

:no_python
echo.
echo No se encontro Python en esta computadora.
echo.
echo   1. Descargalo de https://www.python.org/downloads/
echo   2. En la primera pantalla del instalador, marca la casilla
echo      "Add python.exe to PATH" antes de continuar.
echo   3. Cuando termine, volve a ejecutar LIAMA.bat.
echo.
pause
exit /b 1

:pip_failed
echo.
echo Fallo la instalacion de librerias.
echo Revisa la conexion a internet y volve a ejecutar LIAMA.bat.
echo.
pause
exit /b 1

:app_failed
echo.
echo LIAMA se cerro con un error (ver el mensaje de arriba).
echo.
pause
exit /b 1
