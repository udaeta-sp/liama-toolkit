# LIAMA Toolkit – FTIR-ATR (Lichens)

Herramienta para visualizar y analizar espectros FTIR-ATR de líquenes (archivos `.SPA`), con foco en bandas diagnósticas y comparación por especie, tipo de roca y estrato (INFERIOR / MEDIO / SUPERIOR).

Desarrollado en el **LIAMA (UMYMFOR-CONICET, FCEN, UBA)**.

---

## Requisitos del sistema

- **Windows 10** (64-bit)
- **Python 3.10–3.12** (ver “Instalar Python” abajo)
- Conexión a internet solo la primera vez (para instalar dependencias)
- Suficientes permisos para ejecutar PowerShell

---

## Instalación rápida (usuarios sin Git)

1. **Instalar Python**
   - Descargá **Python 3.11 (64-bit)** desde `python.org/downloads/windows/`.
   - Durante la instalación, **marcá**: “**Add Python to PATH**”.
   - Finalizá el asistente.

2. **Descargar el toolkit**
   - En la página del repositorio de GitHub → **Code** → **Download ZIP**.
   - Descomprimí el ZIP, por ejemplo en: `C:\Users\USUARIO\Desktop\liama-toolkit`.

3. **Crear entorno virtual e instalar dependencias**
   - Abrí **PowerShell** en la carpeta del proyecto:
     ```powershell
     cd "C:\Users\USUARIO\Desktop\liama-toolkit"
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install --upgrade pip
     pip install -r requirements.txt
     ```

   > Si PowerShell bloquea el script de activación:
   > ```powershell
   > Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   > ```
   > Cerrá y reabrí PowerShell y repetí la activación del entorno.

4. **Ejecutar la interfaz gráfica**
   ```powershell
   python -m ftir_toolkit.gui.app
## Instalación rápida (usuarios con Git)
cd "C:\Users\USUARIO\Desktop"
git clone https://github.com/udaeta-sp/liama-toolkit.git
cd .\liama-toolkit\
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m ftir_toolkit.gui.app

## Estructura del proyecto

ftir_toolkit/
  ├─ cli/                # Interfaz de línea de comandos (opcional)
  ├─ core/               # IO, preprocesamiento, métricas de picos
  ├─ gui/                # Aplicación Tkinter (interfaz gráfica)
  ├─ viz/                # Funciones de graficado
  ├─ config.py           # Configuración de bandas/visualización
  └─ logging_setup.py    # Configuración de logging
tests/                   # Pruebas unitarias básicas
requirements.txt         # Dependencias de Python


