# LIAMA Toolkit
Aplicación local en Python para análisis FTIR-ATR de espectros `.SPA`, con visualización interactiva, procesamiento, anotaciones y análisis multivariado.

## Arquitectura del programa
La aplicación está organizada en capas funcionales:

- `liama.main`
  - Punto de arranque de la GUI (crea `QApplication` y abre la ventana principal).
- `liama.mainwindow.MainWindow`
  - Orquestador central de estado y flujo.
  - Coordina carga de espectros, stage, tabs, canvas, exportación y análisis multivariado.
- `liama.core`
  - `spa_reader`: lectura de `.SPA` vía SpectroChemPy.
  - `spectrum`: modelo de datos espectral.
  - `processing`: pipeline de preprocesado (suavizado, derivadas, normalizaciones, escalados).
  - `peak_detection`: detección de picos.
  - `multivariate`: PCA, PLS-DA y Random Forest.
- `liama.widgets`
  - Componentes de UI por dominio: vista, procesamiento, anotaciones, exportación y panel multivariado.
  - `canvas_widget`: visualización matplotlib con ejes y controles interactivos.
- `liama.utils`
  - Tema visual y paletas de color.

## Flujo funcional de la app
1. Carga de carpeta con archivos `.SPA`.
2. Lectura y validación de espectros.
3. Selección de espectros en stage para trabajo activo.
4. Visualización + procesamiento espectral en tiempo de interacción.
5. Detección/anotación de picos.
6. Análisis multivariado (PCA / PLS-DA / RF) sobre el stage.
7. Exportación de figuras y CSV.

## Requisitos
- Python 3.10 o superior
- `pip` actualizado

## Instalación
### Windows (PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux (bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución
### Opción 1: launcher Python
```bash
python run.py
```

### Opción 2 (Windows): lanzador BAT
```bat
LIAMA.bat
```

### Opción 3: script instalable `liama`
```bash
python -m pip install -e .
liama
```

## Dependencias de runtime
Definidas en `requirements.txt` y alineadas con `pyproject.toml`:
- `PyQt6`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `spectrochempy`
- `openpyxl`

## Compatibilidad
- Objetivo: ejecución local en Windows y Linux.
- En Linux, Qt puede requerir paquetes del sistema según la distribución.

## Licencia
Ver `LICENSE`.
