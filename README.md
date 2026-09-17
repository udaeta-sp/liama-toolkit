# LIAMA Toolkit
Aplicación local en Python para análisis FTIR-ATR de espectros `.SPA`, con visualización interactiva, procesamiento, anotaciones y análisis multivariado.

## Instalación en Windows

### Paso 1 — Instalar Python (una sola vez por computadora)
Si la computadora ya tiene Python, saltá al paso 2.

1. Descargar Python desde https://www.python.org/downloads/ (cualquier versión 3.10 o superior).
2. **Importante:** en la primera pantalla del instalador, marcar la casilla
   **"Add python.exe to PATH"** antes de apretar *Install*.
3. Terminar la instalación.

### Paso 2 — Ejecutar LIAMA
Copiar la carpeta del proyecto a la computadora y hacer doble clic en **`LIAMA.bat`**.

La primera vez, el programa avisa qué librerías faltan y pide confirmación para
instalarlas. Son varios cientos de MB, así que conviene tener buena conexión;
puede tardar unos minutos. Las siguientes veces abre directamente.

Eso es todo: no hace falta abrir una terminal ni escribir comandos.

## Instalación en Linux
```bash
python3 -m pip install -r requirements.txt
python3 run.py
```
Según la distribución, Qt puede necesitar paquetes del sistema
(por ejemplo `libxcb-cursor0` en Debian/Ubuntu).

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `LIAMA.bat` dice que no encuentra Python | Reinstalar Python marcando **"Add python.exe to PATH"** (paso 1). |
| La ventana se cierra sola apenas abre | Abrir `LIAMA.bat` y leer el mensaje antes de que cierre, o ejecutar `python run.py` desde una terminal en la carpeta. |
| Falla la instalación de librerías | Casi siempre es la conexión. Volver a ejecutar `LIAMA.bat`: retoma lo que falte. |
| Abre pero no lee los `.SPA` | Es `spectrochempy`. Reinstalarlo con `python -m pip install --force-reinstall spectrochempy`. |

## Instalación manual de librerías
El archivo `requirements.txt` es la lista de librerías que necesita el programa.
`LIAMA.bat` la usa automáticamente, pero se puede correr a mano:

```bat
python -m pip install -r requirements.txt
```

Librerías: `PyQt6`, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`,
`spectrochempy`, `openpyxl`.

## Ejecución sin el `.bat`
```bash
python run.py
```
`run.py` ajusta el path por su cuenta, así que funciona desde cualquier
directorio de trabajo.

## Flujo funcional de la app
1. Carga de carpeta con archivos `.SPA`.
2. Lectura y validación de espectros.
3. Selección de espectros en stage para trabajo activo.
4. Visualización + procesamiento espectral en tiempo de interacción.
5. Detección/anotación de picos.
6. Análisis multivariado (PCA / PLS-DA / RF) sobre el stage.
7. Exportación de figuras y CSV.

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

## Licencia
Ver `LICENSE`.
