import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import spectrochempy as scp

def cargar_archivos_spa(carpeta):
    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith('.spa')]
    rutas = [os.path.join(carpeta, f) for f in archivos]
    return archivos, rutas

def obtener_rango_x_automatico(ruta):
    dataset = scp.read_omnic(ruta)
    x = dataset.x.data
    return float(np.max(x)), float(np.min(x))

def mostrar_espectro(ruta, aplicar_suavizado=False, savgol_window=15, savgol_poly=3, xmin=None, xmax=None, bandas=[], bandas_colores=[], picos=[], picos_colores=[], lineas_x=[]):
    dataset = scp.read_omnic(ruta)
    x = dataset.x.data
    y = dataset.data[0]
    A = 2-np.log10(y)       # absorbancia
    y = A
    print(min(y),max(y))
    if aplicar_suavizado:
        if savgol_window % 2 == 0:
            savgol_window += 1
        segunda_derivada = savgol_filter(y, window_length=savgol_window, polyorder=savgol_poly, deriv=2)
    else:
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        segunda_derivada = np.gradient(np.gradient(y, dx), dx)

    # Rango
    if xmin is None or xmax is None:
        xmin, xmax = float(np.max(x)), float(np.min(x))
    if xmin < xmax:
        xmin, xmax = xmax, xmin
    mask = (x <= xmin) & (x >= xmax)
    x_rango = x[mask]
    y_rango = y[mask]
    der_rango = segunda_derivada[mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, dpi=130)
    ax1.plot(x_rango, y_rango, color='blue', lw=1)
    ax1.set_ylabel(f'{dataset.units}', fontsize=10)
    ax1.set_title('Espectro FTIR-ATR', fontsize=11)
    ax1.tick_params(labelsize=8)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(min(y_rango)*0.95, max(y_rango)*1.05)

    ax2.plot(x_rango, der_rango, color='red', lw=1)
    ax2.set_xlabel(f'{dataset.x.units}', fontsize=10)
    ax2.set_ylabel('2ª derivada', fontsize=10)
    ax2.set_title('Segunda derivada', fontsize=11)
    ax2.tick_params(labelsize=8)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(min(der_rango)*1.1, max(der_rango)*1.1)

    for idx, (centro, ancho) in enumerate(bandas):
        color = bandas_colores[idx] if idx < len(bandas_colores) else 'gray'
        ax1.axvspan(centro - ancho / 2, centro + ancho / 2, color=color, alpha=0.3)
        ax2.axvspan(centro - ancho / 2, centro + ancho / 2, color=color, alpha=0.3)

    for xline in lineas_x:
        ax1.axvline(x=xline, color='green', linestyle='--', linewidth=1)
        ax2.axvline(x=xline, color='green', linestyle='--', linewidth=1)

    for idx, pico in enumerate(picos):
        color = picos_colores[idx] if idx < len(picos_colores) else 'black'
        if pico >= min(x_rango) and pico <= max(x_rango):
            idx_cercano = np.argmin(np.abs(x - pico))
            y_val = y[idx_cercano]
            ax1.scatter(pico, y_val- (max(y_rango)-min(y_rango))*0.03, marker='^', color=color)

    plt.tight_layout()
    plt.show()

def guardar_imagen_superpuesta(ruta, aplicar_suavizado=False, savgol_window=15, savgol_poly=3, xmin=None, xmax=None, bandas=[], bandas_colores=[], picos=[], picos_colores=[], lineas_x=[], carpeta_salida=None, incluir_derivada=True):
    dataset = scp.read_omnic(ruta)
    x = dataset.x.data
    y = dataset.data[0]
    A = 2-np.log10(y)       # absorbancia
    y = A
    if aplicar_suavizado:
        if savgol_window % 2 == 0:
            savgol_window += 1
        segunda_derivada = savgol_filter(y, window_length=savgol_window, polyorder=savgol_poly, deriv=2)
    else:
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        segunda_derivada = np.gradient(np.gradient(y, dx), dx)

    if xmin is None or xmax is None:
        xmin, xmax = float(np.max(x)), float(np.min(x))
    if xmin < xmax:
        xmin, xmax = xmax, xmin
    mask = (x <= xmin) & (x >= xmax)
    x_rango = x[mask]
    y_rango = y[mask]
    der_rango = segunda_derivada[mask]

    fig, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
    ax1.plot(x_rango, y_rango, color='blue', lw=1, label='Espectro FTIR-ATR')
    ax1.set_ylabel('Absorbancia', color='blue', fontsize=10)
    ax1.set_xlabel('Número de onda (cm⁻¹)', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(min(y_rango)*0.95, max(y_rango)*1.05)

    if incluir_derivada:
        ax2 = ax1.twinx()
        ax2.plot(x_rango, der_rango, color='red', lw=1, label='2ª derivada')
        ax2.set_ylabel('2ª derivada', color='red', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='red', labelsize=8)
        ax2.set_ylim(min(der_rango)*1.1, max(der_rango)*1.8)

    for idx, (centro, ancho) in enumerate(bandas):
        color = bandas_colores[idx] if idx < len(bandas_colores) else 'gray'
        ax1.axvspan(centro - ancho / 2, centro + ancho / 2, color=color, alpha=0.3)
        if incluir_derivada:
            ax2.axvspan(centro - ancho / 2, centro + ancho / 2, color=color, alpha=0.3)
    for xline in lineas_x:
        ax1.axvline(x=xline, color='green', linestyle='--', linewidth=1)
        if incluir_derivada:
            ax2.axvline(x=xline, color='green', linestyle='--', linewidth=1)
    for idx, pico in enumerate(picos):
        color = picos_colores[idx] if idx < len(picos_colores) else 'black'
        if pico >= min(x_rango) and pico <= max(x_rango):
            idx_cercano = np.argmin(np.abs(x - pico))
            y_val = y[idx_cercano]
            ax1.scatter(pico, y_val- (max(y_rango)-min(y_rango))*0.03, marker='^', color=color)

    plt.title('FTIR-ATR y segunda derivada' if incluir_derivada else 'FTIR-ATR', fontsize=11)
    plt.tight_layout()
    nombre_base = os.path.splitext(os.path.basename(ruta))[0]
    carpeta_out = carpeta_salida or os.path.dirname(ruta)
    output_path = os.path.join(carpeta_out, f"{nombre_base}_rango.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path

def seleccionar_carpeta():
    carpeta = filedialog.askdirectory()
    if carpeta:
        archivos, rutas = cargar_archivos_spa(carpeta)
        if archivos:
            seleccion.set(archivos[0])
            lista_archivos['menu'].delete(0, 'end')
            for archivo in archivos:
                lista_archivos['menu'].add_command(label=archivo, command=tk._setit(seleccion, archivo))
            rutas_dict.clear()
            rutas_dict.update(dict(zip(archivos, rutas)))
            carpeta_var.set(carpeta)
            cargar_rango_x_default(rutas[0])
        else:
            messagebox.showinfo("Aviso", "No se encontraron archivos .SPA en la carpeta seleccionada.")

def cargar_rango_x_default(ruta):
    xmin, xmax = obtener_rango_x_automatico(ruta)
    entry_min.delete(0, tk.END)
    entry_min.insert(0, str(xmin))
    entry_max.delete(0, tk.END)
    entry_max.insert(0, str(xmax))

def leer_bandas():
    bandas_raw = entry_bandas.get().strip()
    colores_raw = entry_colores_bandas.get().strip()
    bandas = []
    colores = []
    if bandas_raw:
        for b in bandas_raw.split(','):
            if ':' in b:
                centro, ancho = b.split(':')
                bandas.append((float(centro), float(ancho)))
    if colores_raw:
        colores = [c.strip() for c in colores_raw.split(',')]
    return bandas, colores

def leer_picos():
    picos_raw = entry_picos.get().strip()
    colores_raw = entry_colores_picos.get().strip()
    picos = []
    colores = []
    if picos_raw:
        picos = [float(p.strip()) for p in picos_raw.split(',') if p.strip()]
    if colores_raw:
        colores = [c.strip() for c in colores_raw.split(',')]
    return picos, colores

def leer_lineas():
    lineas_raw = entry_lineas_x.get().strip()
    if lineas_raw:
        return [float(val.strip()) for val in lineas_raw.split(',') if val.strip()]
    return []

def leer_rango(ruta=None):
    try:
        xmin = entry_min.get()
        xmax = entry_max.get()
        if xmin == "" or xmax == "":
            if ruta:
                return obtener_rango_x_automatico(ruta)
            else:
                raise ValueError
        xmin = float(xmin)
        xmax = float(xmax)
        if xmin>xmax:(xmin,xmax)=(xmax,xmin)
        return xmin, xmax
    except:
        raise ValueError("Rango X inválido")

def abrir_archivo_seleccionado():
    archivo = seleccion.get()
    ruta = rutas_dict.get(archivo)
    if ruta:
        try:
            xmin, xmax = leer_rango(ruta)
        except:
            messagebox.showerror("Error", "Ingrese un rango válido (xmin < xmax)")
            return
        bandas, bandas_colores = leer_bandas()
        picos, picos_colores = leer_picos()
        lineas_x = leer_lineas()
        mostrar_espectro(
            ruta,
            aplicar_suavizado=var_suavizado.get(),
            savgol_window=savgol_window.get(),
            savgol_poly=savgol_poly.get(),
            xmin=xmin,
            xmax=xmax,
            bandas=bandas,
            bandas_colores=bandas_colores,
            picos=picos,
            picos_colores=picos_colores,
            lineas_x=lineas_x
        )

def exportar_imagen():
    archivo = seleccion.get()
    ruta = rutas_dict.get(archivo)
    if ruta:
        try:
            xmin, xmax = leer_rango(ruta)
        except:
            messagebox.showerror("Error", "Ingrese un rango válido (xmin < xmax)")
            return
        bandas, bandas_colores = leer_bandas()
        picos, picos_colores = leer_picos()
        lineas_x = leer_lineas()
        output_path = guardar_imagen_superpuesta(
            ruta,
            aplicar_suavizado=var_suavizado.get(),
            savgol_window=savgol_window.get(),
            savgol_poly=savgol_poly.get(),
            xmin=xmin,
            xmax=xmax,
            bandas=bandas,
            bandas_colores=bandas_colores,
            picos=picos,
            picos_colores=picos_colores,
            lineas_x=lineas_x,
            incluir_derivada=var_derivada.get()
        )
        messagebox.showinfo("Imagen guardada", f"Se guardó como:\n{output_path}")

def exportar_todos():
    carpeta = carpeta_var.get()
    if carpeta:
        archivos, rutas = cargar_archivos_spa(carpeta)
        if not archivos:
            messagebox.showinfo("Aviso", "No se encontraron archivos .SPA en la carpeta seleccionada.")
            return
        try:
            xmin, xmax = leer_rango(rutas[0])
        except:
            messagebox.showerror("Error", "Ingrese un rango válido (xmin < xmax)")
            return
        bandas, bandas_colores = leer_bandas()
        picos, picos_colores = leer_picos()
        lineas_x = leer_lineas()
        carpeta_salida = os.path.join(carpeta, "imagenes_exportadas")
        os.makedirs(carpeta_salida, exist_ok=True)
        for ruta in rutas:
            guardar_imagen_superpuesta(
                ruta,
                aplicar_suavizado=var_suavizado.get(),
                savgol_window=savgol_window.get(),
                savgol_poly=savgol_poly.get(),
                xmin=xmin,
                xmax=xmax,
                bandas=bandas,
                bandas_colores=bandas_colores,
                picos=picos,
                picos_colores=picos_colores,
                lineas_x=lineas_x,
                carpeta_salida=carpeta_salida,
                incluir_derivada=var_derivada.get()
            )
        messagebox.showinfo("Exportación completa", f"Se guardaron las imágenes en:\n{carpeta_salida}")

ventana = tk.Tk()
ventana.title("Visualizador y Exportador de Espectros .SPA")

frame = tk.Frame(ventana, padx=10, pady=10)
frame.pack()

seleccion = tk.StringVar()
rutas_dict = {}
var_suavizado = tk.BooleanVar()
savgol_window = tk.IntVar(value=15)
savgol_poly = tk.IntVar(value=3)
var_derivada = tk.BooleanVar(value=True)
carpeta_var = tk.StringVar()

tk.Button(frame, text="Seleccionar carpeta", command=seleccionar_carpeta).grid(row=0, column=0, padx=5, pady=5)
lista_archivos = ttk.OptionMenu(frame, seleccion, '')
lista_archivos.grid(row=0, column=1, padx=5)
tk.Button(frame, text="Ver espectro", command=abrir_archivo_seleccionado).grid(row=0, column=2, padx=5, pady=5)

tk.Checkbutton(frame, text="Aplicar suavizado", variable=var_suavizado).grid(row=1, column=0, sticky='w')
tk.Label(frame, text="Ventana Savitzky-Golay:").grid(row=1, column=1, sticky='e')
tk.Entry(frame, textvariable=savgol_window, width=6).grid(row=1, column=2)
tk.Label(frame, text="Orden polinomio:").grid(row=2, column=1, sticky='e')
tk.Entry(frame, textvariable=savgol_poly, width=6).grid(row=2, column=2)

tk.Label(frame, text="Rango x (cm⁻¹):").grid(row=3, column=0, sticky='e')
entry_min = tk.Entry(frame, width=10)
entry_min.grid(row=3, column=1)
entry_max = tk.Entry(frame, width=10)
entry_max.grid(row=3, column=2)

tk.Label(frame, text="Bandas (x:ancho, ...):").grid(row=4, column=0, sticky='e')
entry_bandas = tk.Entry(frame, width=28)
entry_bandas.grid(row=4, column=1, columnspan=2)
tk.Label(frame, text="Colores bandas (coma):").grid(row=5, column=0, sticky='e')
entry_colores_bandas = tk.Entry(frame, width=28)
entry_colores_bandas.insert(0, 'gray')
entry_colores_bandas.grid(row=5, column=1, columnspan=2)

tk.Label(frame, text="Picos (x,x,...):").grid(row=6, column=0, sticky='e')
entry_picos = tk.Entry(frame, width=28)
entry_picos.grid(row=6, column=1, columnspan=2)
tk.Label(frame, text="Colores picos (coma):").grid(row=7, column=0, sticky='e')
entry_colores_picos = tk.Entry(frame, width=28)
entry_colores_picos.insert(0, 'black')
entry_colores_picos.grid(row=7, column=1, columnspan=2)

tk.Label(frame, text="Líneas verticales (x,x,...):").grid(row=8, column=0, sticky='e')
entry_lineas_x = tk.Entry(frame, width=28)
entry_lineas_x.grid(row=8, column=1, columnspan=2)

tk.Checkbutton(frame, text="Guardar segunda derivada", variable=var_derivada).grid(row=9, column=0, columnspan=3, sticky='w')

tk.Button(frame, text="Exportar imagen", command=exportar_imagen).grid(row=10, column=0, columnspan=1, pady=10)
tk.Button(frame, text="Exportar todos", command=exportar_todos).grid(row=10, column=1, columnspan=2, pady=10)

ventana.mainloop()