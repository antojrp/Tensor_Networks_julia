import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# Carpeta con tus resultados
carpeta = r'../resultados/'

# Capas a representar
capas_objetivo = [10, 11, 12]

# Patrón del nombre de archivo
patron = re.compile(r"tiempos_(\d+)_t_(\d+)_(\d+)\.txt")

# Diccionario: resultados[capas] = [(tjulia, tblas, total, varianza)]
resultados = {}

# Leer los archivos
for archivo in os.listdir(carpeta):
    match = patron.match(archivo)
    if match:
        capas, tjulia, tblas = map(int, match.groups())
        if capas in capas_objetivo:
            with open(os.path.join(carpeta, archivo)) as f:
                contenido = f.read()
            total_match = re.search(r"Total=\s*([0-9.eE+-]+)", contenido)
            var_match = re.search(r"Varianza=\s*([0-9.eE+-]+)", contenido)
            if total_match and var_match:
                total = float(total_match.group(1))
                var = float(var_match.group(1))
                resultados.setdefault(capas, []).append((tjulia, tblas, total, var))

print("Archivos procesados:")
for capas, datos in resultados.items():
    print(f"  Capas {capas}: {len(datos)} combinaciones")

# Graficar cada superficie
for capas, datos in resultados.items():
    tjulia = np.array([d[0] for d in datos])
    tblas = np.array([d[1] for d in datos])
    tiempo = np.array([d[2] for d in datos])

    # Crear una cuadrícula regular
    xi = np.linspace(min(tjulia), max(tjulia), 50)
    yi = np.linspace(min(tblas), max(tblas), 50)
    X, Y = np.meshgrid(xi, yi)

    # Interpolación: valores estimados en toda la cuadrícula
    Z = griddata((tjulia, tblas), tiempo, (X, Y), method='cubic')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Dibujar superficie interpolada y coloreada
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', antialiased=True)

    # Añadir puntos medidos reales
    ax.scatter(tjulia, tblas, tiempo, color='k', s=40, label='Datos medidos')

    # Barra de color y etiquetas
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label='Tiempo total (s)')
    ax.set_xlabel('Threads Julia')
    ax.set_ylabel('Threads BLAS')
    ax.set_zlabel('Tiempo total (s)')
    ax.set_title(f'Superficie 3D interpolada - {capas} capas')
    ax.legend()

    plt.tight_layout()
    plt.show()
