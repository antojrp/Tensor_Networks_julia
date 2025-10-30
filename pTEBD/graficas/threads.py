import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necesario para algunas versiones de Matplotlib

# Carpeta donde están los resultados
carpeta = r'../resultados/'

# Capas que quieres graficar
capas_objetivo = [10, 11, 12]

# Patrón del nombre de archivo: tiempos_<capas>_t_<threadsJulia>_<threadsBlas>.txt
patron = re.compile(r"tiempos_(\d+)_t_(\d+)_(\d+)\.txt")

# Diccionario donde guardamos los resultados
# resultados[capas] = [(threadsJulia, threadsBlas, total, varianza), ...]
resultados = {}

# Leer todos los archivos de la carpeta
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

# Mostrar qué archivos se usaron
print("Archivos procesados:")
for capas, datos in resultados.items():
    print(f"  Capas {capas}: {len(datos)} combinaciones leídas")

# Graficar superficie 3D para cada número de capas
for capas, datos in resultados.items():
    tjulia = np.array([d[0] for d in datos])
    tblas = np.array([d[1] for d in datos])
    tiempo = np.array([d[2] for d in datos])
    varianza = np.array([d[3] for d in datos])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Dibujar superficie 3D (irregular)
    surf = ax.plot_trisurf(tjulia, tblas, tiempo,
                           cmap='plasma', linewidth=0.2, antialiased=True)

    # Agregar puntos medidos reales
    ax.scatter(tjulia, tblas, tiempo, color='k', s=30, label='Datos medidos')

    # Agregar barras de error (opcional)
    for x, y, z, var in zip(tjulia, tblas, tiempo, varianza):
        ax.plot([x, x], [y, y], [z - np.sqrt(var), z + np.sqrt(var)],
                color='black', alpha=0.6, linewidth=1)

    # Personalización
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label='Tiempo total (s)')
    ax.set_xlabel('Threads Julia')
    ax.set_ylabel('Threads BLAS')
    ax.set_zlabel('Tiempo total (s)')
    ax.set_title(f'Superficie 3D - {capas} capas')
    ax.legend()

    plt.tight_layout()
    plt.show()
