

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

plt.rcParams.update({
    'font.size': 18,       # Tamaño de fuente general
    'axes.titlesize': 18,  # Tamaño del título de los ejes
    'axes.labelsize': 18,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 18, # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 18  # Tamaño de las etiquetas del eje y
})
plt.figure()
# Paso 1: Leer el archivo (suponiendo que es un archivo de texto con espacios)
archivo = 'p=4.txt'  # Cambia esto por la ruta de tu archivo de texto
# Leer el archivo de texto, indicando que el delimitador es un espacio
df = pd.read_csv(archivo, delim_whitespace=True, header=None)
# Paso 2: Extraer las columnas
x = df[0]  # Primera columna para el eje X
y = df[1]  # Segunda columna para el eje Y
plt.plot(x, y, markersize=3,color='#073a4b')
plt.xlabel('Qubits')
plt.ylabel('Memory (GB)')
plt.title('Necessary memory to represent \n the MPS $\chi_{lim}=2^{4}$')
# Mostrar la gráfica
plt.tight_layout()
plt.show()
plt.savefig('p=4.pdf',format='pdf', bbox_inches='tight')
