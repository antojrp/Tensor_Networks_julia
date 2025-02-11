# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:22:36 2024

@author: ajrp
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np

plt.rcParams.update({
    'font.size': 14,       # Tamaño de fuente general
    'axes.titlesize': 14,  # Tamaño del título de los ejes
    'axes.labelsize': 14,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 14, # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 14  # Tamaño de las etiquetas del eje y
})

# Cargar el archivo CSV
df = pd.read_csv('Tamaño.txt',delimiter='\t')
rows, cols = np.where(np.isnan(df))

# Extraer la columna de N (la primera columna) y las columnas de valores (potencias de 2)
N = df['N'].values  # Columna de N
columnas = df.columns[1:]  # Las otras columnas son las potencias de 2
valores = df[columnas].values  # Los valores correspondientes a cada N y columna

# Ahora vamos a crear un gráfico de dispersión
# Cada punto estará definido por un valor de N (en el eje X) y un valor de la columna correspondiente (en el eje Y)
plt.figure()

# Usar un bucle para graficar los puntos de cada columna

for i, columna in enumerate(columnas[:17]):
    plt.scatter(N,np.repeat(i+1,len(N)),c=valores[:, i]/1500,cmap='Spectral_r', norm=LogNorm(vmin=1e-3, vmax=1))

cbar = plt.colorbar(label='Percentaje of memory used ')

# Cambiar el formato de los valores de la barra de color a número normal (no científico)
def fmt(x, pos):
    return f'{x:.3f}'.rstrip('0').rstrip('.')  # Puedes ajustar los decimales según prefieras

cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt))

for r, c in zip(rows, cols):
    if (20+r)/2<c:  
        # plt.scatter(r+20, c-1, color='grey')  # Puntos rojos en huecos
        a=0
    
    else:
        plt.scatter(N[r], c, color='grey')  # Puntos rojos en huecos

plt.scatter(N[r], c, color='grey',label='Out of memory')
N_values = np.arange(20, 52)  # Aseguramos que solo usamos los valores de N que están en los datos
y_values = N_values // 2  # Calcular N//2 para cada N
plt.step(N_values-0.5, y_values+0.5, where='post', color='black', label='2$^{N/2}$=D')


# Etiquetas y título del gráfico
plt.xlabel('Number of qubits')
plt.ylabel('log$_2$(D)')
plt.title('Memory used to represent MPS')
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()
plt.savefig('Tamaño.pdf',format='pdf', bbox_inches='tight')