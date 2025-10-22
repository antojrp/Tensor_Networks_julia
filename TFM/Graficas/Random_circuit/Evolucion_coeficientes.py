# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 19:09:02 2025

@author: ajrp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.size': 14,       # Tamaño de fuente general
    'axes.titlesize': 14,  # Tamaño del título de los ejes
    'axes.labelsize': 14,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 14, # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 14  # Tamaño de las etiquetas del eje y
})
color=['#073a4b','#108ab1','#06d7a0','#ffd167','#f04770']

def leer_coeficientes_schmidt(archivo, num_qubits):
    with open(archivo, 'r') as f:
        lineas = f.readlines()

    coeficientes = []
    leer = False

    for linea in lineas:
        linea = linea.strip()
        if linea.startswith("Numero de qubits:"):
            if int(linea.split(":")[1].strip()) == num_qubits:
                leer = True
                coeficientes = []  # Reiniciar la lista para el nuevo número de qubits
            else:
                leer = False
        elif leer and linea:
            try:
                coeficientes.append(float(linea))
            except ValueError:
                pass  # Ignorar líneas no numéricas

    return coeficientes


archivo=r'../../Programas/resultados/Random_coeficientes_3.txt'
N=20
coef=leer_coeficientes_schmidt(archivo, N)
plt.figure()
x=range(len(coef))
plt.plot(x, coef, markersize=3,color=color[0], linestyle='-')   
    

plt.xlabel('N')
plt.ylabel('$\lambda^2$')
plt.title('Hola')
# Mostrar la leyenda
# Mostrar la gráfica
plt.tight_layout()
plt.show()
#savefig('Comparation_2'+str(i)+'.pdf',format='pdf', bbox_inches='tight')



