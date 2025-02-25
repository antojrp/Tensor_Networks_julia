# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 19:09:02 2025

@author: ajrp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


archivo=r'../../Programas/resultados/Random_coeficientes_15.txt'
N=21  
coef=leer_coeficientes_schmidt(archivo, N)
plt.figure()
x=range(len(coef))
plt.plot(x, coef, markersize=3,color=color[0], linestyle='-')   
    
# plt.hlines(1/pow(2,N/2),0,len(df)) 
plt.xlabel('N')
plt.ylabel('$\lambda$')
plt.title('Hola')
# Mostrar la leyenda
# Mostrar la gráfica
plt.tight_layout()
plt.show()
#savefig('Comparation_2'+str(i)+'.pdf',format='pdf', bbox_inches='tight')



# Crear histograma normalizado para la distribución de probabilidad
num_bins = 80  # Número de intervalos
counts, bins = np.histogram(coef, bins=num_bins, density=True)
counts=counts / (sum(counts) * np.diff(bins))
# Definir parámetros para el ejemplo
N_A = 2**(N//2)  # Número de modos en A
N_B = 2**(N-N//2)  # Número de modos en B

# Definir a y b
a = (1/np.sqrt(N_A) - 1/np.sqrt(N_B))**2
b = (1/np.sqrt(N_A) + 1/np.sqrt(N_B))**2

# Definir la función ω_s(p)
def omega_s(p, N_A, N_B, a, b):
    if a <= p <= b:
        return (N_A * N_B / (2 * np.pi)) * np.sqrt((p - a) * (b - p)) /( N_A*p)
    else:
        return 0

# Crear valores de p en el rango relevante
p_values = np.linspace(0.000001, 2*b, 1000)
omega_values = [omega_s(p, N_A, N_B, a, b) for p in p_values]


# Graficar la distribución de probabilidad
plt.figure(figsize=(8, 5))
plt.plot(p_values, omega_values, label=r'$\omega_s(p)$', color='b')
plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.7)
plt.xlabel("Valores")
plt.ylabel("Probabilidad")
plt.title("Distribución de Probabilidad")
plt.show()
