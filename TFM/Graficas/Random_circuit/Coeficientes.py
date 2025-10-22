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
    'font.size': 19,       # Tamaño de fuente general
    'axes.titlesize': 19,  # Tamaño del título de los ejes
    'axes.labelsize': 19,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 19, # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 19  # Tamaño de las etiquetas del eje y
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


archivo=r'../../Programas/resultados/Random_coeficientes_21.txt'


N=20
coef=leer_coeficientes_schmidt(archivo, N)
plt.figure()
x=range(len(coef))
plt.plot(x, coef, markersize=3,color=color[0], linestyle='-')   
    

plt.xlabel('N')
plt.ylabel('$\lambda$')
plt.title('Hola')
# Mostrar la leyenda
# Mostrar la gráfica
plt.tight_layout()
plt.show()
#savefig('Comparation_2'+str(i)+'.pdf',format='pdf', bbox_inches='tight')


def freedman_diaconis(data):
    num_data = len(data)
    irq = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * irq / np.power(num_data, 1/3)
    num_bins = int((np.max(data) -  np.min(data)) / bin_width)  + 1
    return num_bins


# Crear histograma normalizado para la distribución de probabilidad
num_bins = freedman_diaconis(coef)  # Número de intervalos
nsim=21
counts_i= [None] * nsim
for i in range(nsim):    
    counts_i[i], bins = np.histogram(coef[1024*i:1024*(i+1)], bins=num_bins, density=True)

varianza=np.sqrt(np.var(counts_i,axis=0))
counts=np.sum(counts_i, axis=0)    
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

omega_vals =[omega_s(p, N_A, N_B, a, b) for p in bins[:-1]]
error = (counts[5:25] - omega_vals[5:25])**2/omega_vals[5:25]
chi=sum(error)
print(chi)

# Crear valores de p en el rango relevante
p_values = np.linspace(0.000001, 1.05*b, 1000)
omega_values = [omega_s(p, N_A, N_B, a, b) for p in p_values]

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24
})
# Graficar la distribución de probabilidad
fig, ax = plt.subplots(figsize=(8, 6))

# Gráfica principal
ax.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.7)
ax.errorbar(bins[:-1], counts, yerr=varianza/np.sqrt(nsim), fmt='none', color='black', capsize=5)
ax.plot(p_values, omega_values, label=r'$\omega_s(p)$', color='b')
ax.set_xticks([0, 0.001, 0.002, 0.003, 0.004])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim(top=3500)
ax.set_title("Probability distribution of squared \n Schmidt coefficients N="+str(N))

# Gráfica de zoom dentro de la principal
axins = inset_axes(ax, width="40%", height="40%", loc="center")  # Tamaño y posición

axins.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.7)
axins.errorbar(bins[:-1], counts, yerr=varianza/np.sqrt(nsim), fmt='none', color='black', capsize=5)
axins.plot(p_values, omega_values, label=r'$\omega_s(p)$', color='b')

# Definir los límites del zoom
x_min, x_max = 0.00022, 0.00078  # Ajusta según lo que quieras ver
y_min, y_max = 300, 900
axins.set_xlim(x_min, x_max)
axins.set_ylim(y_min, y_max)
axins.set_xticks([])
axins.set_yticks([])
# Conectar el recuadro de zoom con la gráfica principal
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.tight_layout()
#fig.savefig('Coefficients_distribution_'+str(N)+'.pdf',format='pdf', bbox_inches='tight')
plt.show()


