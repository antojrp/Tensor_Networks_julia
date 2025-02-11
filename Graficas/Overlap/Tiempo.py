# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:07:03 2025

@author: ajrp
"""
import shutil
import subprocess
import numpy as np
from math import atan
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.size': 14,       # Tamaño de fuente general
    'axes.titlesize': 14,  # Tamaño del título de los ejes
    'axes.labelsize': 14,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 14, # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 14  # Tamaño de las etiquetas del eje y
})
color=['#073a4b','#108ab1','#06d7a0','#ffd167','#f04770']

dirfiles=r'../../resultados/'
p_max=12
N_ini=20
num=31
datos=np.empty((num,p_max))
var=np.empty((num,p_max))

filename=dirfiles+'Overlap_t15.txt'    

with open(filename, 'r') as f:
       p = -1
       N = 0
       for line in f:
           line = line.strip()
           if line.startswith("D:"):
               p=p+1
               N=0
           elif line.startswith("Total="):
               total = float(line.split("=")[1])
               datos[N,p]=total
           elif line.startswith("Varianza="):
               variance = float(line.split("=")[1])
               var[N,p]=variance
               N=N+1

N=30
p=10
plt.figure()
x=2**np.arange(1,p_max+1)
y=datos[N-N_ini,]
error=var[N-N_ini,]


def funcion_cubica(x, m):
    return m * x**3
popt, pcov = curve_fit(funcion_cubica, x, y, sigma=error, absolute_sigma=True)
m = popt[0]
y_fit = funcion_cubica(x, *popt)
residuals = y - y_fit  
ss_res = np.sum(residuals**2)  
ss_tot = np.sum((y - np.mean(y))**2)  
r_squared = 1 - (ss_res / ss_tot)

x_fino = np.linspace(np.min(x), np.max(x), 1000) 
y_fino = funcion_cubica(x_fino, *popt)

plt.plot(x_fino, y_fino, color='#f04770', label='Regression')
plt.errorbar(x, y, yerr=error, markersize=3, fmt='o',color=color[0], capsize=5, linestyle='None', label='MPS')

    
plt.xlabel('p')
plt.ylabel('Tiempo')
plt.title('Hola')
# Mostrar la leyenda
plt.legend()
# Mostrar la gráfica
plt.tight_layout()
plt.show()
#savefig('Comparation_2'+str(i)+'.pdf',format='pdf', bbox_inches='tight')

plt.figure()
x=range(N_ini,N_ini+num)
y=datos[:,p]
error=var[:,p]

def funcion_lineal(x, m,n):
    return m * x + n
popt, pcov = curve_fit(funcion_lineal, x, y, sigma=error, absolute_sigma=True)
m = popt[0]
y_fit = funcion_lineal(x, *popt)
residuals = y - y_fit  
ss_res = np.sum(residuals**2)  
ss_tot = np.sum((y - np.mean(y))**2)  
r_squared = 1 - (ss_res / ss_tot)

x_fino = np.linspace(np.min(x), np.max(x), 1000) 
y_fino = funcion_lineal(x_fino, *popt)

plt.plot(x_fino, y_fino, color='#f04770', label='Regression')

plt.errorbar(x, y, yerr=error, markersize=3, fmt='o',color=color[0], capsize=5, linestyle='None', label='MPS')

    
plt.xlabel('N')
plt.ylabel('Tiempo')
plt.title('Hola')
# Mostrar la leyenda
plt.legend()
# Mostrar la gráfica
plt.tight_layout()
plt.show()
#savefig('Comparation_2'+str(i)+'.pdf',format='pdf', bbox_inches='tight')