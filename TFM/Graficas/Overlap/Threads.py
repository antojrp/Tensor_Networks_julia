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

plt.rcParams.update({
    'font.size': 14,       # Tamaño de fuente general
    'axes.titlesize': 14,  # Tamaño del título de los ejes
    'axes.labelsize': 14,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 14, # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 14  # Tamaño de las etiquetas del eje y
})
color=['#073a4b','#108ab1','#06d7a0','#ffd167','#f04770']

dirfiles=r'./Files/'
p_max=12
t_min=1
t_max=20
datos=np.empty((t_max-t_min+1,p_max))
var=np.empty((t_max-t_min+1,p_max))

t=0
for i in range(t_max-t_min+1):
    filename=dirfiles+'Overlap_t'+str(i+1)+'.txt'    
    
    with open(filename, 'r') as f:
           p = -1
           for line in f:
               line = line.strip()
               if line.startswith("Qubits:30"):
                   b=True
                   p=p+1
               elif line.startswith("Total=") and b==True:
                   total = float(line.split("=")[1])
                   datos[t,p]=total
               elif line.startswith("Varianza=") and  b==True:
                   variance = float(line.split("=")[1])
                   var[t,p]=variance
                   b=False
    t=t+1

for p in range(p_max):
    
    plt.figure()
    x=range(t_min,t_max+1)
    y=datos[:,p]
    error=var[:,p]
    
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