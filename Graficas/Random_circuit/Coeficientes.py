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

plt.figure()
N=20   
df = pd.read_csv('../../resultados/Random_coeficientes_70.txt', delim_whitespace=True, header=None)
x=range(len(df))
plt.plot(x, df, markersize=3,color=color[0], linestyle='-')   
    
# plt.hlines(1/pow(2,N/2),0,len(df)) 
plt.xlabel('N')
plt.ylabel('$\lambda$')
plt.title('Hola')
# Mostrar la leyenda
# Mostrar la gráfica
plt.tight_layout()
plt.show()
#savefig('Comparation_2'+str(i)+'.pdf',format='pdf', bbox_inches='tight')
