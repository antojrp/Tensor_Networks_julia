

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

# Función para procesar el archivo y generar matrices
def generar_matrices_tiempo(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    matrices = {
        'Time': [],
        'var(Time)': []
    }
    qubits = []
    current_qubits = None
    data = []

    for line in lines:
        line = line.strip()
        if line.startswith('Numero de qubits:'):
            if current_qubits is not None:
                # Convertir los datos actuales en DataFrame
                df = pd.DataFrame(data, columns=['Layer', 'Time', 'var(Time)'])
                for key in matrices:
                    matrices[key].append(df[key].values)
                qubits.append(current_qubits)
                data = []
            current_qubits = int(line.split(':')[1].strip())
        elif line and not line.startswith('Layer') and not line.startswith('Total') and not line.startswith('Varianza'):
            data.append([float(x) for x in line.split()])

    # Agregar los últimos datos
    if current_qubits is not None:
        df = pd.DataFrame(data, columns=['Layer', 'Time', 'var(Time)'])
        for key in matrices:
            matrices[key].append(df[key].values)
        qubits.append(current_qubits)

    # Crear las matrices finales
    final_matrices = {}
    for key in matrices:
        final_matrices[key] = np.column_stack(matrices[key])

    return final_matrices, qubits

tiempo_total=np.zeros(30)
for i in range(1,31):
    file_path = '../../Programas/resultados/Random_tiempo_15_t'+str(i)+'.txt'
    matrices, qubits = generar_matrices_tiempo(file_path)
    tiempo_total[i-1]=sum(matrices['Time'][:,0])


t=30
plt.figure()
x=range(1,t+1)
error=0

plt.errorbar(x, tiempo_total, yerr=error, markersize=3, fmt='o',color=color[0], capsize=5, linestyle='None')    
plt.xlabel('Threads')
plt.ylabel('Time(s)')
plt.title('Simulation time N=20, L=15')
# Mostrar la leyenda
#plt.legend()
# Mostrar la gráfica
plt.tight_layout()
plt.show()
plt.savefig('Threads.pdf',format='pdf', bbox_inches='tight')
