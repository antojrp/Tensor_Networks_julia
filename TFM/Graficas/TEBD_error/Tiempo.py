

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

# Ruta del archivo de texto
file_path = '../../resultados/Random_tiempo_30.txt'

# Generar matrices
matrices, qubits = generar_matrices_tiempo(file_path)



L=30
N=20
c=0
plt.figure()
for p in [8,9,9.5,10]:
    # Ruta del archivo de texto
    file_path = '../../Programas/resultados/TEBD_tiempo_'+str(p)+'.txt'
    # Generar matrices
    matrices, qubits = generar_matrices_tiempo(file_path)
    x=range(1,L+1)
    y=matrices['Time'][:,0]
    error=matrices['var(Time)'][:,0]
    plt.errorbar(x, y, yerr=error, markersize=3, fmt='o',color=color[c], capsize=5, linestyle='None', label='N='+str(N)) 
    c=c+1
    
plt.xlabel('Layer')
plt.ylabel('Time(s)')
plt.title('Simulation time per layer')
# Mostrar la leyenda
#plt.legend()
# Mostrar la gráfica
plt.tight_layout()
plt.show()
#savefig('Comparation_2'+str(i)+'.pdf',format='pdf', bbox_inches='tight')
