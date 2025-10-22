

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
def generar_matrices(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    matrices = {
        'D': [],
        'var(D)': [],
        'Renyi': [],
        'var(Renyi)': []
    }
    qubits = []
    current_qubits = None
    data = []

    for line in lines:
        line = line.strip()
        if line.startswith('Numero de qubits:'):
            if current_qubits is not None:
                # Convertir los datos actuales en DataFrame
                df = pd.DataFrame(data, columns=['Layer', 'D', 'var(D)', 'Renyi', 'var(Renyi)'])
                for key in matrices:
                    matrices[key].append(df[key].values)
                qubits.append(current_qubits)
                data = []
            current_qubits = int(line.split(':')[1].strip())
        elif line and not line.startswith('Layer'):
            data.append([float(x) for x in line.split()])

    # Agregar los últimos datos
    if current_qubits is not None:
        df = pd.DataFrame(data, columns=['Layer', 'D', 'var(D)', 'Renyi', 'var(Renyi)'])
        for key in matrices:
            matrices[key].append(df[key].values)
        qubits.append(current_qubits)

    # Crear las matrices finales
    final_matrices = {}
    for key in matrices:
        final_matrices[key] = np.column_stack(matrices[key])

    return final_matrices, qubits





L=30
N=20
plt.figure()
c=0
for p in [8,9,9.5,10]:
    # Ruta del archivo de texto
    file_path = '../../Programas/resultados/TEBD_entrelazamiento_'+str(p)+'.txt'
    # Generar matrices
    matrices, qubits = generar_matrices(file_path)
    x=range(1,L+1)
    y=matrices['D'][:L,0]
    error=matrices['var(D)'][:L,0]
    plt.errorbar(x, y, yerr=error, markersize=3, fmt='o',color=color[c], capsize=5, linestyle='None', label='N='+str(N))
    c=c+1

    
plt.xlabel('Layer')
plt.ylabel('D$_{Max}$')
plt.title('Maximum bond dimension per layer')
# Mostrar la leyenda
#plt.legend()
# Mostrar la gráfica
plt.tight_layout()
plt.show()
#savefig('Comparation_2'+str(i)+'.pdf',format='pdf', bbox_inches='tight')
plt.figure()


N=20
x=range(1,L+1)
c=0
for p in [8,9,9.5,10]:
    # Ruta del archivo de texto
    file_path = '../../Programas/resultados/TEBD_entrelazamiento_'+str(p)+'.txt'
    # Generar matrices
    matrices, qubits = generar_matrices(file_path)
    y=matrices['Renyi'][:L,0]
    error=matrices['var(Renyi)'][:L,0]
    plt.errorbar(x, y, yerr=error, markersize=3, fmt='o',color=color[c], capsize=5, linestyle='None', label='N='+str(N))
    c=c+1


plt.xlabel('Layer')
plt.ylabel('Renyi entroppy S$_2$')
plt.title('Renyi entropy per layer')
# Mostrar la leyenda
#plt.legend()
# Mostrar la gráfica
plt.tight_layout()
plt.show()
#savefig('Comparation_2'+str(i)+'.pdf',format='pdf', bbox_inches='tight')