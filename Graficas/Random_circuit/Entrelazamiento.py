

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 18,       # Tamaño de fuente general
    'axes.titlesize': 18,  # Tamaño del título de los ejes
    'axes.labelsize': 18,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 18, # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 18  # Tamaño de las etiquetas del eje y
})
color=['#073a4b','#108ab1','#06d7a0','#ffd167','#f04770','#073a4b']

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

# Ruta del archivo de texto
file_path = '../../Programas/resultados/Random_entrelazamiento_15.txt'

# Generar matrices
matrices, qubits = generar_matrices(file_path)



L=15
N_ini=20
plt.figure()
for N in range(0,len(qubits),2):
    print(N)
    x=range(1,L+1)
    y=matrices['D'][:L,N]
    error=matrices['var(D)'][:L,N]
    
    plt.errorbar(x, y, yerr=error, markersize=3, fmt='o',color=color[int(N/2)], capsize=5, linestyle='-', label='N='+str(N_ini+N))
    
plt.xlabel('Layer')
plt.ylabel('D')
plt.title('Maximum bond dimension of the MPS per layer')
#plt.yscale('log', base=2)  # Escala logarítmica base 2 en el eje Y
plt.yticks([2**10, 2**11, 2**12, 2**13, 2**14], labels=['$2^{10}$', '$2^{11}$', '$2^{12}$', '$2^{13}$', '$2^{14}$'])  # Etiquetas específicas en potencias de 2
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Cuadrícula
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Maximum_D.pdf',format='pdf', bbox_inches='tight')


plt.rcParams.update({
    'font.size': 14,       # Tamaño de fuente general
    'axes.titlesize': 14,  # Tamaño del título de los ejes
    'axes.labelsize': 14,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 14, # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 14  # Tamaño de las etiquetas del eje y
})

plt.figure()

N=21
x=range(1,L+1)
y=matrices['Renyi'][:L,N-N_ini]
error=matrices['var(Renyi)'][:L,N-N_ini]

plt.errorbar(x, y, yerr=error, markersize=3, fmt='o',color=color[0], capsize=5, linestyle='None', label='N='+str(N))

    
plt.xlabel('Layer')
plt.ylabel('Renyi entroppy S$_2$')
plt.title('Renyi entropy per layer N='+str(N)+' qubits')
# Mostrar la leyenda
#plt.legend()
# Mostrar la gráfica
plt.axhline(10, color='r', linestyle='--', label="Línea horizontal")
plt.tight_layout()
plt.show()
plt.savefig('Renyi_'+str(N)+'.pdf',format='pdf', bbox_inches='tight')