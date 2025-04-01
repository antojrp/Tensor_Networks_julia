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
    total_times = []
    total_variances = []
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
        elif line.startswith('Total='):
            total_times.append(float(line.split('=')[1].strip()))
        elif line.startswith('Varianza='):
            total_variances.append(float(line.split('=')[1].strip()))
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
    
    return final_matrices, qubits, np.array(total_times), np.array(total_variances)

L=10

# Ruta del archivo de texto
file_path = '../../Programas/resultados/Random_tiempo_'+str(L)+'.txt'

# Generar matrices
matrices, qubits, total_times, total_variances = generar_matrices_tiempo(file_path)


N_ini=20

# for N in range(len(qubits[:2])):
#     plt.figure()
#     x=range(1,L+1)
#     y=matrices['Time'][:,N]
#     error=matrices['var(Time)'][:,N]/np.sqrt(20)
    
#     plt.errorbar(x, y, yerr=error, markersize=3, fmt='o', color=color[N % len(color)], capsize=5, linestyle='None', label='N='+str(N_ini+N))    
#     plt.xlabel('Layer')
#     plt.ylabel('Time(s)')
#     plt.title('Simulation time per layer N='+str(N+N_ini)+' qubits')
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('Random_circuit_N_'+str(N+N_ini)+'_L_'+str(L)+'.pdf',format='pdf', bbox_inches='tight')



plt.rcParams.update({
    'font.size': 14,       # Tamaño de fuente general
    'axes.titlesize': 14,  # Tamaño del título de los ejes
    'axes.labelsize': 14,  # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 14, # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 14  # Tamaño de las etiquetas del eje y
})

plt.figure()
plt.errorbar(qubits, total_times, yerr=total_variances/np.sqrt(20), markersize=3, fmt='o', color=color[0], capsize=5, linestyle='None')
plt.plot(qubits, total_times, linestyle='--', color='grey') 
plt.xlabel('N qubits')
plt.ylabel('Time(s)') 
plt.title('Total time of a random circuit with '+str(L)+' layers') 
plt.tight_layout()
plt.show()
plt.savefig('Random_circuit_L_'+str(L)+'.pdf',format='pdf', bbox_inches='tight')

