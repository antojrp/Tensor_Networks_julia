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

# Función para procesar el archivo y generar matrices separadas por número de qubits
def generar_matrices_por_qubits(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    datos_por_qubits = {}
    current_qubits = None
    data = []

    for line in lines:
        line = line.strip()
        if line.startswith('Numero de qubits:'):
            if current_qubits is not None:
                # Guardar los datos actuales en el diccionario
                if current_qubits not in datos_por_qubits:
                    datos_por_qubits[current_qubits] = []
                datos_por_qubits[current_qubits].append(pd.DataFrame(data, columns=['Layer', 'Time', 'var(Time)']))
                data = []
            current_qubits = int(line.split(':')[1].strip())
        elif line and not line.startswith('Layer') and not line.startswith('Total') and not line.startswith('Varianza'):
            data.append([float(x) for x in line.split()])

    # Agregar los últimos datos
    if current_qubits is not None:
        if current_qubits not in datos_por_qubits:
            datos_por_qubits[current_qubits] = []
        datos_por_qubits[current_qubits].append(pd.DataFrame(data, columns=['Layer', 'Time', 'var(Time)']))

    return datos_por_qubits

# Procesar los archivos y graficar los resultados
tiempo_total_por_qubits = {}
for i in range(1, 31):
    file_path = f'../../Programas/resultados/Random_tiempo_15_t{i}.txt'
    datos_por_qubits = generar_matrices_por_qubits(file_path)
    for qubits, dataframes in datos_por_qubits.items():
        if qubits not in tiempo_total_por_qubits:
            tiempo_total_por_qubits[qubits] = np.zeros(30)
        for df in dataframes:
            tiempo_total_por_qubits[qubits][i-1] += sum(df['Time'])

# Graficar los resultados
t = 30
x = range(1, t+1)
for idx, (qubits, tiempos) in enumerate(tiempo_total_por_qubits.items()):
    plt.figure()
    plt.errorbar(x, tiempos, yerr=0, markersize=3, fmt='o', color=color[idx % len(color)], capsize=5, linestyle='None')
    plt.xlabel('Threads')
    plt.ylabel('Time(s)')
    plt.title(f'Simulation time for {qubits} qubits, L=15')
    plt.tight_layout()
    plt.savefig(f'Threads_{qubits}_qubits.pdf', format='pdf', bbox_inches='tight')
    plt.show()
