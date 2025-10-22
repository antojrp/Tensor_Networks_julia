import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
from scipy.stats import linregress

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

# Parámetros iniciales
L_min = 5
L_max = 15
q_min = 20
q_max = 60
q_extr = 100
umbral_tiempo = 600
L_extrapolar = [5,6,7,8,9,10,11,12]

# Inicialización
matriz_tiempos = np.zeros((L_max - L_min + 1, q_extr - q_min + 1))
pendiente = np.zeros(L_max - L_min + 1)
interceptos = np.zeros(L_max - L_min + 1)

# Lectura y procesamiento de archivos
for L in range(L_min, L_max + 1):
    file_path = f"../../Programas/resultados/Random_tiempo_{L}.txt"
    with open(file_path, "r") as file:
        lines = file.readlines()
        qubits_data = []
        time_data = []

        for line in lines:
            if "Numero de qubits" in line:
                qubits = int(line.split(":")[1].strip())
            elif "Total=" in line:
                total_time = float(line.split("=")[1].strip())
                if total_time <= umbral_tiempo and q_min <= qubits <= q_max:
                    qubits_data.append(qubits)
                    time_data.append(total_time)

        qubits_data = np.array(qubits_data)
        time_data = np.array(time_data)
        mask = qubits_data >= 25
        qubits_data_filtered = qubits_data[mask]
        time_data_filtered = time_data[mask]

        if L in L_extrapolar and len(qubits_data_filtered) > 1:
            slope, intercept = np.polyfit(qubits_data_filtered, time_data_filtered, 1)
            pendiente[L-L_min] = slope
            interceptos[L-L_min] = intercept
            qubits_extr = np.arange(q_min, q_extr + 1)
            time_extr = slope * qubits_extr + intercept
            L_index = L - L_min
            for i, qubit in enumerate(qubits_extr):
                q_index = qubit - q_min
                if matriz_tiempos[L_index, q_index] == 0:
                    matriz_tiempos[L_index, q_index] = time_extr[i]
        else:
            print(f"Datos insuficientes para ajuste lineal en L = {L} o L no está en la lista de extrapolación.")

        for qubit, time in zip(qubits_data, time_data):
            if q_min <= qubit <= q_max:
                L_index = L - L_min
                q_index = qubit - q_min
                matriz_tiempos[L_index, q_index] = time


# Tiempo vs Layer con líneas ajustadas
plt.figure(figsize=(15, 6))
for q in range(20,30 ):
    L_index = L - L_min
    layers = np.array(range(L_min, 13))
    tiempos = matriz_tiempos[:8, q]
    plt.plot(layers, tiempos, 'o', label=f'q={q}')


plt.xlabel('Layers')
plt.ylabel('Tiempo (s)')
plt.title('Tiempo de simulación vs Número de qubits con líneas ajustadas')
plt.grid()
plt.legend()
plt.show()

