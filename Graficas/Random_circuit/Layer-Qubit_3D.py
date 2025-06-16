import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

# Parámetros generales
L_min = 5
L_max = 15
q_min = 20
q_max = 60
q_extr = 100
umbral_tiempo = 600
L_extrapolar = [5,6,7,8,9,10,11,12]

# Inicializar matriz de tiempos
matriz_tiempos = np.zeros((L_max - L_min + 1, q_extr - q_min + 1))

# Leer archivos y llenar matriz
for L in range(L_min, L_max + 1):
    file_path = f"../../Programas/resultados/Random_tiempo_{L}.txt"
    
    with open(file_path, "r") as file:
        lines = file.readlines()
        qubits = None
        qubits_data = []
        time_data = []
        
        for line in lines:
            if "Numero de qubits" in line:
                qubits = int(line.split(":")[1].strip())
            elif "Total=" in line and qubits is not None:
                total_time = float(line.split("=")[1].strip())
                if total_time <= umbral_tiempo:
                    if q_min <= qubits <= q_max:
                        qubits_data.append(qubits)
                        time_data.append(total_time)

        qubits_data = np.array(qubits_data)
        time_data = np.array(time_data)
        
        mask = qubits_data >= 25
        qubits_data_filtered = qubits_data[mask]
        time_data_filtered = time_data[mask]
        
        if L in L_extrapolar and len(qubits_data_filtered) > 1:
            slope, intercept = np.polyfit(qubits_data_filtered, time_data_filtered, 1)
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

# -------------------------
# GRÁFICO 3D SOLO
# -------------------------

# Configurar el rango de capas y qubits para graficar
L_min_plot = 5    # capa mínima a mostrar
L_max_plot = 11   # capa máxima a mostrar
q_min_plot = 20   # qubits mínimo a mostrar
q_max_plot = 60   # qubits máximo a mostrar

# Índices relativos para la submatriz
L_indices = np.arange(L_min_plot, L_max_plot + 1) - L_min
q_indices = np.arange(q_min_plot, q_max_plot + 1) - q_min

submatriz_tiempos = matriz_tiempos[np.ix_(L_indices, q_indices)]
qubits_range_plot = np.arange(q_min_plot, q_max_plot + 1)
layers_range_plot = np.arange(L_min_plot, L_max_plot + 1)

Q, L = np.meshgrid(qubits_range_plot, layers_range_plot)

# Normalización con rango dinámico de colores basado en tiempos mostrados
tiempo_min = np.min(submatriz_tiempos[submatriz_tiempos > 0])
tiempo_max = np.max(submatriz_tiempos)
norm = mcolors.LogNorm(vmin=tiempo_min, vmax=tiempo_max)

# Crear figura 3D
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(Q, L, submatriz_tiempos, cmap='viridis', norm=norm, edgecolor='none')

ax.set_xlabel('Number of Qubits')
ax.set_ylabel('Layers (L)')
ax.set_zlabel('Time (s)')
ax.set_title('Time as a Function of Qubits and Layers')

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Time (s)')

# Ajustar ticks de colorbar a valores dentro del rango visible
tick_vals = [1, 10, 60, 600]
tick_vals = [v for v in tick_vals if tiempo_min <= v <= tiempo_max]
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([f"{v:.0f} s" if v < 60 else f"{int(v/60)} min" for v in tick_vals])

plt.tight_layout()
plt.savefig('Time-Qubit-3D.pdf', bbox_inches='tight')
plt.show()
