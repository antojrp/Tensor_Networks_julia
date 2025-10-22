import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Colores para las curvas
color = ['#073a4b', '#108ab1', '#06d7a0', '#ffd167', '#f04770']

# Parámetros
L_min = 5
L_max = 15
q_min = 20
q_max = 60
q_extr = 100
umbral_tiempo = 600
L_extrapolar = [5, 6, 7, 8, 9, 10, 11, 12]
L_max_grafica = 11

# Inicialización de matrices
matriz_tiempos = np.zeros((L_max - L_min + 1, q_extr - q_min + 1))
pendiente = np.zeros(L_max - L_min + 1)
interceptos = np.zeros(L_max - L_min + 1)
matriz_varianzas = np.zeros_like(matriz_tiempos)

# Procesamiento de archivos
for L in range(L_min, L_max + 1):
    file_path = f"../../Programas/resultados/Random_tiempo_{L}.txt"
    with open(file_path, "r") as file:
        lines = file.readlines()
        qubits_data = []
        time_data = []
        varianza_data = []

        qubits = None
        total_time = None

        for line in lines:
            if "Numero de qubits" in line:
                qubits = int(line.split(":")[1].strip())
            elif "Total=" in line:
                total_time = float(line.split("=")[1].strip())
            elif "Varianza=" in line:
                varianza = float(line.split("=")[1].strip())
                if total_time is not None and qubits is not None:
                    if total_time <= umbral_tiempo and q_min <= qubits <= q_max:
                        qubits_data.append(qubits)
                        time_data.append(total_time)
                        varianza_data.append(varianza)

        # Asegurarse de consistencia
        assert len(qubits_data) == len(time_data) == len(varianza_data), f"Inconsistencia en L={L}"

        # Convertir a arrays y filtrar
        qubits_data = np.array(qubits_data)
        time_data = np.array(time_data)
        varianza_data = np.array(varianza_data)

        mask = qubits_data >= 25
        qubits_data_filtered = qubits_data[mask]
        time_data_filtered = time_data[mask]
        varianza_data_filtered = varianza_data[mask]

        L_index = L - L_min

        # Ajuste lineal para extrapolación si corresponde
        if L in L_extrapolar and L <= L_max_grafica and len(qubits_data_filtered) > 1:
            slope, intercept = np.polyfit(qubits_data_filtered, time_data_filtered, 1)
            pendiente[L_index] = slope
            interceptos[L_index] = intercept
            qubits_extr = np.arange(q_min, q_extr + 1)
            time_extr = slope * qubits_extr + intercept
            for i, qubit in enumerate(qubits_extr):
                q_index = qubit - q_min
                if matriz_tiempos[L_index, q_index] == 0:
                    matriz_tiempos[L_index, q_index] = time_extr[i]
        else:
            print(f"Datos insuficientes para ajuste en L = {L} o fuera del rango de gráfico.")

        # Guardar datos originales
        for qubit, time, varianza in zip(qubits_data, time_data, varianza_data):
            if q_min <= qubit <= q_max:
                q_index = qubit - q_min
                matriz_tiempos[L_index, q_index] = time
                matriz_varianzas[L_index, q_index] = varianza

# --- Gráficas: Tiempo vs Qubits ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Subgráficos independientes

label_fontsize = 19
tick_fontsize = 19
legend_fontsize = 19
title_fontsize = 20

# Subgráfico 1: L = 5, 6, 7, 8
for L in range(5, 9):
    L_index = L - L_min
    qubits = np.arange(q_min, q_max)
    tiempos = matriz_tiempos[L_index, :q_max - q_min]
    errores = matriz_varianzas[L_index, :q_max - q_min]/np.sqrt(20)
    axes[0].errorbar(qubits, tiempos, yerr=errores, fmt='o', color=color[L - 5], label=f'L={L}', zorder=10, capsize=2)
    if pendiente[L_index] > 0:
        ajuste = pendiente[L_index] * qubits + interceptos[L_index]
        axes[0].plot(qubits, ajuste, '--', color='grey')

axes[0].set_xlabel('Number of qubits', fontsize=label_fontsize)
axes[0].set_ylabel('Time (s)', fontsize=label_fontsize)
axes[0].tick_params(axis='both', labelsize=tick_fontsize)
axes[0].grid()
axes[0].legend(fontsize=legend_fontsize)
axes[0].set_title('L = 5, 6, 7, 8', fontsize=title_fontsize)

# Subgráfico 2: L = 9, 10, 11
for L in range(9, 12):
    L_index = L - L_min
    qubits = np.arange(q_min, q_max)
    tiempos = matriz_tiempos[L_index, :q_max - q_min]
    errores = matriz_varianzas[L_index, :q_max - q_min]/np.sqrt(20)
    axes[1].errorbar(qubits, tiempos, yerr=errores, fmt='o', color=color[L - 7], label=f'L={L}', zorder=10, capsize=2)
    if pendiente[L_index] > 0:
        ajuste = pendiente[L_index] * qubits + interceptos[L_index]
        axes[1].plot(qubits, ajuste, '--', color='grey')

axes[1].set_xlabel('Number of qubits', fontsize=label_fontsize)
axes[1].set_ylabel('Time (s)', fontsize=label_fontsize)
axes[1].tick_params(axis='both', labelsize=tick_fontsize)
axes[1].grid()
axes[1].legend(fontsize=legend_fontsize)
axes[1].set_title('L = 9, 10, 11', fontsize=title_fontsize)

# Título global
fig.suptitle('Time vs Number of qubits', fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Time-Qubit.pdf', bbox_inches='tight')
plt.show()

# --- Ajuste log2(pendiente) = a * L + b ---
L_vals_plot = np.arange(L_min, L_max_grafica + 1)
pendiente_plot = pendiente[:L_max_grafica + 1 - L_min]
mask = pendiente_plot > 0
L_vals_fit = L_vals_plot[mask]
log2_pend = np.log2(pendiente_plot[mask])
a, b, r_value, p_value, std_err = linregress(L_vals_fit, log2_pend)

print("\nAjuste log2(pendiente) = a * L + b")
print(f"  a = {a:.4f}")
print(f"  b = {b:.4f}")
print(f"  R^2 = {r_value**2:.4f}")

# Gráfica del ajuste logarítmico
plt.figure(figsize=(8, 5))
plt.plot(L_vals_fit, log2_pend, color=color[1], linestyle='None', marker='o', label='Data')
plt.plot(L_vals_fit, a * L_vals_fit + b, linestyle='--', color='grey', label=f'Linear regression: $R^2$={r_value**2:.4f}')
plt.xlabel('Layer', fontsize=17)
plt.ylabel('log₂(Slope)', fontsize=17)
plt.title('Linear regression of the slope in logarithmic scale', fontsize=17)
plt.grid()
plt.legend(fontsize=17)
plt.tight_layout()
# plt.savefig('log(slope).pdf', bbox_inches='tight')
# plt.show()
