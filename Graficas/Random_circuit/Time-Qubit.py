import numpy as np
import matplotlib.pyplot as plt
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
L_extrapolar = [5, 6, 7, 8, 9, 10, 11, 12]
L_max_grafica = 11  # límite para gráficas y ajuste de pendiente

# Inicialización
matriz_tiempos = np.zeros((L_max - L_min + 1, q_extr - q_min + 1))
pendiente = np.zeros(L_max - L_min + 1)
interceptos = np.zeros(L_max - L_min + 1)

# Lectura y procesamiento
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

        L_index = L - L_min
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

        # Guardar valores leídos directamente
        for qubit, time in zip(qubits_data, time_data):
            if q_min <= qubit <= q_max:
                q_index = qubit - q_min
                matriz_tiempos[L_index, q_index] = time

# --- Gráfica: Tiempo vs Qubits con líneas ajustadas (dos paneles con escalas y independientes) ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # Sin sharey

# Subgráfico 1: L = 5, 6, 7, 8
for L in range(5, 9):
    L_index = L - L_min
    qubits = np.arange(q_min, q_max)
    tiempos = matriz_tiempos[L_index, :q_max - q_min]

    axes[0].plot(qubits, tiempos, 'o', label=f'L={L}', zorder=10)
    if pendiente[L_index] > 0:
        ajuste = pendiente[L_index] * qubits + interceptos[L_index]
        axes[0].plot(qubits, ajuste, '--', color='grey')

axes[0].set_xlabel('Number of qubits')
axes[0].set_ylabel('Time (s)')
axes[0].grid()
axes[0].legend()

# Subgráfico 2: L = 9, 10, 11
for L in range(9, 12):
    L_index = L - L_min
    qubits = np.arange(q_min, q_max)
    tiempos = matriz_tiempos[L_index, :q_max - q_min]

    axes[1].plot(qubits, tiempos, 'o', label=f'L={L}', zorder=10)
    if pendiente[L_index] > 0:
        ajuste = pendiente[L_index] * qubits + interceptos[L_index]
        axes[1].plot(qubits, ajuste, '--', color='grey')

axes[1].set_xlabel('Number of qubits')
axes[1].set_ylabel('Time (s)')
axes[1].grid()
axes[1].legend()

fig.suptitle('Time vs Number of qubits', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Time-Qubit.pdf', bbox_inches='tight')
plt.show()


# --- Gráfica: Pendiente vs L ---
plt.figure(figsize=(15, 6))
L_vals_plot = np.arange(L_min, L_max_grafica + 1)
pendiente_plot = pendiente[:L_max_grafica + 1 - L_min]
plt.plot(L_vals_plot, pendiente_plot, 'o-', color='blue')
plt.xlabel('Layers')
plt.ylabel('Slope')
plt.title('Slope vs Layers')
plt.grid()
plt.savefig('Slopes.pdf', bbox_inches='tight')
plt.show()

# --- Ajuste: log2(pendiente) = a * L + b ---
mask = pendiente_plot > 0
L_vals_fit = L_vals_plot[mask]
log2_pend = np.log2(pendiente_plot[mask])
a, b, r_value, p_value, std_err = linregress(L_vals_fit, log2_pend)

print("\nAjuste log2(pendiente) = a * L + b")
print(f"  a = {a:.4f}")
print(f"  b = {b:.4f}")
print(f"  R^2 = {r_value**2:.4f}")

# Gráfica del ajuste logarítmico
plt.figure(figsize=(10, 6))
plt.plot(L_vals_fit, log2_pend, 'o', label='Data')
plt.plot(L_vals_fit, a * L_vals_fit + b, '-', label=f'Linear regresion: {a:.2f}·L + {b:.2f}')
plt.xlabel('Layer')
plt.ylabel('log₂(Slope)')
plt.title('Linear regression of the slope in logarithmic scale')
plt.grid()
plt.legend()
plt.savefig('log(slope).pdf', bbox_inches='tight')
plt.show()
