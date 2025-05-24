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

# --- Gráfica: Tiempo vs Qubits con líneas ajustadas ---
plt.figure(figsize=(15, 6))
for L in range(L_min, L_max_grafica + 1):
    L_index = L - L_min
    qubits = np.arange(q_min, q_max)
    tiempos = matriz_tiempos[L_index, :q_max - q_min]

    plt.plot(qubits, tiempos, 'o', label=f'L={L}')
    if pendiente[L_index] > 0:
        ajuste = pendiente[L_index] * qubits + interceptos[L_index]
        plt.plot(qubits, ajuste, '--', color='grey')

plt.xlabel('Número de qubits')
plt.ylabel('Tiempo (s)')
plt.title('Tiempo de simulación vs Número de qubits')
plt.grid()
plt.legend()
plt.show()

# --- Gráfica: Pendiente vs L ---
plt.figure(figsize=(15, 6))
L_vals_plot = np.arange(L_min, L_max_grafica + 1)
pendiente_plot = pendiente[:L_max_grafica + 1 - L_min]
plt.plot(L_vals_plot, pendiente_plot, 'o-', color='blue', label='Pendiente')
plt.xlabel('Número de capas L')
plt.ylabel('Pendiente')
plt.title('Pendiente del ajuste lineal vs Número de capas L')
plt.grid()
plt.legend()
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
plt.plot(L_vals_fit, log2_pend, 'o', label='Datos')
plt.plot(L_vals_fit, a * L_vals_fit + b, '-', label=f'Ajuste: {a:.2f}·L + {b:.2f}')
plt.xlabel('Número de capas L')
plt.ylabel('log₂(Pendiente)')
plt.title('Ajuste logarítmico de la pendiente vs L')
plt.grid()
plt.legend()
plt.show()
