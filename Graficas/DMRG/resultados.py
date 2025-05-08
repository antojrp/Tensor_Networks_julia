import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe  # Para la integral elíptica completa de segunda especie

# Especificar los valores de N que deseas leer
N_values = [50,100]

# Inicializar listas para almacenar datos
all_data = {}

for N in N_values:
    file = "resultados/DMRG_L"+str(N)+".txt"
    
    try:
        data = np.loadtxt(file, skiprows=1)
        gamma, E, varE, T, varT, S, varS  = data.T
        
        all_data[N] = {
            "gamma": gamma,
            "E": E,
            "varE": varE,
            "T": T,
            "varT": varT,
            "S": S,
            "varS": varS
        }
    except FileNotFoundError:
        print(f"Archivo no encontrado: {file}")

# Crear la gráfica del tiempo
plt.figure(figsize=(10, 6))
for N, data in all_data.items():
    plt.errorbar(data["gamma"], data["T"], yerr=np.sqrt(data["varT"]), label=f"N={N}", capsize=3)
plt.xlabel("Gamma")
plt.ylabel("Tiempo (T)")
plt.title("Tiempo vs Gamma con Varianza como Error")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for N, data in all_data.items():
    plt.errorbar(data["gamma"], data["S"], yerr=np.sqrt(data["varS"]), label=f"N={N}", capsize=3)
plt.xlabel("Gamma")
plt.ylabel("Entropia (T)")
plt.title("Entropia vs Gamma con Varianza como Error")
plt.legend()
plt.grid()
plt.show()

# Crear la gráfica de la energía con solución analítica
plt.figure(figsize=(10, 6))
for N, data in all_data.items():
    plt.errorbar(data["gamma"], -data["E"]/N, yerr=np.sqrt(data["varE"])/N, label=f"DMRG N={N}", capsize=3)


# Solución analítica
gamma_vals = data["gamma"]
h_vals = 1 / (2 * gamma_vals)  # h = J/(2Γ), con J=1
k_vals = 4 * h_vals / (1 + h_vals)**2
E_analytic = -( gamma_vals / np.pi) * (1 + h_vals) * ellipe(k_vals)

plt.plot(gamma_vals, -E_analytic, '--', color='black', label="Solución analítica (Pfeuty)", zorder=5)
plt.xlabel("Gamma")
plt.ylabel("Energía por sitio (-E/N)")
plt.title("Energía vs Gamma con solución analítica")
plt.legend()
plt.grid()
plt.show()
