import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe  # Integral elíptica completa de segunda especie

# Valores de N
N_values = [2, 3, 4, 5, 6]

# Inicializar contenedor de datos
all_data = {}

# Cargar datos desde archivos
for N in N_values:
    file = f"resultados/DMRG_L50_{N}.txt"
    
    try:
        data = np.loadtxt(file, skiprows=1)
        gamma, E, varE, T, varT, S, varS, dimension, vardimension = data.T
        
        all_data[N] = {
            "gamma": gamma,
            "E": E,
            "varE": varE,
            "T": T,
            "varT": varT,
            "S": S,
            "varS": varS,
            "dimension": dimension,
            "vardimension": vardimension
        }
    except FileNotFoundError:
        print(f"Archivo no encontrado: {file}")

# --- Gráfica de entropía ---
plt.figure(figsize=(10, 6))
for N, data in all_data.items():
    plt.errorbar(
        data["gamma"],
        data["S"],
        yerr=np.sqrt(data["varS"]),
        label=f"N={N}",
        capsize=3
    )
plt.xlabel("Gamma")
plt.ylabel("Entropía de entrelazamiento")
plt.title("Entropía de entrelazamiento vs Gamma")
plt.grid()
plt.legend(title="Tamaño N")
plt.tight_layout()
plt.show()

# --- Gráfica de energía con comparación analítica ---
plt.figure(figsize=(10, 6))
for N, data in all_data.items():
    plt.errorbar(
        data["gamma"],
        -data["E"]/50,
        yerr=np.sqrt(data["varE"]),
        label=f"N={N}",
        capsize=3
    )

# Solución analítica (Pfeuty)
if all_data:
    gamma_vals = data["gamma"]
    h_vals = 1 / (2 * gamma_vals)
    k_vals = 4 * h_vals / (1 + h_vals)**2
    E_analytic = -(gamma_vals / np.pi) * (1 + h_vals) * ellipe(k_vals)
    plt.plot(gamma_vals, -E_analytic, '--', color='black', label="Solución analítica (Pfeuty)", zorder=5)

plt.xlabel("Gamma")
plt.ylabel("Energía")
plt.title("Energía vs Gamma")
plt.grid()
plt.legend(title="Tamaño N")
plt.tight_layout()
plt.show()

# --- Gráfica de dimensión efectiva ---
plt.figure(figsize=(10, 6))
for N, data in all_data.items():
    plt.errorbar(
        data["gamma"],
        data["dimension"],
        yerr=np.sqrt(data["vardimension"]),
        label=f"N={N}",
        capsize=3
    )
plt.xlabel("Gamma")
plt.ylabel("Dimensión del espacio efectivo")
plt.title("Dimensión efectiva vs Gamma")
plt.grid()
plt.legend(title="Tamaño N")
plt.tight_layout()
plt.show()

# --- Gráfica de error relativo de energía ---
plt.figure(figsize=(10, 6))
for N, data in all_data.items():
    gamma_vals = data["gamma"]
    h_vals = 1 / (2 * gamma_vals)
    k_vals = 4 * h_vals / (1 + h_vals)**2
    E_analytic = -(gamma_vals / np.pi) * (1 + h_vals) * ellipe(k_vals)

    error_relativo = np.abs(-data["E"]/50 - (-E_analytic)) / np.abs(E_analytic)
    
    plt.plot(gamma_vals, error_relativo, label=f"N={N}")

plt.xlabel("Gamma")
plt.ylabel("Error relativo de la energía")
plt.title("Error relativo vs Gamma (Energía DMRG vs Pfeuty)")
plt.yscale("log")  # Escala logarítmica para ver mejor las diferencias pequeñas
plt.grid(True, which="both")
plt.legend(title="Tamaño N")
plt.tight_layout()
plt.show()
