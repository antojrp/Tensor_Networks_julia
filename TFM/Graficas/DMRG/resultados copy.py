import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe  # Integral elíptica completa de segunda especie

plt.rcParams.update({'font.size': 17})
# Valores de N
N_values = [100, 500, 1000, 1500]

# Inicializar contenedor de datos
all_data = {}

# Cargar datos desde archivos
for N in N_values:
    file = f"resultados/DMRG_L{N}.txt"
    
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

# --- Gráficas de entropía y dimensión en una misma figura (lado a lado) ---
plt.rcParams.update({'font.size': 18})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=False)

# Entropía
for N, data in all_data.items():
    ax1.errorbar(
        data["gamma"],
        data["S"],
        yerr=np.sqrt(data["varS"])/np.sqrt(40),
        label=f"N={N}",
        capsize=3
    )
ax1.set_xlabel("$\\Gamma$")
ax1.set_ylabel("S$_{VN}$")
ax1.set_title("Von Neumann Entropy vs Gamma")
ax1.grid()
ax1.legend(ncols=1)

# Dimensión
for N, data in all_data.items():
    ax2.errorbar(
        data["gamma"],
        data["dimension"],
        yerr=np.sqrt(data["vardimension"])/np.sqrt(40),
        label=f"N={N}",
        capsize=3
    )
ax2.set_xlabel("$\\Gamma$")
ax2.set_ylabel("D")
ax2.set_title("Maximum Bond Dimension vs Gamma")
ax2.grid()
ax2.legend(ncols=1, loc='lower right')

plt.tight_layout()
plt.savefig("DMRG_SVN_and_Dimension.pdf")
plt.show()

# --- Gráfica de energía con comparación analítica ---
plt.rcParams.update({'font.size': 17})
plt.figure(figsize=(10, 6))
for N, data in all_data.items():
    plt.errorbar(
        data["gamma"],
        -data["E"]/N,
        yerr=np.sqrt(data["varE"])/N,
        label=f"N={N}",
        capsize=3
    )

# Solución analítica (Pfeuty)
if all_data:
    gamma_vals = data["gamma"]
    h_vals = 1 / (2 * gamma_vals)
    k_vals = 4 * h_vals / (1 + h_vals)**2
    E_analytic = -(gamma_vals / np.pi) * (1 + h_vals) * ellipe(k_vals)
    plt.plot(gamma_vals, -E_analytic, '--', color='black', label="Analytic solution", zorder=5)

plt.xlabel("$\\Gamma$")
plt.ylabel("-E/N")
plt.title("Energy as a function of Gamma")
plt.grid()
plt.legend(ncols=2)
plt.tight_layout()
plt.savefig("DMRG_Energy.pdf")
plt.show()
