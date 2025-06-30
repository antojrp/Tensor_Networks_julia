import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe  # Integral elíptica completa de segunda especie

plt.rcParams.update({'font.size': 17})

# Valores de N
N_values = [100, 500, 1500, 2500]

color=["#0b5269",'#108ab1','#06d7a0','#ffd167',"#c53333"]

# Inicializar contenedor de datos
all_data = {}

# Cargar datos desde archivos
for N in N_values:
    file = f"../../Programas/resultados/DMRG_L{N}.txt"
    
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

# --- Gráficas de Entropía y Dimensión en cuadrícula 2x2 ---
plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, N in enumerate(N_values):
    if N in all_data:
        data = all_data[N]
        ax = axes[idx]
        
        ax2 = ax.twinx()  # Segundo eje y para la dimensión

        # Entropía con barras de error
        ax.errorbar(
            data["gamma"],
            data["S"],
            yerr=np.sqrt(data["varS"])/np.sqrt(40),
            label="S$_{VN}$",
            color=color[0],
            capsize=3
        )
        ax.set_ylabel("S$_{VN}$", color=color[0])
        ax.tick_params(axis='y', labelcolor= color[0])

        # Dimensión con barras de error
        ax2.errorbar(
            data["gamma"],
            data["dimension"],
            yerr=np.sqrt(data["vardimension"])/np.sqrt(40),
            label="D",
            color=color[4],
            linestyle='--',
            capsize=3
        )
        ax2.set_ylabel("D", color= color[4])
        ax2.tick_params(axis='y', labelcolor= color[4])

        ax.set_xlabel("$\\Gamma$")
        ax.set_title(f"N = {N}")
        ax.grid()


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("DMRG_Entropy_and_Dimension_Grid.pdf")
plt.show()

# --- Gráficas de Energía en cuadrícula 2x2 con comparación analítica ---
plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, N in enumerate(N_values):
    if N in all_data:
        data = all_data[N]
        gamma_vals = data["gamma"]
        E_vals = data["E"]
        varE_vals = data["varE"]

        ax = axes[idx]

        # Datos DMRG
        ax.errorbar(
            gamma_vals,
            -E_vals/N,
            yerr=np.sqrt(varE_vals)/N,
            label="DMRG",
            color=color[(idx+1)%len(color)],
            capsize=3
        )

        # Solución analítica (Pfeuty)
        h_vals = 1 / (2 * gamma_vals)
        k_vals = 4 * h_vals / (1 + h_vals)**2
        E_analytic = -(gamma_vals / np.pi) * (1 + h_vals) * ellipe(k_vals)
        ax.plot(gamma_vals, -E_analytic, '--', color='black', label="Analytic solution", zorder=5)

        ax.set_xlabel("$\\Gamma$")
        ax.set_ylabel("-E/N")
        ax.set_title(f"N = {N}")
        ax.grid()
        ax.legend()


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("DMRG_Energy_Grid.pdf")
plt.show()
