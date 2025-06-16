import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe  # Integral elíptica completa de segunda especie

plt.rcParams.update({'font.size': 16})

# Valores de N
N_values = [20]

# Inicializar contenedor de datos
all_data = {}

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

# Crear la gráfica de energía con eje adicional para la dimensión
for N, data in all_data.items():
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Solución analítica
    gamma_vals = data["gamma"]
    h_vals = 1 / (2 * gamma_vals)  # h = J/(2Γ), con J=1
    k_vals = 4 * h_vals / (1 + h_vals)**2
    E_analytic = -( gamma_vals / np.pi) * (1 + h_vals) * ellipe(k_vals)
    ax1.plot(gamma_vals, -E_analytic, '--', color='black', label="Analytic (Pfeuty)", zorder=5)

    # Eje principal: energía por sitio vs gamma
    ax1.errorbar(
        data["gamma"],
        -data["E"] / N,
        yerr=np.sqrt(data["varE"]) / N,
        label=f"DMRG N={N}",
        capsize=3,
        fmt='o',           # Solo puntos, sin línea
        linestyle='None',   # Asegura que no haya línea
        zorder=10
    )
    ax1.set_xlabel("$\Gamma$")
    ax1.set_ylabel("-E/N")
    ax1.grid()

    # Eje secundario: dimensión en la parte superior
    ax2 = ax1.twiny()

    # Emparejar la escala del eje superior con el eje inferior
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(data["gamma"])
    ax2.set_xticklabels([f"{d:.1f}" for d in data["dimension"]], rotation=45)
    ax2.set_xlabel("D")

    ax1.set_title("Energy per site as a function of $\Gamma$ and Dimension D (J=1)")
    ax1.legend(loc="best")

    plt.tight_layout()
    plt.savefig(f"graficas/DMRG_L{N}.pdf")
    plt.show()
