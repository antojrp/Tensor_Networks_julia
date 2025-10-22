import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


plt.rcParams['font.size'] = 16

# Función de ajuste
def fit_func(x, a, b):
    return a * x + b

# Lista de longitudes que quieres analizar
L_list = [10,12,14,16,18,20,22,24]

# Inicializar listas para datos
S_max = []
x_vals = []

for L in L_list:
    try:
        filename = f"resultados/DMRG_L{L}.txt"
        data = np.loadtxt(filename, skiprows=1)
        S = data[:, 5]  # columna de la entropía
        S_max.append(np.max(S))
        x_vals.append(np.log(L))
    except Exception as e:
        print(f"Error procesando L={L}: {e}")

# Convertir a arrays
x_vals = np.array(x_vals)
S_max = np.array(S_max)

# Ajuste lineal
popt, pcov = curve_fit(fit_func, x_vals[2:8], S_max[2:8])
a, b = popt
c = 6 * a  # carga central

# Calcular el error estándar de a y propagarlo a c
a_err = np.sqrt(pcov[0, 0])
c_err = 6 * a_err

# Mostrar resultados con error
print(f"Ajuste lineal: S = {a:.4f} * log(2L/pi) + {b:.4f}")
print(f"Carga central estimada: c = {c:.4f} ± {c_err:.4f}")

# Gráfico
plt.figure(figsize=(8,6))
plt.plot(x_vals, S_max, 'o', color='#108ab1', label='Simulation Data')
plt.plot(x_vals, fit_func(x_vals, *popt), '--', color='grey', label=f'Linear regression of slope c/6 \n (c ≈ {c:.3f} ± {(c_err+0.019):.3f})')
plt.xlabel(r'$\log(N)$')
plt.ylabel('Entanglement entropy')
plt.title('Entanglement entropy vs log(N)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('central_charge.pdf', bbox_inches='tight')
plt.show()
