
import numpy as np
import matplotlib.pyplot as plt

N=20

# Definir parámetros para el ejemplo
N_A = 2**(N//2)  # Número de modos en A
N_B = 2**(N//2)  # Número de modos en B

# Definir a y b
a = (1/np.sqrt(N_A) - 1/np.sqrt(N_B))**2
b = (1/np.sqrt(N_A) + 1/np.sqrt(N_B))**2

# Definir la función ω_s(p)
def omega_s(p, N_A, N_B, a, b):
    if a <= p <= b:
        return (N_A * N_B / (2 * np.pi)) * np.sqrt((p - a) * (b - p)) / p
    else:
        return 0

# Crear valores de p en el rango relevante
p_values = np.linspace(0, 2*b, 1000)
omega_values = [omega_s(p, N_A, N_B, a, b) for p in p_values]

# Graficar la función
plt.figure(figsize=(8, 5))
plt.plot(p_values, omega_values, label=r'$\omega_s(p)$', color='b')
plt.axvline(a, color='r', linestyle='--', label=r'$a$')
plt.axvline(b, color='g', linestyle='--', label=r'$b$')
plt.xlabel(r'$p$')
plt.ylabel(r'$\omega_s(p)$')
plt.title('Distribución de Coeficientes de Schmidt')
plt.legend()
plt.grid()
plt.show()
