import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Datos proporcionados por el usuario
data = pd.read_csv('../../resultados/Random_coeficientes_70.txt', delim_whitespace=True, header=None)

# Crear histograma normalizado para la distribución de probabilidad
num_bins = 80  # Número de intervalos
counts, bins = np.histogram(data, bins=num_bins, density=True)

# Graficar la distribución de probabilidad
plt.figure(figsize=(8, 5))
plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.7)
plt.xlabel("Valores")
plt.ylabel("Probabilidad")
plt.title("Distribución de Probabilidad")
plt.show()

def varianza(datos):
    if len(datos) == 0:
        return None  # Evitar división por cero
    media = sum(datos) / len(datos)
    return sum((x - media) ** 2 for x in datos) / len(datos)

print(varianza(data))