import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# ==========================================================
# CONFIGURACIÓN
folder = Path("results/Sonar")   # ejemplo: Path("resultados_sonar/")
pattern = "*.out"                    # patrón de archivos
threshold = 0.75                     # corte inferior del colormap
# ==========================================================


rows = []

# Leer archivos
for file_path in folder.glob(pattern):
    with open(file_path, "r") as f:
        txt = f.read()

    # Detectar L global del archivo
    mL = re.search(r"L\s*=\s*(\d+)", txt)
    if not mL:
        continue
    L = int(mL.group(1))

    current_dim = None

    # Extraer Max dim y luego Mean accuracy asociado
    for line in txt.splitlines():
        m_dim = re.search(r"Max dim\s*=\s*(\d+)", line)
        if m_dim:
            current_dim = int(m_dim.group(1))
            continue

        m_acc = re.search(r"Mean accuracy:\s*([0-9.]+)", line)
        if m_acc and current_dim is not None:
            acc = float(m_acc.group(1))
            rows.append(dict(L=L, dim=current_dim, accuracy=acc))


# Convertir a DataFrame
df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError(f"No se encontraron datos en {folder}")


# Separar por umbral
low = df[df["accuracy"] < threshold]
high = df[df["accuracy"] >= threshold]

# Colormap azul→rojo
cmap = plt.cm.coolwarm
norm = colors.Normalize(vmin=threshold, vmax=df["accuracy"].max())
base_color = cmap(norm(threshold))  # color mínimo del colormap


# ==================== PLOT ====================
plt.figure(figsize=(10, 6))

# low accuracy en color mínimo del colormap
plt.scatter(
    np.log2(low["dim"]), low["L"],
    c=[base_color],
    s=150,
    marker='o',
    label=f"accuracy < {threshold}"
)

# high accuracy con colormap
sc = plt.scatter(
    np.log2(high["dim"]), high["L"],
    c=high["accuracy"],
    cmap=cmap,
    norm=norm,
    s=150,
    marker='o'
)


# Eje X etiquetado con dim real
dims_sorted = sorted(df["dim"].unique())
xticks = np.log2(dims_sorted)
plt.xticks(xticks, [str(d) for d in dims_sorted])


plt.xlabel("dim (log2, etiquetas en dimensión real)")
plt.ylabel("L")
plt.title("Accuracy codificada en color (azul→rojo)")

cbar = plt.colorbar(sc)
cbar.set_label("accuracy")

plt.legend()
plt.tight_layout()
plt.show()
