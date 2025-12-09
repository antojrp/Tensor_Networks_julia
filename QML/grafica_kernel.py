from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN
dataset_name = "sonar"
base_folder = Path("Kernels")
# ==========================================

folder = base_folder / dataset_name
csv_files = sorted(folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No hay .csv en {folder}")

# Extraer L y k desde nombres tipo dataset_L_<L>_log(D)_<k>.csv
LD_list = []
for f in csv_files:
    parts = f.stem.split("_")
    try:
        idx_L = parts.index("L")
        L = int(parts[idx_L + 1])

               # log(D)_k  → D = 2^k
        idx_k = parts.index("log(D)")
        k = int(parts[idx_k + 1])
        D = 2**k

        LD_list.append((L, D, f))
    except Exception:
        print("Ignorado:", f.name)

if not LD_list:
    raise RuntimeError("No hay archivos válidos con L y D")

Ls = sorted({L for (L, D, f) in LD_list})
Ds = sorted({D for (L, D, f) in LD_list})
Ds_desc = list(reversed(Ds))   # arriba D grande, abajo D pequeño

nL = len(Ls)
nD = len(Ds)

file_map = {(L, D): f for (L, D, f) in LD_list}

# ===================================================
# FIGURA COMPRIMIDA
# ===================================================

fig, axes = plt.subplots(
    nD,
    nL,
    figsize=(2.6 * nL + 2, 2.6 * nD + 2),
    squeeze=False
)

plt.subplots_adjust(
    left=0.12,
    right=0.88,
    bottom=0.18,
    top=0.88,
    wspace=0.10,
    hspace=0.10
)

vmin, vmax = 0.0, 1.0
ims = []

# Pintamos matrices
for col, L in enumerate(Ls):
    for row, D in enumerate(Ds_desc):
        ax = axes[row, col]
        f = file_map.get((L, D))
        if f is None:
            ax.axis("off")
            continue

        K = np.loadtxt(f, delimiter=",")

        im = ax.imshow(
            K.T,
            origin="upper",
            aspect="equal",
            vmin=vmin,
            vmax=vmax
        )
        ims.append(im)

        ax.set_xticks([])
        ax.set_yticks([])

# Colorbar
cbar = fig.colorbar(
    ims[0],
    ax=axes.ravel().tolist(),
    fraction=0.035,
    pad=0.01
)
cbar.set_label("K_ij", fontsize=11)

# ===================================================
# EJE GLOBAL PARA FLECHAS Y TEXTOS
# ===================================================

ax_global = fig.add_axes([0, 0, 1, 1])
ax_global.set_axis_off()

# Flecha L → (abajo)
ax_global.annotate(
    "",
    xy=(0.88, 0.13),
    xytext=(0.12, 0.13),
    arrowprops=dict(arrowstyle="->", lw=2),
    annotation_clip=False
)

ax_global.text(
    0.50, 0.16,
    "Número de capas L",
    ha="center",
    va="center",
    fontsize=14
)

for col, L in enumerate(Ls):
    x = 0.12 + (col + 0.5) * ((0.88 - 0.12) / nL)
    ax_global.text(
        x, 0.08,
        str(L),
        ha="center",
        va="center",
        fontsize=11
    )

# Flecha D ↑ (izquierda)
ax_global.annotate(
    "",
    xy=(0.12, 0.88),     # punta (D grande, arriba)
    xytext=(0.12, 0.18), # origen (D pequeño, abajo)
    arrowprops=dict(arrowstyle="->", lw=2),
    annotation_clip=False
)

ax_global.text(
    0.07, 0.52,
    "Bond dimension D",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=14
)

# >>> AQUÍ ESTABA EL FALLO: cálculo de y <<<
y_top = 0.88
y_bottom = 0.18
step = (y_top - y_bottom) / nD

for row, D in enumerate(Ds_desc):
    # row = 0 es la fila de arriba, así que empezamos desde y_top
    y = y_top - (row + 0.5) * step
    ax_global.text(
        0.055, y,
        str(D),
        ha="center",
        va="center",
        fontsize=11
    )

fig.suptitle(f"Kernels: {dataset_name}", fontsize=16, y=0.95)

plt.show()
