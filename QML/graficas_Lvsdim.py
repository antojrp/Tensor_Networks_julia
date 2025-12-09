import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ==========================================================
# CONFIGURACIÓN
dataset_names = ["Breast_cancer", "Ionosphere", "Sonar","Arrhythmia"]
base_results_folder = Path("results")
pattern = "*.out"

dataset_thresholds = {
    "Breast_cancer": 0.94,
    "Ionosphere": 0.89,
    "Sonar": 0.84,
    "Arrhythmia": 0.70,
}

default_threshold = 0.94
# ==========================================================

rows = []

for dname in dataset_names:
    dataset_folder = base_results_folder / dname

    c_folders = sorted(
        [p for p in dataset_folder.iterdir() if p.is_dir() and p.name.startswith("C=")],
        key=lambda p: float(re.search(r"C=([\d.]+)", p.name).group(1))
    )

    if not c_folders:
        print(f"Advertencia: no se encontraron carpetas C=x en {dataset_folder}")
        continue

    for c_folder in c_folders:
        mC = re.search(r"C=([\d.]+)", c_folder.name)
        if not mC:
            continue

        C_value = float(mC.group(1))

        for file_path in c_folder.glob(pattern):
            with open(file_path, "r") as f:
                txt = f.read()

            mL = re.search(r"L\s*=\s*(\d+)", txt)
            if not mL:
                continue
            L = int(mL.group(1))

            current_dim = None

            for line in txt.splitlines():
                m_dim = re.search(r"Max dim\s*=\s*(\d+)", line)
                if m_dim:
                    current_dim = int(m_dim.group(1))
                    continue

                m_acc = re.search(r"Mean accuracy:\s*([0-9.]+)", line)
                if m_acc and current_dim is not None:
                    rows.append(dict(
                        dataset=dname,
                        C=C_value,
                        L=L,
                        dim=current_dim,
                        accuracy=float(m_acc.group(1))
                    ))

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No hay datos en ningún dataset")

datasets = sorted(df["dataset"].unique())
nD = len(datasets)

C_per_dataset = {
    d: sorted(df[df["dataset"] == d]["C"].unique())
    for d in datasets
}
max_nC = max(len(v) for v in C_per_dataset.values())

fig, axes = plt.subplots(
    nD, max_nC,
    figsize=(3 * max_nC + 1.2, 3.2 * nD),
    sharey="row",
    gridspec_kw={"wspace": 0.15, "hspace": 0.45}
)

if nD == 1:
    axes = np.array([axes])
if max_nC == 1:
    axes = axes.reshape(nD, 1)

for i, dname in enumerate(datasets):
    df_d = df[df["dataset"] == dname]
    th = dataset_thresholds.get(dname, default_threshold)

    local_min = th
    local_max = df_d["accuracy"].max()

    cmap = plt.cm.coolwarm
    norm = colors.Normalize(vmin=local_min, vmax=local_max)
    base_color = cmap(norm(th))

    C_values = C_per_dataset[dname]
    nC = len(C_values)

    for j in range(max_nC):
        ax = axes[i, j]

        if j >= nC:
            ax.axis("off")
            continue

        C_val = C_values[j]
        df_C = df_d[df_d["C"] == C_val]

        low = df_C[df_C["accuracy"] < th]
        high = df_C[df_C["accuracy"] >= th]

        # === scatter con L en X y dim en Y ===
        if not low.empty:
            ax.scatter(
                low["L"], np.log2(low["dim"]),
                c=[base_color],
                s=80,
                marker="o"
            )

        if not high.empty:
            sc = ax.scatter(
                df_C["L"], np.log2(df_C["dim"]),
                c=df_C["accuracy"],
                cmap=cmap,
                norm=norm,
                s=80,
                marker="o"
            )

        # === ticks del eje Y en base a dim reales ===
        dims_sorted = sorted(df_C["dim"].unique())
        yticks = np.log2(dims_sorted)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(d) for d in dims_sorted], fontsize=8)

        # === eje X es L tal cual ===
        Ls_sorted = sorted(df_C["L"].unique())
        ax.set_xticks(Ls_sorted)
        ax.set_xticklabels([str(L) for L in Ls_sorted], fontsize=8)

        ax.set_xlabel("L", fontsize=9)
        ax.set_ylabel("dim (real)", fontsize=9)
        ax.set_title(f"C = {C_val}", fontsize=10)
    axes[i, 0].set_ylabel(f"{dname}\nL", fontsize=10)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[i, :nC].ravel().tolist(),
        location="right",
        fraction=0.03,
        pad=0.02
    )
    cbar.set_label(f"accuracy ({dname})", fontsize=10)

fig.suptitle("Accuracy codificada en color por dataset", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
