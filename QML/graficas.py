import matplotlib.pyplot as plt
import numpy as np

# ============================
# Configuración
# ============================

filename = "./results/Sonar_2.dat"

num_runs = 25
# Diccionarios anidados
accuracy = {}
accuracy_var = {}
renyi2 = {}
renyi2_var = {}

current_bond = None
current_L = None

# ============================
# Parseo del archivo
# ============================

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue

        if s.startswith("Bond:"):
            current_bond = float(s.split(":")[1])
            accuracy.setdefault(current_bond, {})
            accuracy_var.setdefault(current_bond, {})
            renyi2.setdefault(current_bond, {})
            renyi2_var.setdefault(current_bond, {})

        elif s.startswith("L ="):
            current_L = int(s.split("=")[1])

        elif s.startswith("Mean accuracy:"):
            val = float(s.split(":")[1])
            accuracy[current_bond][current_L] = val

        elif s.startswith("Variance of accuracy:"):
            val = float(s.split(":")[1])
            accuracy_var[current_bond][current_L] = val
           
        elif s.startswith("Mean Renyi2:"):
            val = float(s.split(":")[1])
            renyi2[current_bond][current_L] = val

        elif s.startswith("Var  Renyi2:"):
            val = float(s.split(":")[1])
            renyi2_var[current_bond][current_L] = val

# ============================
# Ejes
# ============================

bonds = sorted(accuracy.keys())
Ls = sorted({L for b in bonds for L in accuracy[b].keys()})

print("Bonds:", bonds)
print("Ls:", Ls)

# ============================
# 1) Accuracy vs L con errorbars
# ============================

plt.figure(figsize=(7, 5))

for b in bonds:
    ys = np.array([accuracy[b][L] for L in Ls])
    err = np.sqrt(np.array([accuracy_var[b][L] for L in Ls]))/num_runs
    plt.errorbar(Ls, ys, yerr=err, marker="o", capsize=4, label=f"bond = {b}")

plt.xlabel("L")
plt.ylabel("Mean accuracy")
plt.title("Accuracy vs L")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 2) Accuracy vs bond con errorbars
# ============================

plt.figure(figsize=(7, 5))

for L in Ls:
    xs = np.array(bonds)
    ys = np.array([accuracy[b][L] for b in bonds])
    err = np.sqrt(np.array([accuracy_var[b][L] for b in bonds]))/num_runs
    plt.errorbar(xs, ys, yerr=err, marker="o", capsize=4, label=f"L = {L}")

plt.xlabel("bond")
plt.ylabel("Mean accuracy")
plt.title("Accuracy vs bond")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 3) Renyi2 vs L con errorbars
# ============================

plt.figure(figsize=(7, 5))

for b in bonds:
    ys = np.array([renyi2[b][L] for L in Ls])
    err = np.sqrt(np.array([renyi2_var[b][L] for L in Ls]))/num_runs
    plt.errorbar(Ls, ys, yerr=err, marker="o", capsize=4, label=f"bond = {b}")

plt.xlabel("L")
plt.ylabel("Mean Renyi2")
plt.title("Renyi2 vs L")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
