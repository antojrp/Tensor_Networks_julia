import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 17}) 
file = "tamaño.txt"  
data = pd.read_csv(file, sep="\t")
data.set_index('N', inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')

# Plot memory size as a function of bond dimension for fixed N, with fit a*35*D^2
def plot_fixed_qubits_with_fit(data, N_fixed):
    if N_fixed not in data.index:
        print(f"N={N_fixed} is not in the data")
        return

    y = data.loc[N_fixed].dropna()
    x_D = y.index.astype(int)  # Log-scale D (D = 2^x)
    D_values = 2 ** x_D  # Actual bond dimension

    def model(D, a):
        return (a * 35 * D)*D

    popt, _ = curve_fit(model, D_values, y.values, maxfev=10000)
    a_fit = popt[0]
    y_fit = model(D_values, a_fit)

    residuals = y.values - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y.values - np.mean(y.values))**2)
    r2 = 1 - (ss_res / ss_tot)

    plt.figure()
    plt.plot(x_D, y_fit, '--', color='grey', label=f"Fit: a·D², $R^2$={r2:.4f}")
    plt.plot(x_D, y.values, 'o', label="Data")
    plt.xlabel("log$_2$(D)")
    plt.ylabel("Memory size (GB)")
    plt.title(f"Memory size vs D (N = {N_fixed})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"memory_vs_bond_N_{N_fixed}.pdf")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_fixed_qubits_with_fit2(data, N_fixed):
    if N_fixed not in data.index:
        print(f"N={N_fixed} is not in the data")
        return

    y = data.loc[N_fixed].dropna()
    x_D_log = y.index.astype(int)           # This is log2(D)
    D_values = 2 ** x_D_log                 # Actual bond dimensions
    log_D = np.log2(D_values)
    log_memory = np.log2(y.values)          # Use log2 for consistency

    # Linear fit in log-log space: log2(Memory) = slope * log2(D) + intercept
    slope, intercept, r_value, _, _ = linregress(log_D, log_memory)
    y_fit_log = slope * log_D + intercept
    print(f"Fit: slope = {slope:.4f} ± {linregress(log_D, log_memory).stderr:.4f}, intercept = {intercept:.4f}, R² = {r_value**2:.4f}")
    y_fit = 2 ** y_fit_log

    plt.figure()
    plt.plot(log_D, y_fit_log, '--', color='grey', label=f"Linear Fit, $R^2$={r_value**2:.4f}")
    plt.plot(log_D, log_memory, 'o', label="Data")
    plt.xlabel("log$_2$(D)")
    plt.ylabel("log$_2$(Memory)")
    plt.title(f"log$_2$(Memory) vs log$_2$(D) (N = {N_fixed})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"memory_vs_bond_loglogfit_N_{N_fixed}_2.pdf")
    plt.show()



# Plot memory size as a function of qubits for fixed D, with linear fit a*N + b
def plot_fixed_bond_with_fit(data, D_fixed):
    D_fixed = str(D_fixed)
    if D_fixed not in data.columns:
        print(f"D={D_fixed} is not in the data")
        return

    y = data[D_fixed].dropna()
    x = y.index.astype(int)

    def model(N, a, b):
        return a * N + b

    popt, _ = curve_fit(model, x, y.values)
    a_fit, b_fit = popt
    y_fit = model(x, a_fit, b_fit)

    residuals = y.values - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y.values - np.mean(y.values))**2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"Fit: a = {a_fit}, b = {b_fit}")

    plt.figure()
    plt.plot(x, y_fit, '--', color='grey', label=f"Linear Fit, $R^2$={r2:.2f}")
    plt.plot(x, y.values, 'o', label="Data")
    plt.xlabel("N")
    plt.ylabel("Memory size (GB)")
    plt.title(f"Memory size vs N (log$_2$D = {D_fixed})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"memory_vs_qubits_D_{D_fixed}.pdf")
    plt.show()

# Example usage
#plot_fixed_qubits_with_fit(data, 35)
#plot_fixed_bond_with_fit(data, 10)
plot_fixed_qubits_with_fit2(data, 35)
