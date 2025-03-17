import numpy as np
import matplotlib.pyplot as plt

def parse_dmrg_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = {}
    current_qubits = None
    current_dimension = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Numero de qubits"): 
            current_qubits = int(line.split(": ")[1])
        elif line.startswith("Dimension"): 
            current_dimension = int(line.split(": ")[1])
            data[(current_qubits, current_dimension)] = {"time": [], "var_time": []}
        elif line.startswith("Sweep") or line.startswith("Total"): 
            continue
        elif line.startswith("Varianza"): 
            data[(current_qubits, current_dimension)]["var_total"] = float(line.split("= ")[1])
        else:
            parts = line.split()
            if len(parts) == 3:
                _, time, var_time = map(float, parts)
                data[(current_qubits, current_dimension)]["time"].append(time)
                data[(current_qubits, current_dimension)]["var_time"].append(var_time)
    
    return data

def create_matrices(data):
    dimensions = sorted(set(dim for _, dim in data.keys()))
    qubits = sorted(set(qb for qb, _ in data.keys()))

    num_sweeps = 10
    
    time_matrix = np.zeros((len(qubits), len(dimensions), num_sweeps))
    var_time_matrix = np.zeros_like(time_matrix)
    total_time = []
    total_variance = []
    
    for i, ((qubit, dim), values) in enumerate(data.items()):
        time_matrix[qubits.index(qubit), dimensions.index(dim), :] = values["time"]
        var_time_matrix[qubits.index(qubit), dimensions.index(dim), :] = values["var_time"]
        total_time.append(sum(values["time"]))
        total_variance.append(values["var_total"])
    
    return time_matrix, var_time_matrix, np.array(total_time), np.array(total_variance), dimensions, qubits


file_path = "../../Programas/resultados/DMRG_tiempo_L100_D100_t15.txt"
data = parse_dmrg_file(file_path)
time_matrix, var_time_matrix, total_time, total_variance, dimensions, qubits = create_matrices(data)


plt.figure(figsize=(8, 5))
plt.plot(dimensions, total_time, marker='o', linestyle='-', color='b', label='Tiempo Total')
plt.xlabel('D')
plt.ylabel('Tiempo Total')
plt.title('Tiempo Total en función de la Dimensión')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
sweeps = np.arange(1, 11)
for i, dim in enumerate(dimensions):
    plt.plot(sweeps, time_matrix[0, i, :], marker='o', linestyle='-', label=f'D={dim}')
plt.xlabel('Sweep')
plt.ylabel('Energía')
plt.title('Energía en función de Sweep para diferentes Dimensiones')
plt.legend(ncols=2)
plt.grid()
plt.show()
