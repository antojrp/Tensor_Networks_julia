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
            data[(current_qubits, current_dimension)] = {"energy": [], "var_energy": []}
        elif line.startswith("Sweep"):
            continue
        else:
            parts = line.split()
            if len(parts) == 3:
                _, energy, var_energy = map(float, parts)
                data[(current_qubits, current_dimension)]["energy"].append(energy)
                data[(current_qubits, current_dimension)]["var_energy"].append(var_energy)
    
    return data

def create_matrices(data):
    dimensions = sorted(set(dim for _, dim in data.keys()))
    qubits = sorted(set(qb for qb, _ in data.keys()))
    num_sweeps = 10
    
    energy_matrix = np.zeros((len(data), len(dimensions), num_sweeps))
    var_energy_matrix = np.zeros_like(energy_matrix)
    final_energy = []
    total_variance = []
    
    for i, ((qubit, dim), values) in enumerate(data.items()):
        energy_matrix[qubits.index(qubit), dimensions.index(dim), :] = values["energy"]
        var_energy_matrix[qubits.index(qubit), dimensions.index(dim), :] = values["var_energy"]
        final_energy.append(values["energy"][-1])
        total_variance.append(sum(values["var_energy"]))
    
    return energy_matrix, var_energy_matrix, np.array(final_energy), np.array(total_variance), dimensions, qubits


file_path = "../../Programas/resultados/DMRG_energia_L100_D100_t15.txt"
data = parse_dmrg_file(file_path)
energy_matrix, var_energy_matrix, final_energy, total_variance, dimensions, qubits = create_matrices(data)


plt.figure(figsize=(8, 5))
plt.plot(dimensions, final_energy, marker='o', linestyle='-', color='b', label='Energía Final')
plt.xlabel('Dimensión')
plt.ylabel('Energía Final')
plt.title('Energía Final en función de la Dimensión')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
sweeps = np.arange(1, 11)
for i, dim in enumerate(dimensions):
    plt.plot(sweeps, energy_matrix[0, i, :], marker='o', linestyle='-', label=f'Dimensión {dim}')
plt.xlabel('Sweep')
plt.ylabel('Energía')
plt.title('Energía en función de Sweep para diferentes Dimensiones')
plt.legend()
plt.grid()
plt.show()