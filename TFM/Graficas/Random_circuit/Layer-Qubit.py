import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 19})

# Definir los valores de L mínimo, L máximo, q mínimo y q máximo
L_min = 5  # Cambia según necesites
L_max = 15  # Cambia según necesites
q_min = 20  # Número de qubits mínimo
q_max = 60  # Número de qubits máximo
q_extr = 100  # Número de qubits al que queremos extrapolar
umbral_tiempo = 600  # Umbral máximo de tiempo total permitido

# Lista de capas L en las que deseas realizar la extrapolación
L_extrapolar = [5,6,7,8,9,10,11,12]  # Elige las capas L específicas para las cuales quieres extrapolar

# Inicializar una matriz de ceros con tamaño (L_max - L_min + 1, q_extr - q_min + 1)
matriz_tiempos = np.zeros((L_max - L_min + 1, q_extr - q_min + 1))

# Leer los archivos en el rango de L
for L in range(L_min, L_max + 1):
    file_path = f"../../Programas/resultados/Random_tiempo_{L}.txt"
    
    with open(file_path, "r") as file:
        lines = file.readlines()
        qubits = None  # Se reinicia para cada archivo
        
        qubits_data = []  # Lista para almacenar los qubits
        time_data = []  # Lista para almacenar los tiempos
        
        for line in lines:
            if "Numero de qubits" in line:
                qubits = int(line.split(":")[1].strip())  # Guardar el número de qubits
            elif "Total=" in line and qubits is not None:
                total_time = float(line.split("=")[1].strip())  # Guardar el tiempo total
                
                # Filtrar valores mayores a 600
                if total_time <= umbral_tiempo:
                    if q_min <= qubits <= q_max:
                        qubits_data.append(qubits)
                        time_data.append(total_time)

        # Realizar el ajuste lineal para qubits >= 25 solo si L está en la lista de extrapolación
        qubits_data = np.array(qubits_data)
        time_data = np.array(time_data)
        
        # Seleccionar datos con qubits >= 25 para el ajuste
        mask = qubits_data >= 25
        qubits_data_filtered = qubits_data[mask]
        time_data_filtered = time_data[mask]
        
        if L in L_extrapolar and len(qubits_data_filtered) > 1:  # Solo hacer ajuste si L está en la lista de extrapolación y hay suficientes datos
            # Ajuste lineal (usamos np.polyfit para un ajuste de primer grado)
            slope, intercept = np.polyfit(qubits_data_filtered, time_data_filtered, 1)
            
            # Extrapolar hasta q_extr
            qubits_extr = np.arange(q_min, q_extr + 1)
            time_extr = slope * qubits_extr + intercept  # Extrapolación
            
            # Obtener la fila correspondiente a L (restamos L_min para hacer el índice)
            L_index = L - L_min
            
            # Almacenar los tiempos extrapolados en la matriz, solo si no hay datos
            for i, qubit in enumerate(qubits_extr):
                q_index = qubit - q_min  # Índice ajustado para la matriz
                if matriz_tiempos[L_index, q_index] == 0:  # Si no hay datos, almacenar el extrapolado
                    matriz_tiempos[L_index, q_index] = time_extr[i]
        
        else:
            print(f"Datos insuficientes para ajuste lineal en L = {L} o L no está en la lista de extrapolación.")

        # Almacenar los tiempos leídos en la matriz (sin modificarlos)
        for qubit, time in zip(qubits_data, time_data):
            if q_min <= qubit <= q_max:
                L_index = L - L_min
                q_index = qubit - q_min
                matriz_tiempos[L_index, q_index] = time



# Crear el mapa de calor (mapa gradiente) de los tiempos
plt.figure(figsize=(15, 6))

# Crear la normalización logarítmica para la escala de colores
norm = mcolors.LogNorm(vmin=1e-2, vmax=600)

# Mostrar la matriz de tiempos usando un mapa de colores (imshow)
# Usamos extent para centrar el mapa de calor en las celdas
extent = [q_min - 0.5, q_extr + 0.5, L_min - 0.5, L_max + 0.5]
matriz_tiempos[matriz_tiempos > 600] = 0
matriz_unos = np.ones_like(matriz_tiempos)
plt.imshow(matriz_unos, aspect='auto', cmap='gray', origin='lower',extent=extent, vmax=1.1,vmin=0)
plt.imshow(matriz_tiempos, aspect='auto', cmap='viridis', origin='lower', extent=extent, norm=norm)

# Añadir una barra de colores
cbar = plt.colorbar()
cbar.set_label("Time")
cbar.set_ticks([1, 60, 600])  # Mostrar solo los valores 1, 60 y 600
cbar.set_ticklabels(["1 s", "1 min", "10 min"])  # Etiquetas de los valores

# Arrays para guardar las coordenadas x e y donde el primer tiempo superior a 60
x_first_above_60 = []
y_first_above_60 = []
x_first_above_1 = []
y_first_above_1 = []

# Recorrer cada fila (para cada valor de L)
for L_index in range(matriz_tiempos.shape[0]):
    # Obtener los qubits (de q_min a q_extr) y los tiempos correspondientes a esa fila
    qubits_vals = np.arange(q_min, q_extr + 1)
    time_vals = matriz_tiempos[L_index, :]
    
    # Buscar el primer valor de qubits donde el tiempo supera 60
    for i, time in enumerate(time_vals):
        if time > 1:
            x_first_above_1.append(qubits_vals[i] - 0.5)  # Restar 0.5 para ajustar a la izquierda
            y_first_above_1.append(L_min + L_index - 0.5)  # Sumar 0.5 para mover hacia arriba
            break
    for i, time in enumerate(time_vals):
        if time > 60:
            # Guardar el primer valor de x y la capa correspondiente
            x_first_above_60.append(qubits_vals[i] - 0.5)  # Restar 0.5 para ajustar a la izquierda
            y_first_above_60.append(L_min + L_index - 0.5)  # Sumar 0.5 para mover hacia arriba
            break  # Salir del bucle una vez encontramos el primer valor > 60


# Añadir el primer escalón al principio (en el borde izquierdo)
x_first_above_60.insert(0, q_extr + 0.5 )  # Añadir el borde izquierdo
y_first_above_60.insert(0, y_first_above_60[0])  # Mantener la misma altura que el primer valor de y
x_first_above_60.insert(len(x_first_above_60), x_first_above_60[-1])  
y_first_above_60.insert(len(y_first_above_60), L_max+0.5) 

x_first_above_1.insert(0, q_extr + 0.5 )  # Añadir el borde izquierdo
y_first_above_1.insert(0, y_first_above_1[0])  # Mantener la misma altura que el primer valor de y
x_first_above_1.insert(len(x_first_above_1), x_first_above_1[-1] )  
y_first_above_1.insert(len(y_first_above_1), L_max+0.5) 


plt.step(x_first_above_60, y_first_above_60, color='black', linewidth=2, zorder=5,label='1 min')
plt.step(x_first_above_1, y_first_above_1, color='black', linewidth=1.5 , linestyle='dashdot' , zorder=5,label='1 s')
plt.vlines(60,L_min-0.5,L_max+0.5,linestyles="--", color='grey', linewidth=2)


# Etiquetas de los ejes
plt.xlabel("Number of Qubits")
plt.ylabel("Layers")
#plt.title("Time required to update the MPS representation\nafter applying the quantum circuit")
plt.legend()
gray_patch = mpatches.Patch(color='#e8e8e8', label='>10 min')
plt.legend(handles=[gray_patch] + plt.gca().get_legend_handles_labels()[0])
plt.text(36,14.8,'Simulation')
plt.text(67,14.2,'Estimation from \n    linear fit')
# Ajustar la disposición para que los elementos no se solapen
plt.tight_layout()

# Mostrar el mapa de calor
plt.savefig('Layer-Qubit.pdf',format='pdf', bbox_inches='tight')
plt.show()

