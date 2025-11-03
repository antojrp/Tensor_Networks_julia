using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using Base.Threads
using BenchmarkTools

let
    include("pTEBD(debug).jl")
    using .pTEBD
    include("random_circuit.jl")
    using .random_circuits

    # Configurar hilos
    BLAS.set_num_threads(1)
    println("Threads Julia activos: ", nthreads())
    println("Threads BLAS activos: ", BLAS.get_num_threads())

    # Parámetros del experimento
    N = 15                          # número de sitios fijos
    L_vals = [8, 9, 10, 11, 12, 13] # distintos tamaños locales (2^L dimensión)

    println("\n──────────────────────────────────────────────────────────────")
    println("⚙️  Test de escalado paralelo ITensor (multiplicaciones)")
    println("──────────────────────────────────────────────────────────────")
    println(rpad("L", 5), rpad("Tiempo seq (s)", 20), rpad("Tiempo par (s)", 20), "Speedup")
    println("──────────────────────────────────────────────────────────────")

    for L in L_vals
        # Crear índices
        i = [Index(2^L) for k in 1:N]

        # Función para generar un vector de tensores aleatorios
        function generar_vector()
            vec = [random_itensor(i[k], i[k+1]) for k in 1:N-1]
            for k in 2:2:N-2
                vec[k] = vec[k-1] * vec[k] * vec[k+1]
            end
            return vec
        end

        # Ejecución paralela
        vector_par = generar_vector()
        t_par = @elapsed begin
            @threads for k in 2:2:N-2
                vector_par[k] = vector_par[k-1] * vector_par[k] * vector_par[k+1]
            end
        end

        # Ejecución secuencial
        vector_seq = generar_vector()
        t_seq = @elapsed begin
            for k in 2:2:N-2
                vector_seq[k] = vector_seq[k-1] * vector_seq[k] * vector_seq[k+1]
            end
        end

        speedup = t_seq / t_par

        println(rpad(string(L), 5),
                rpad(string(round(t_seq, digits=4)), 20),
                rpad(string(round(t_par, digits=4)), 20),
                string(round(speedup, digits=2), "×"))
    end

    println("──────────────────────────────────────────────────────────────")
end
