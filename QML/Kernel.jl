using LinearAlgebra
using ITensors
using ITensorMPS
using CSV
using Statistics
using Random
using DelimitedFiles

Random.seed!()

let
    # 1) Cargar funciones
    include("QKernelFunctions.jl")
    using .QKernelFunctions

    # CONFIGURACIÓN
    featuremap   = :ZZ        # :ZZ o :ising
    dataset_name = "arrhythmia"
    data_file    = "arrythmia.csv"

    bond   = 0.05
    L_ini  = 1
    L_fin  = 6
    L_step = 1

    println("Dataset: $dataset_name")
    println("Featuremap: $featuremap")

    # 2) Cargar datos
    #y, samples, nsamples, nfeatures = load_data_and_samples("breast_cancer.csv"; label_col = 2, feature_cols = collect(3:32), pos_label = "B", neg_label = "M", bond = bond)
    #y, samples, nsamples, nfeatures = load_data_and_samples("ionosphere.csv"; label_col = 35, feature_cols = Int[], pos_label = "g", neg_label = "b", bond = bond)
    #y, samples, nsamples, nfeatures = load_data_and_samples("sonar.csv"; label_col = 61, feature_cols = Int[], pos_label = "M", neg_label = "R", bond = bond)
    y, samples, nsamples, nfeatures = load_data_and_samples("arrhythmia.csv"; label_col = 280, feature_cols = vcat(1:10, 16:279), pos_label = 1, neg_label = 2, bond = bond)
        
    # 2.1) Reordenar: primero positivos (y = 1.0), luego negativos (y = -1.0)
    pos_idx = findall(==(1.0), y)
    neg_idx = findall(==(-1.0), y)
    perm    = vcat(pos_idx, neg_idx)

    y       = y[perm]
    samples = samples[perm]

    println("n+ = $(length(pos_idx)), n- = $(length(neg_idx))")
    println("Muestras reordenadas: primero clase +1, luego clase -1")

    # 3) Estado inicial |0...0>
    N     = nfeatures
    sites = siteinds("Qubit", N)
    ψ0    = MPS(sites, "0")

    # Asegurar carpeta de salida
    folder = joinpath("Kernels", dataset_name)
    isdir(folder) || mkpath(folder)

    # 4) Barrer en L y en log(D)
    for L in L_ini:L_step:L_fin
        for k in 1:L
            D = 2^k
            println("\nL = $L, Max dim = $D")

            println("Calculando todos los estados...")
            states = compute_all_states(
                samples, ψ0, L;
                featuremap    = featuremap,
                compute_stats = false,
                D             = D,
            )

            println("Calculando kernel completo...")
            K = compute_full_kernel(states)

            outname = joinpath(folder,
                string(dataset_name, "_L_", L, "_log(D)_", k, ".csv")
            )
            writedlm(outname, K, ',')

            println("Kernel guardado en: ", outname)
        end
    end
end
