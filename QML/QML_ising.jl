using LinearAlgebra
using ITensors
using ITensorMPS
using CSV
using Statistics
using LIBSVM
using Random
using DelimitedFiles

Random.seed!()

let

    # 1) CARGA DE DATOS Y PREPROCESADO
    include("QKernelFunctions.jl")
    using .QKernelFunctions

    bond = 2*pi

    y, samples, nsamples, nfeatures = load_data_and_samples("breast_cancer.csv"; label_col = 2, feature_cols = collect(3:32), pos_label = "B", neg_label = "M", bond = 2*pi)

    println("Datos cargados: $nsamples muestras, $nfeatures características.")


    # 2) PARÁMETROS DE LA SIMULACIÓN

    N = nfeatures
    L = 1

    compute_stats = true
    featuremap = :ising           # :ZZ, :ising

    k_folds = 5         
    n_runs  = 5

    accuracies = Float64[]


    # 3) PRECOMPUTAR TODOS LOS ESTADOS UNA ÚNICA VEZ
    states = nothing

    gamma_ising   = 0.3      # el valor de γ que quieras
    println("Calculando estado fundamental del modelo de Ising para γ = $gamma_ising ...")
    nsweeps_dmrg  = 10        # nº de sweeps de DMRG
    maxdim_dmrg   = 32        # dimensión máxima del enlace
    sites = siteinds("Qubit", N)
    # Estado fundamental del modelo de Ising
    ψ0 = ground_state_ising(sites, gamma_ising; nsweeps = nsweeps_dmrg, maxdim = maxdim_dmrg)
    cut = fld(N, 2)
    S = QKernelFunctions.entropy(ψ0, cut)
    D = schmidt(ψ0, N)
    println("Entropía de von Neumann del estado fundamental del Ising (N=$N, γ=$gamma_ising): S = $S  D = $D")


    if compute_stats
        Ds_all = nothing
        Renyis_all = nothing
        states, Ds_all, Renyis_all = compute_all_states(samples, ψ0, L; featuremap = featuremap, compute_stats = true)
    else
        states = compute_all_states(samples, ψ0, L;featuremap = featuremap, compute_stats = false)
    end

    println("Estados precomputados para las $nsamples muestras.")
    println("Memoria: ", Base.summarysize(states) / 1024^2, " MB")

    if compute_stats
        mean_D      = mean(Ds_all)
        var_D       = var(Ds_all)
        mean_Renyi  = mean(Renyis_all)
        var_Renyi   = var(Renyis_all)

        println("\n********** ESTADÍSTICAS GLOBALES DE ENTANGLEMENT **********")
        println("Mean D (sobre los $nsamples estados):        ", mean_D)
        println("Var  D (sobre los $nsamples estados):        ", var_D)
        println("Mean Renyi2 (sobre los $nsamples estados):   ", mean_Renyi)
        println("Var  Renyi2 (sobre los $nsamples estados):   ", var_Renyi)
    end


    # 4) MÚLTIPLES RUNS CON DIFFERENTES SPLITS TRAIN/TEST

    for run in 1:n_runs
        println("\n================ RUN $run ================")

        # Split usando solo índices y etiquetas
        folds = split_train_test(y, k_folds)
        accuracies_folds = Float64[] 

        for f in 1:k_folds
            println("\n---------- FOLD $f / $k_folds ----------")

            # Índices de test para este fold
            test_idx = folds[f]

            # Índices de train: todos los demás folds concatenados
            train_idx = vcat((folds[i] for i in 1:k_folds if i != f)...)

            y_train_int = Int.(y[train_idx])
            y_test_int  = Int.(y[test_idx])

            n_train_samples = length(train_idx)
            n_test_samples  = length(test_idx)

            println("Train size = $n_train_samples, Test size = $n_test_samples")

            # KERNEL TRAIN
            K_train = compute_train_kernel(states, train_idx)
            println("Kernel train calculado: ", size(K_train))

            # ENTRENAR SVM
            model = svmtrain(K_train, y_train_int; kernel = Kernel.Precomputed)
            println("SVM entrenado.")

            # KERNEL TEST
            K_test = compute_test_kernel(states, train_idx, test_idx)
            println("Kernel test calculado: ", size(K_test))

            # PREDICCIÓN
            y_pred, _ = svmpredict(model, K_test)

            # ACCURACY DEL FOLD
            acc_fold = sum(y_pred .== y_test_int) / n_test_samples
            push!(accuracies, acc_fold)
            push!(accuracies_folds, acc_fold)
            println("Accuracy test (fold $f) = ", acc_fold)

            println("Train: #1 = ", sum(y_train_int .== 1),  "  #-1 = ", sum(y_train_int .== -1))
            println("Test:  #1 = ", sum(y_test_int  .== 1),  "  #-1 = ", sum(y_test_int  .== -1))
        end

        # Accuracy media de este run (sobre los k folds)
        acc_run = mean(accuracies_folds)
        println("\nAccuracy media del RUN $run (sobre $k_folds folds) = ", acc_run)
    end

    # 5) RESUMEN FINAL SOBRE TODOS LOS RUNS
    println("\n********** RESUMEN FINAL SVM **********")
    println("L = $L, k_folds = $k_folds, n_runs = $n_runs")

    mean_accuracy = mean(accuracies)
    var_accuracy  = var(accuracies)
    println("Mean accuracy (media sobre runs): ", mean_accuracy)
    println("Variance of accuracy (entre runs): ", var_accuracy)
end