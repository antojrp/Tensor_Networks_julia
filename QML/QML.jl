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

    bond = pi/2

    y, samples, nsamples, nfeatures = load_data_and_samples("data.csv"; bond = bond)

    println("Datos cargados: $nsamples muestras, $nfeatures características.")


    # 2) PARÁMETROS DE LA SIMULACIÓN

    n_train_samples = 450
    n_test_samples  = nsamples - n_train_samples
    N = nfeatures
    L = 6
    compute_stats = true
    n_runs = 10

    accuracies = Float64[]


    # 3) PRECOMPUTAR TODOS LOS ESTADOS UNA ÚNICA VEZ
    states = nothing


    if compute_stats
        Ds_all = nothing
        Renyis_all = nothing
        states, Ds_all, Renyis_all = compute_all_states(samples, N, L; compute_stats = true)
    else
        states = compute_all_states(samples, N, L; compute_stats = false)
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
        train_idx, test_idx, y_train_int, y_test_int =
            split_train_test(y, nsamples, n_train_samples, n_test_samples)


        K_train = compute_train_kernel(states, train_idx)
        println("Kernel train calculado: ", size(K_train))

        model = svmtrain(K_train, y_train_int; kernel = Kernel.Precomputed)
        println("SVM entrenado.")


        K_test = compute_test_kernel(states, train_idx, test_idx)
        println("Kernel test calculado: ", size(K_test))

        y_pred, _ = svmpredict(model, K_test)


        acc = sum(y_pred .== y_test_int) / n_test_samples
        push!(accuracies, acc)
        println("Accuracy test = ", acc)

        println("Train: #1 = ", sum(y_train_int .== 1), "  #-1 = ", sum(y_train_int .== -1))
        println("Test:  #1 = ", sum(y_test_int .== 1),  "  #-1 = ", sum(y_test_int .== -1))
    end


    println("\n********** RESUMEN FINAL SVM **********")
    println("L = $L sobre $n_runs runs")

    mean_accuracy = mean(accuracies)
    var_accuracy  = var(accuracies)
    println("Mean accuracy: ", mean_accuracy, "   Variance: ", var_accuracy)
end
