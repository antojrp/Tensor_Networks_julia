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

    y, samples, nsamples, nfeatures = load_data_and_samples("data.csv"; bond= bond)

    println("Datos cargados: $nsamples muestras, $nfeatures características.")

    # 2) TRAIN / TEST
    n_train_samples = 450
    n_test_samples  = nsamples - n_train_samples
    #n_test_samples  = 50
    N = nfeatures
    L = 1
    compute_stats = true
    n_runs = 10
    accuracies = []
    Ds = []
    varDs = []
    Renyis = []
    varRenyis = []


    for run in 1:n_runs
        samples_train, samples_test, y_train_int, y_test_int = split_train_test(samples, y, nsamples, n_train_samples, n_test_samples)


        # 3) KERNEL TRAIN 
        if compute_stats
            K_train, mean_D, mean_Renyis, var_D, var_Renyis = compute_train_kernel(samples_train, n_train_samples, N, L; compute_stats=true)
            push!(Ds, mean_D)
            push!(varDs, var_D)
            push!(Renyis, mean_Renyis)
            push!(varRenyis, var_Renyis)

        else
            K_train = compute_train_kernel(samples_train, n_train_samples, N, L)
        end
        println("Kernel train calculado: ", size(K_train))


        # 4) ENTRENAR SVM
        model = svmtrain(K_train, y_train_int; kernel = Kernel.Precomputed)
        println("SVM entrenado.")


        # 5) KERNEL TEST 
        K_test = compute_test_kernel(samples_train, samples_test, n_train_samples, n_test_samples, N, L)
        println("Kernel test calculado: ", size(K_test))


        # 6) PREDICCIÓN
        y_pred, _ = svmpredict(model, K_test)

         # 7) ACCURACY
        acc = sum(y_pred .== y_test_int) / n_test_samples
        push!(accuracies, acc)
        println("Accuracy test = ", acc)

        println("Train: #1 = ", sum(y_train_int .== 1), "  #-1 = ", sum(y_train_int .== -1))
        println("Test:  #1 = ", sum(y_test_int .== 1),  "  #-1 = ", sum(y_test_int .== -1))

    
    end
    println("L= $L over $n_runs runs:")
    mean_D = mean(Ds)
    var_D = mean(varDs)
    mean_Renyis = mean(Renyis)
    var_Renyis = mean(varRenyis)
    println("Mean D: ", mean_D, "  Variance: ", var_D)
    println("Mean Renyi2s: ", mean_Renyis, "  Variance: ", var_Renyis)

    mean_accuracy = mean(accuracies)
    var_accuracy = var(accuracies)
    println("Mean accuracy: ", mean_accuracy, "  Variance: ", var_accuracy)

end
