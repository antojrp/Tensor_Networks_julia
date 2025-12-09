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
    include("QKernelFunctions.jl")
    using .QKernelFunctions
    featuremap = :ZZ         # :ZZ, :ising
    println("Dataset: Arrhythmia. Featuremap: ", featuremap)

    bond_ini = 0.01
    bond_fin = 0.01
    bond_step = 0.1

    L_ini = 5
    L_fin = 5
    L_step = 1

    for bond in bond_ini:bond_step:bond_fin

        println("\nBond: ", bond)
        # 1) LOAD DATASET

        #y, samples, nsamples, nfeatures = load_data_and_samples("breast_cancer.csv"; label_col = 2, feature_cols = collect(3:32), pos_label = "B", neg_label = "M", bond = bond)
        #y, samples, nsamples, nfeatures = load_data_and_samples("ionosphere.csv"; label_col = 35, feature_cols = Int[], pos_label = "g", neg_label = "b", bond = bond)
        #y, samples, nsamples, nfeatures = load_data_and_samples("sonar.csv"; label_col = 61, feature_cols = Int[], pos_label = "M", neg_label = "R", bond = bond)
        y, samples, nsamples, nfeatures = load_data_and_samples("arrhythmia.csv"; label_col = 280, feature_cols = vcat(1:10, 16:279), pos_label = 1, neg_label = 2, bond = bond)
        for L in L_ini:L_step:L_fin
            println("\nL = $L")
            for k in 1:L 
            #for k in L:L
                D=2^k
                println("\nMax dim = $D")

                N = nfeatures
                compute_stats = true


                k_folds = 5         
                n_runs  = 1

                accuracies = Float64[]
                states = nothing

                # Define initial MPS state |0...0>
                sites = siteinds("Qubit", N)
                ψ0 = MPS(sites, "0")

                # 2) PRECOMPUTE ALL STATES
                if compute_stats
                    Ds_all = nothing
                    Renyis_all = nothing
                    states, Ds_all, Renyis_all = compute_all_states(samples, ψ0, L; featuremap = featuremap, compute_stats = true, D = D)
                else
                    states = compute_all_states(samples, ψ0, L;featuremap = featuremap, compute_stats = false, D = D)
                end

                println("Memoria: ", Base.summarysize(states) / 1024^2, " MB")

                if compute_stats
                    mean_D      = mean(Ds_all)
                    var_D       = var(Ds_all)
                    mean_Renyi  = mean(Renyis_all)
                    var_Renyi   = var(Renyis_all)

                    #Print stats
                    println("Mean D: ", mean_D)
                    println("Var  D: ", var_D)
                    println("Mean Renyi2: ", mean_Renyi)
                    println("Var  Renyi2: ", var_Renyi)
                end


                for run in 1:n_runs
                    # 3) K-FOLD CROSS-VALIDATION WITH SVM

                    # Split data in k folds
                    folds = split_train_test(y, k_folds)
                    for f in 1:k_folds
                        # Test indexes: fold f
                        test_idx = folds[f]

                        # Train indexes: all except fold f
                        train_idx = vcat((folds[i] for i in 1:k_folds if i != f)...)

                        y_train_int = Int.(y[train_idx])
                        y_test_int  = Int.(y[test_idx])

                        n_train_samples = length(train_idx)
                        n_test_samples  = length(test_idx)



                        # Compute train kernel
                        K_train = compute_train_kernel(states, train_idx)
                        #println("Kernel train calculado: ", size(K_train))

                        # SVM Train
                        model = svmtrain(K_train, y_train_int; kernel = Kernel.Precomputed, cost=1.0)
                        #println("SVM entrenado.")

                        # Compute test kernel
                        K_test = compute_test_kernel(states, train_idx, test_idx)
                        #println("Kernel test calculado: ", size(K_test))

                        # Prediction
                        y_pred, _ = svmpredict(model, K_test)

                        # Accuracy
                        acc_fold = sum(y_pred .== y_test_int) / n_test_samples
                        push!(accuracies, acc_fold)
                    end
                end

                # 4) PRINT RESULTS
                mean_accuracy = mean(accuracies)
                var_accuracy  = var(accuracies)
                println("Mean accuracy: ", mean_accuracy)
                println("Variance of accuracy: ", var_accuracy)
            end
        end
    end
end