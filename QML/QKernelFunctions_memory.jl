module QKernelFunctions

# Export functions

export load_data_and_samples, compute_train_kernel, split_train_test, compute_test_kernel


using LinearAlgebra
using ITensors
using ITensorMPS
using CSV
using Statistics
using LIBSVM
using Random
using DelimitedFiles

include("Featuremaps.jl")
using .Featuremaps

function schmidt(state,N)
    # suma=0
    max=0
    for i in 1:N-1
      let
        if max < dim(inds(state[i])[end])
          max=dim(inds(state[i])[end])
        end
      end
    end
    # return suma/(N-1)
    return max
end

function entropy(psi,b)
    psi = orthogonalize(psi, b)
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    SvN = 0.0
    for n=1:dim(S, 1)
      p = S[n,n]^2
      #SvN -= p * log(p)
      SvN += p^2
    end
    return -log2(SvN)
end

function load_data_and_samples(filename::AbstractString; bond = pi/2)
    # Cargar datos
    data = CSV.File(filename; header = true, silencewarnings = true)
    n = length(data)

    # Contar número real de features (ignorando las 2 primeras columnas)
    vals_first = collect(first(data))
    vals_features_first = vals_first[3:end]
    vals_features_first = filter(!ismissing, vals_features_first)
    p = length(vals_features_first)

    # Matriz X (n × p) y vector y
    X = Array{Float64}(undef, n, p)
    y = Vector{Float64}(undef, n)

    for (i, row) in enumerate(data)
        vals = collect(row)

        # Etiqueta: B → 1.0, M → -1.0
        diag = String(vals[2])
        y[i] = diag == "B" ? 1.0 : -1.0

        # Features desde la 3ª columna
        vals_features = filter(!ismissing, vals[3:end])
        X[i, :] = Float64.(vals_features)
    end

    # Normalización
    mu = mean(X, dims = 1)
    X_norm = X .- mu

    max_abs = maximum(abs.(X_norm))
    X_scaled = (bond) .* X_norm ./ max_abs

    # Convertimos cada fila en un vector
    samples = [vec(X_scaled[i, :]) for i in 1:n]

    return y, samples, n, p
end

function compute_train_kernel(samples_train, n_train_samples::Int, N::Int, L::Int; compute_stats::Bool = false)

    K_train = Matrix{Float64}(undef, n_train_samples, n_train_samples)
    sites = siteinds("Qubit", N)
    if compute_stats
        Ds = Float64[]
        Renyis = Float64[]
    end
    for i in 1:n_train_samples
        #println("Calculando fila $i del kernel train...")
        x_i = samples_train[i]

        circuit_i = ZZfeaturemap_linear(sites, N, L, x_i, false)
        ψ_i = productMPS(sites, "0")

        for layer in circuit_i
            for gate in layer
                ψ_i = apply(gate, ψ_i)
            end
        end

        if compute_stats
            link_dim = schmidt(ψ_i, N)
            S2 = entropy(ψ_i, Int(floor(N/2)))
            push!(Ds, link_dim)
            push!(Renyis, S2)
        end 

        K_train[i, i] = 1.0

        for j in (i+1):n_train_samples
            x_j = samples_train[j]

            circuit_j = ZZfeaturemap_linear(sites, N, L, x_j, false)
            ψ_j = productMPS(sites, "0")

            for layer in circuit_j
                for gate in layer
                    ψ_j = apply(gate, ψ_j)
                end
            end

            ov = inner(ψ_i, ψ_j)
            k = abs2(ov)

            K_train[i, j] = k
            K_train[j, i] = k
        end
    end
    if compute_stats
        mean_D = mean(Ds)
        var_D = var(Ds)
        mean_Renyis = mean(Renyis)
        var_Renyis = var(Renyis)

        return K_train, mean_D, mean_Renyis, var_D, var_Renyis
    else
        return K_train
    end

end

function split_train_test(samples, y, nsamples::Int, n_train_samples::Int, n_test_samples::Int)
    idx = randperm(nsamples)

    train_idx = idx[1:n_train_samples]
    test_idx  = idx[n_train_samples+1 : n_train_samples + n_test_samples]

    samples_train = samples[train_idx]
    samples_test  = samples[test_idx]

    y_train = y[train_idx]
    y_test  = y[test_idx]

    y_train_int = Int.(y_train)
    y_test_int  = Int.(y_test)

    return samples_train, samples_test, y_train_int, y_test_int
end

function compute_test_kernel(samples_train, samples_test,
                             n_train_samples::Int, n_test_samples::Int,
                             N::Int, L::Int)

    K_test = Matrix{Float64}(undef, n_train_samples, n_test_samples)
    sites = siteinds("Qubit", N)

    for (t_pos, x_test_vec) in enumerate(samples_test)
        #println("Calculando columna (test) $t_pos del kernel...")

        # Estado para el sample de test
        circuit_test = ZZfeaturemap_linear(sites, N, L, x_test_vec, false)
        ψ_test = productMPS(sites, "0")
        for layer in circuit_test
            for gate in layer
                ψ_test = apply(gate, ψ_test)
            end
        end

        # Comparar con cada sample de train
        for (tr_pos, x_train_vec) in enumerate(samples_train)
            circuit_train = ZZfeaturemap_linear(sites, N, L, x_train_vec, false)
            ψ_train = productMPS(sites, "0")
            for layer in circuit_train
                for gate in layer
                    ψ_train = apply(gate, ψ_train)
                end
            end

            ov = inner(ψ_train, ψ_test)
            K_test[tr_pos, t_pos] = abs2(ov)
        end
    end

    return K_test
end

function compute_all_states(samples, N::Int, L::Int; compute_stats::Bool=false)
    nsamples = length(samples)
    sites = siteinds("Qubit", N)

    states = Vector{MPS}(undef, nsamples)

    Ds = compute_stats ? Vector{Float64}(undef, nsamples) : Float64[]
    Renyis = compute_stats ? Vector{Float64}(undef, nsamples) : Float64[]

    for i in 1:nsamples
        x_i = samples[i]

        circuit_i = ZZfeaturemap_linear(sites, N, L, x_i, false)
        ψ_i = productMPS(sites, "0")

        for layer in circuit_i
            for gate in layer
                ψ_i = apply(gate, ψ_i)
            end
        end

        states[i] = ψ_i

        if compute_stats
            Ds[i] = schmidt(ψ_i, N)
            Renyis[i] = entropy(ψ_i, Int(floor(N/2)))
        end
    end

    if compute_stats
        return states, Ds, Renyis
    else
        return states
    end
end


end 