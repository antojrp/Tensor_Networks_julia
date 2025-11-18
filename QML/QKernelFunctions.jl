module QKernelFunctions

# Export functions

export load_data_and_samples, compute_train_kernel, split_train_test, compute_test_kernel, compute_all_states


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

function compute_train_kernel(states::Vector{MPS}, train_idx::AbstractVector{Int})

    n_train = length(train_idx)
    K_train = Matrix{Float64}(undef, n_train, n_train)

    for a in 1:n_train
        i = train_idx[a]
        ψ_i = states[i]

        # Diagonal
        K_train[a, a] = 1.0

        for b in (a+1):n_train
            j = train_idx[b]
            ψ_j = states[j]

            ov = inner(ψ_i, ψ_j)
            k = abs2(ov)

            K_train[a, b] = k
            K_train[b, a] = k
        end
    end

    return K_train
end

function split_train_test(y::AbstractVector, nsamples::Int, n_train_samples::Int, n_test_samples::Int)

    idx = randperm(nsamples)

    train_idx = idx[1:n_train_samples]
    test_idx  = idx[n_train_samples+1 : n_train_samples + n_test_samples]

    y_train_int = Int.(y[train_idx])
    y_test_int  = Int.(y[test_idx])

    return train_idx, test_idx, y_train_int, y_test_int
end

function compute_test_kernel(states::Vector{MPS}, train_idx::AbstractVector{Int}, test_idx::AbstractVector{Int})

    n_train = length(train_idx)
    n_test  = length(test_idx)

    K_test = Matrix{Float64}(undef, n_train, n_test)

    for (col, j) in enumerate(test_idx)
        ψ_test = states[j]
        for (row, i) in enumerate(train_idx)
            ψ_train = states[i]
            ov = inner(ψ_train, ψ_test)
            K_test[row, col] = abs2(ov)
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
        println("Computing state for sample $i / $nsamples")
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