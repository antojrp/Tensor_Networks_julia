module QKernelFunctions

export load_data_and_samples, compute_train_kernel, split_train_test, compute_test_kernel, compute_all_states, ground_state_ising, schmidt, entropy

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

using CSV
using Statistics

function load_data_and_samples(filename::AbstractString; label_col::Int, feature_cols::Vector{Int} = Int[], pos_label = nothing, neg_label = nothing, bond = pi/2)
    # Cargar datos
    data = CSV.File(filename; header = true, silencewarnings = true)
    n = length(data)
    n == 0 && error("El fichero $filename no tiene filas")

    # Número de columnas total (miramos la primera fila)
    first_row = collect(first(data))
    ncols = length(first_row)

    # Si no se especifican feature_cols, usamos todas menos label_col
    if isempty(feature_cols)
        cols = collect(1:ncols)
        feature_cols = [c for c in cols if c != label_col]
    end

    p = length(feature_cols)

    X = Array{Float64}(undef, n, p)
    y = Vector{Float64}(undef, n)

    for (i, row) in enumerate(data)
        vals = collect(row)

        # Etiqueta
        lab_val = vals[label_col]

        if pos_label !== nothing && neg_label !== nothing
            if lab_val == pos_label
                y[i] = 1.0
            elseif lab_val == neg_label
                y[i] = -1.0
            else
                error("Etiqueta desconocida en fila $i: $lab_val (esperaba $pos_label o $neg_label)")
            end
        else
            if lab_val === missing
                error("Etiqueta missing en fila $i")
            end
            y[i] = Float64(lab_val)
        end

        # Features
        for (k, j) in enumerate(feature_cols)
            v = vals[j]
            if v === missing
                error("Hay un missing en fila $i columna $j, ahora mismo no lo estoy tratando")
            end
            X[i, k] = Float64(v)
        end
    end

    # Normalización como hacías antes
    #mu = mean(X, dims = 1)
    #X_norm = X .- mu

    #max_abs = maximum(abs.(X_norm))
    #X_scaled = bond .* X_norm ./ max_abs

    mu = mean(X, dims = 1)
    X_centered = X .- mu

    max_abs_per_feature = maximum(abs.(X_centered), dims = 1)
    safe_scale = map(x -> x == 0 ? 1 : x, max_abs_per_feature)

    X_scaled = bond .* (X_centered ./ safe_scale)

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

function split_train_test(y::AbstractVector, k_folds::Int)
    # Devuelve un vector de folds,
    # cada folds[f] es un Vector{Int} con los índices de test de ese fold,
    # estratificados por clase.

    # Asumo etiquetas 1.0 y -1.0 en y
    pos_idx = findall(==(1.0), y)
    neg_idx = findall(==(-1.0), y)

    # Barajamos cada clase por separado
    Random.shuffle!(pos_idx)
    Random.shuffle!(neg_idx)

    # Inicializamos los folds vacíos
    folds = [Int[] for _ in 1:k_folds]

    # Reparto round-robin de los positivos
    for (i, idx) in enumerate(pos_idx)
        fold_id = mod1(i, k_folds)   # 1,2,...,k,1,2,...
        push!(folds[fold_id], idx)
    end

    # Reparto round-robin de los negativos
    for (i, idx) in enumerate(neg_idx)
        fold_id = mod1(i, k_folds)
        push!(folds[fold_id], idx)
    end

    return folds
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


function compute_all_states(samples, init_state::MPS, L::Int; featuremap::Symbol = :ZZ, compute_stats::Bool = false)
    nsamples = length(samples)

    # Sacamos sites y N del estado inicial
    sites = siteinds(init_state)
    N = length(sites)

    states = Vector{MPS}(undef, nsamples)

    Ds     = compute_stats ? Vector{Float64}(undef, nsamples) : Float64[]
    Renyis = compute_stats ? Vector{Float64}(undef, nsamples) : Float64[]

    for i in 1:nsamples
        println("Computing state for sample $i / $nsamples")
        x_i = samples[i]

        # Elegir feature map
        circuit_i = if featuremap === :ZZ
            ZZfeaturemap_linear(sites, N, L, x_i)
        elseif featuremap === :ising
            IsingFeaturemap_linear(sites, N, L, x_i)
        else
            error("Feature map desconocido: $featuremap. Usa :Z, :ZZ o :ising, o añade tu caso.")
        end

        # Copia del estado inicial para este sample
        ψ_i = deepcopy(init_state)

        # Aplicar el circuito
        for layer in circuit_i
            for gate in layer
                ψ_i = apply(gate, ψ_i)
            end
        end

        states[i] = ψ_i

        if compute_stats
            Ds[i]     = schmidt(ψ_i, N)
            Renyis[i] = entropy(ψ_i, Int(floor(N/2)))
        end
    end

    if compute_stats
        return states, Ds, Renyis
    else
        return states
    end
end

function ground_state_ising(sites, gamma; nsweeps::Int = 10, maxdim::Int = 32, cutoff::Float64 = 1e-10, J::Float64 = 1.0, periodic::Bool = true)

    N = length(sites)

    # Construimos el OpSum del Hamiltoniano de Ising
    os = OpSum()

    # Término de acoplo Sz_j Sz_{j+1}
    for j in 1:N-1
        os += -J, "Sz", j, "Sz", j+1
    end

    # Opción de condiciones periódicas
    if periodic && N > 2
        os += -J, "Sz", N, "Sz", 1
    end

    # Término de campo transversal -gamma * Σ Sx_j
    for j in 1:N
        os += -gamma, "Sx", j
    end

    # MPO del Hamiltoniano
    H = MPO(os, sites)

    # Estado inicial para DMRG (aleatorio pero razonable)
    psi0 = random_mps(sites; linkdims = min(4, maxdim))

    # Ejecutar DMRG
    energy, psi = dmrg(H, psi0; nsweeps = nsweeps, maxdim = maxdim, cutoff = cutoff)

    return psi
end


end 