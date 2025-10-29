module pTEBD

# Export functions
export vidal_form, mps_from_vidal, apply_layer_parallel!, apply_circuit!

using LinearAlgebra
using ITensors
using ITensorMPS
using .Threads

function vidal_form(mps::MPS, sites::Vector{Index{Int64}})
    N = length(mps)
    @assert length(sites) == N "Debe haber un site index para cada tensor del MPS"

    Gammas = []
    Deltas = []
    ψ = deepcopy(mps)

    # Ortonormalizamos desde la izquierda
    orthogonalize!(ψ, 1)

    # Primer tensor i = 1 (no hay link izquierdo)
    A = ψ[1]
    s = sites[1]
    U, S, V = svd(A, s; cutoff=1e-12)
    push!(Gammas, U)
    push!(Deltas, S)
    ψ[2] = V*ψ[2]

    # Iteramos desde i = 2 hasta N-1
    for i in 2:N-1
        A = ψ[i]
        s = sites[i]
        link_left = inds(A)[1]
        U, S, V = svd(A, (link_left,s); cutoff=1e-12)     
        push!(Gammas, U)
        push!(Deltas, S)
        ψ[i+1] = V*ψ[i+1]
    end

    # Último Gamma
    push!(Gammas, ψ[N])

    return Gammas, Deltas
end

function mps_from_vidal(Gammas::Vector{Any}, Deltas::Vector{Any}, sites::Vector{Index{Int}})
    N = length(Gammas)
    @assert length(Deltas) == N - 1 "Debe haber N-1 tensores Delta para N Gammas"
    @assert length(sites) == N "El número de sitios debe coincidir con el número de Gammas"

    # Construir los tensores del MPS
    tensors = ITensor[]

    for i in 1:N-1
        push!(tensors, Gammas[i] * Deltas[i])
    end
    push!(tensors, Gammas[end])

    # Creamos el MPS vacío con los sitios y linkdims=1 (temporal)
    psi = MPS(sites; linkdims=1)

    # Asignamos los tensores al MPS uno por uno
    for i in 1:N
        psi[i] = tensors[i]
    end

    return psi
end

function apply_layer_parallel!(Gammas::Vector{Any}, Deltas::Vector{Any}, layer::Vector{ITensor}, sites::Vector{Index{Int64}})
    N = length(Gammas)
    @assert length(sites) == N
    affected_sites_all = [[i for i in 1:N if !isempty(commoninds(U, sites[i]))] for U in layer]
    Deltas_inv = Vector{ITensor}(undef, N)


    Deltas_inv = Vector{ITensor}(undef, N)
    for i in 1:N-1
        Deltas_inv[i] = diag_itensor(
            [1 / Deltas[i][k, k] for k in 1:dim(inds(Deltas[i])[1])],
            reverse(inds(Deltas[i]))
        )
    end

    for j in 1:length(layer)

        U = layer[j]

        affected_sites = affected_sites_all[j]

        if length(affected_sites) == 1
            i = affected_sites[1]
            Gammas[i] = noprime(Gammas[i]*U)
            if i != N
                Gammas[i] = permute(Gammas[i], inds(Gammas[i])[1:end-2]..., inds(Gammas[i])[end], inds(Gammas[i])[end-1])
            end    

        elseif length(affected_sites) == 2
            i, ip1 = affected_sites

            Λ_left     = i > 1   ? Deltas[i-1]     : ITensor(1.0)
            Λ_right    = i < N-1 ? Deltas[i+1]     : ITensor(1.0)
            Λ_left_inv  = i > 1   ? Deltas_inv[i-1] : ITensor(1.0)
            Λ_right_inv = i < N-1 ? Deltas_inv[i+1] : ITensor(1.0)


            # Construimos el tensor local Ψ_i,i+1
            Ψ = Λ_left * Gammas[i] * Deltas[i] * Gammas[i+1] * Λ_right

            # Aplicamos la puerta
            Ψ′ = Ψ * U

            # Hacemos SVD para volver a forma canónica
            s1 = prime(sites[i])  # índice físico del sitio i
             Unew, S, Vnew = 
            if i == 1
                svd(Ψ′, s1; cutoff = 1e-12)
            else
                svd(Ψ′, (inds(Ψ′)[1],s1); cutoff = 1e-12)
            end
            Vnew = permute(Vnew, reverse(inds(Vnew))...) 

            Gammas[i] = noprime(Λ_left_inv* Unew)
            Deltas[i]   = S
            Gammas[i+1] = noprime( Vnew * Λ_right_inv)
        else
            error("Puerta actúa sobre más de 2 sitios, no soportado")
        end
    end

end


function apply_circuit!(Gammas::Vector{Any}, Deltas::Vector{Any}, circuit::Vector{Any}, sites::Vector{Index{Int64}}; compute_stats::Bool = false)

    N = length(Gammas)
    # Solo preparamos almacenamiento si se piden las estadísticas
    if compute_stats
        Ds = Float64[]
        Renyis = Float64[]
    end

    times_per_layer = Float64[]

    for (layer_idx, layer) in enumerate(circuit)
        t=@elapsed apply_layer_parallel!(Gammas, Deltas, layer, sites)
        push!(times_per_layer, t)

        if compute_stats && layer_idx % 2 == 0
            link_dim = max_link_dimension(Deltas)
            S2 = renyi2(Deltas[Int(floor(N/2))])
            push!(Ds, link_dim)
            push!(Renyis, S2)
        end
    end

    if compute_stats
        return times_per_layer, Ds, Renyis
    else
        return times_per_layer
    end
end



function renyi2(Delta::ITensor)
    # Convertimos el ITensor a vector de valores singulares
    sv_vector = vec(diag(Delta))  # asumimos que es diagonal
       
    # Entropía de Rényi-2
    S2 = -log2(sum(abs2.(sv_vector).^2))    
    return S2
end

function max_link_dimension(singular_values::Vector{Any})
    max_dim = 0
    for sv_tensor in singular_values
        # Convertimos a vector para obtener la longitud
        sv_vector = vec(diag(sv_tensor))
        link_dim = length(sv_vector)
        if link_dim > max_dim
            max_dim = link_dim
        end
    end
    return max_dim
end


end 