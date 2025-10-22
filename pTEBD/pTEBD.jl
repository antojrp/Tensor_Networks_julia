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

    Gammas = ITensor[]
    Deltas = ITensor[]
    ψ = deepcopy(mps)

    # Ortonormalizamos desde la izquierda
    orthogonalize!(ψ, 1)

    # Primer tensor i = 1 (no hay link izquierdo)
    A = ψ[1]
    s = sites[1]
    U, S, V = svd(A, s; cutoff=1e-12)
    push!(Gammas, U)
    push!(Deltas, S)
    ψ[2] *= V

    # Iteramos desde i = 2 hasta N-1
    for i in 2:N-1
        A = ψ[i]
        s = sites[i]
        link_left = inds(A)[3]
        U, S, V = svd(A, (s, link_left); cutoff=1e-12)
        push!(Gammas, U)
        push!(Deltas, S)
        ψ[i+1] *= V
    end

    # Último Gamma
    push!(Gammas, ψ[N])

    return Gammas, Deltas
end

function mps_from_vidal(Gammas::Vector{ITensor}, Deltas::Vector{ITensor}, sites::Vector{Index{Int}})
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

function apply_layer_parallel!(Gammas::Vector{ITensor}, Deltas::Vector{ITensor}, layer::Vector{ITensor}, sites::Vector{Index{Int64}})
    N = length(Gammas)
    @assert length(sites) == N
    affected_sites_all = [[i for i in 1:N if any(ind -> ind in inds(U), [sites[i]])] for U in layer]
    Deltas_inv = Vector{ITensor}(undef, N)
    for i in 1:N-1
            Deltas_inv[i] = diag_itensor(
                [1 / Deltas[i][k, k] for k in 1:dim(inds(Deltas[i])[1])],
                inds(Deltas[i])
            )
    end

    @threads for j in 1:length(layer)
        #println("Aplicando puerta ", j, " en thread ", threadid())
        U = layer[j]
        #inds_U = inds(U)

        # Determinamos los sitios afectados comparando con los índices del MPS
        #affected_sites = [i for i in 1:N if any(ind -> ind in inds_U, [sites[i]])]
        affected_sites = affected_sites_all[j]

        if length(affected_sites) == 1
            i = affected_sites[1]
            Gammas[i] = noprime(Gammas[i]*U)
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
            if i==1
                Unew, S, Vnew = svd(Ψ′, s1; cutoff=1e-12)
            else
                #link = commonind(Ψ′, Gammas[i-1])
                link = findindex(Ψ′, "Link,u")

                Unew, S, Vnew = svd(Ψ′, (s1, link); cutoff=1e-12)
            end 

            # Actualizamos tensores
            #if length(inds(Λ_left)) == 0
            #    Λ_left_inv = ITensor(1.0)
            #else
            #    Λ_left_inv  = diag_itensor([1/Λ_left[i,i] for i in 1:dim(inds(Λ_left)[1])], inds(Λ_left))
            #end

            #if length(inds(Λ_right)) == 0
            #    Λ_right_inv = ITensor(1.0)
            #else
            #    Λ_right_inv  = diag_itensor([1/Λ_right[i,i] for i in 1:dim(inds(Λ_right)[1])], inds(Λ_right))
            #end

            Gammas[i]   = noprime(Unew * Λ_left_inv)
            Deltas[i]   = S
            Gammas[i+1] = noprime(Vnew * Λ_right_inv)

            #println("Puerta aplicada entre sitios ", i, " y ", ip1)
        else
            error("Puerta actúa sobre más de 2 sitios, no soportado")
        end
    end
end

function apply_circuit!(Gammas::Vector{ITensor}, Deltas::Vector{ITensor}, circuit::Vector{Any}, sites::Vector{Index{Int64}})
    for layer in circuit
        #println("Aplicando capa del circuito...")
        apply_layer_parallel!(Gammas, Deltas, layer, sites)
    end
end


end 