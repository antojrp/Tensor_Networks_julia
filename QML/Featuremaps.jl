module Featuremaps

# Export functions
export Zfeaturemap, ZZfeaturemap_linear, IsingFeaturemap_linear

using LinearAlgebra
using ITensors
using ITensorMPS

function layerH(N, i1)
    l = ITensor[]
    for j in 1:N
        gate = op("H", i1[j])   # Hadamard en el qubit j
        push!(l, gate)
    end
    return l
end

function layerZ(N, i1, x)
    l = ITensor[]
    for j in 1:N
        gate = op("Rz", i1[j]; θ = x[j])  # Rotación Rz(θ_j) en el qubit j
        push!(l, gate)
    end
    return l
end

function Zfeaturemap(i1, N, L, x; reduced=false)
    circuit = Any[]

    for _ in 1:L
        # Primera capa: Hadamards sobre i1
        layer = layerH(N, i1)
        push!(circuit, layer)

        # Escogemos qué índices usar para la capa Z
        i2 = reduced ? i1' : i1

        # Segunda capa: Rz(θ) sobre i2
        layer = layerZ(N, i2, x)
        push!(circuit, layer)

    end

    return circuit
end

function entanglement_linear(N, i1, x, m)
    l = ITensor[]

    # j recorre m, m+2, m+4, ... hasta N-1
    for j in m:2:N-1
        θ = 2 * (pi - x[j]) * (pi - x[j+1])

        # Bloque ZZ entre qubits j y j+1
        # Aquí asumo que tienes definida una puerta CNOT(iA, iB)
        push!(l, op("CNOT",i1[j], i1[j+1]))          # CNOT control j, target j+1
        push!(l, op("Rz", i1[j+1]; θ = θ))      # Fase dependiente de x
        push!(l, op("CNOT",i1[j], i1[j+1]))          # CNOT de vuelta
    end

    return l
end

function ZZfeaturemap_linear(i1, N, L, x, reduced=false)
    circuit = Any[]

    if L == 0
        circuit = Zfeaturemap(i1, N, L, x; reduced=reduced)
        return circuit
    else
        for r in 1:L
            # 1) Capa de Hadamards
            layer = layerH(N, i1)
            push!(circuit, layer)

            # 2) Capa local de Z con los datos
            layer = layerZ(N, i1, x)
            push!(circuit, layer)

            # 3) Capa ZZ con entanglement linear alternando m
            i2 = reduced ? i1' : i1

            m = 2 - (r % 2)

            layer = entanglement_linear(N, i2, x, m)
            push!(circuit, layer)


            i1 = reduced ? i2' : i2
        end
        return circuit
    end
end

function layerX(N, i1, x)
    l = ITensor[]
    for j in 1:N
        θ = x[j]
        push!(l, op("Rx", i1[j]; θ = θ))
    end
    return l
end

function IsingFeaturemap_linear(i1, N, L, x; reduced=false)
    circuit = Any[]
    if L == 0
        push!(circuit, layerH(N, i1))
        push!(circuit, layerX(N, i1, x))
        return circuit
    else
        for r in 1:L
            # Hadamards
            push!(circuit, layerH(N, i1))

            # Rx con datos
            push!(circuit, layerX(N, i1, x))

            # ZZ igual que en tu ZZfeaturemap_linear
            i2 = reduced ? i1' : i1
            m = 2 - (r % 2)
            push!(circuit, entanglement_linear(N, i2, x, m))
            i1 = reduced ? i2' : i2
        end
        return circuit
    end
end


end