module qudits

# Export functions
export block_circuit, qubit_to_qudit

using LinearAlgebra
using ITensors
using ITensorMPS
using .Threads

function block_circuit(circuit, sites, blockv, blockh)
    N = length(sites)
    L = length(circuit)

    reduced = []  # estructura por capas

    h = 1
    while h <= L
        hmax = min(h + blockh - 1, L)
        layer_reduced = ITensor[]  # esta será la capa reducida

        v = 1
        while v <= N
            vmax = min(v + blockv - 1, N)
            contracted_block = ITensor(1.0)

            # Recorremos las capas dentro del bloque horizontal
            for l in h:hmax
                layer = circuit[l]
                for U in layer

                    half_inds = inds(U)
                    affected = [i for i in 1:N if !isempty(commoninds(noprime(half_inds), sites[i]))]

                    # si toca el bloque vertical actual
                    if any((v .<= affected .<= vmax))
                        # contraer directamente
                        contracted_block = contracted_block * U
                        circuit[l] = filter(g -> g != U, circuit[l])
                    end
                end
            end
            push!(circuit[hmax], contracted_block)

            # avanzar verticalmente
            v = vmax + 1
        end

        push!(reduced, layer_reduced)

        # avanzar horizontalmente
        h = hmax + 1
    end
    circuit = filter(layer -> !isempty(layer), circuit)
    for (l, layer) in enumerate(circuit)
        for U in layer
            for k in 1:2*(l-1)
                prime!(U,-1)
            end
        end
    end

    return circuit
end

function qubit_to_qudit(circuit, qubit_sites, qudit_sites, blockv, blockh)

    circuit = block_circuit(circuit, qubit_sites, blockv, blockh)
    N = length(qubit_sites)

    # Crear combiners por bloques verticales
    combiners = ITensor[]
    deltas = ITensor[]
    v = 1
    contador = 1
    while v <= N
        vmax = min(v + blockv - 1, N)
        C = combiner(qubit_sites[v:vmax]...)
        kr = delta(inds(C)[1], qudit_sites[contador])
        push!(combiners, C)
        push!(deltas, kr)
        v = vmax + 1
        contador += 1
    end

    # Aplicar combiners a las puertas que afecten a cada bloque
    for i in 1:length(circuit)
    layer = circuit[i]
        for j in 1:length(layer)
            U = layer[j]
            for (delta, C) in zip(deltas, combiners)
                if !isempty(commoninds(inds(U), inds(C)))
                    U = U * C * delta
                    C_aux = C
                    delta_aux = delta
                    for _ in 1:blockh
                        C_aux = prime(C_aux)
                        delta_aux = prime(delta_aux)
                    end
                    U = U * C_aux * delta_aux

                end
            end
            for (i, ind) in enumerate(inds(U))
                if iseven(i)
                    replaceind!(U, ind, setprime(ind, 1))
                end
            end
            circuit[i][j] = U 
        end
    end

    return circuit
end

end