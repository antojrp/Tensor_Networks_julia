using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using Statistics
using Base.Threads


BLAS.set_num_threads(1)

let
  ITensors.disable_warn_order()

  #Initialize state to |0>
  function init_state(N,i)
    phi=[]
    for j in 1:N
      j=ITensor(i[j])
      j[1]=1
      push!(phi,j)
    end
    return phi
  end

  #Contract state
  function state(phi,N)
    aux=phi[1]
    for i in 2:N
      aux=aux*phi[i]
    end
    return aux
  end

  #Create 1-qubit random gate given index i
  function random_1gate(i,random,j)
    s=rand(1:2)
    s=(s+random[j])%3
    random[j]=s
    if s == 0
      gate=op("Rx", i;θ=π/2)
    
    elseif s == 1
      gate=op("Ry", i;θ=π/2)

    else
      gate=op("W", i)

    end

    return gate
  end


  ITensors.op(::OpName"fSim",::SiteType"Qubit")= [
      1  0                 0                 0
      0  cos(π/2)           -im*sin(π/2)        0
      0  -im*sin(π/2)       cos(π/2)            0
      0  0                 0                 exp(-im*π/6)
  ]

  ITensors.op(::OpName"W",::SiteType"Qubit")= (1/sqrt(2)) * [1 -sqrt(im); sqrt(-im) 1]

  

  #Create 2-qubit random gate given index i,j
  function random_2gate(i,j)
    gate = op("fSim",i,j)
    return gate
  end

  #Create a layer of 1-qubit random gate given the number ob qubits (N) and a array of index i1
  function layer1(N,i1,random)
    l=ITensor[]
    for j in 1:N
      gate=random_1gate(i1[j],random,j)
      push!(l,gate)
    end
    return l
  end

  #Create a layer of 2-qubit random gate given the number ob qubits (N) and a array of index i1, m {1,2} is the qubit in which begins to apply the gates
  function layer2(N,i1,m)
    l=ITensor[]
    if m==2
      gate=delta(i1[1],i1[1]')
      push!(l,gate)
    end
    for j in m:2:N-1
      gate=random_2gate(i1[j],i1[j+1])
      push!(l,gate)
    end
    if m==2
      if N%2==0
        gate=delta(i1[N],i1[N]')
        push!(l,gate)
      end
    end
    if m==1
      if N%2==1
        gate=delta(i1[N],i1[N]')
        push!(l,gate)
      end
    end
    return l
  end

  #Creates a random circuit of L layers, each layer is composed by a 1-qubit layer and 2-qubit layer.
  function random_circuit(i1,N,L)
    circuit=[]
    random=rand(0:2,N)
    for i in 1:L
      layer=layer1(N,i1,random)
      push!(circuit,layer)
      #i2=i1'
      i2=i1
      layer=layer2(N,i2,2-(i%2))
      push!(circuit,layer)
      #i1=i2'
      i1=i2
    end
    return circuit
  end



function reduce_circuit(circuit, L)
    reduced_circuit = Vector{Vector{ITensor}}(undef, L)
    #t_start_total = time()  # tiempo total de todas las capas

    # Paralelizamos sobre capas
    @threads for l in 1:L
        #t_start_layer = time()  # tiempo de inicio de la capa

        layer1 = circuit[2*(l-1)+1]
        layer2_list = [prime(g) for g in circuit[2*(l-1)+2]]
        n = length(layer2_list)
        reduced_layer = Vector{ITensor}(undef, n)

        #iteration_times = zeros(Float64, n)  # para medir cada puerta

        # Paralelizamos las puertas de 2 qubits dentro de la capa
        @threads for idx2 in 1:n
            #t_start = time()

            g2 = layer2_list[idx2]

            # Contraer con todas las puertas de 1 qubit que compartan índices
            for g1 in layer1
                if !isempty(intersect(inds(g1), inds(g2)))
                    g2 = g1 * g2
                end
            end

            #for idx in inds(g2)
            #  if primed(idx) == 2  # si el índice está primado doble
            #      replaceind!(g2, idx, prime(idx, 1))  # reemplaza i'' → i'
            #  end
            #end
            reduced_layer[idx2] = g2
            #iteration_times[idx2] = time() - t_start
        end

        
        # Guardamos resultados de esta capa
        reduced_circuit[l] = reduced_layer

        #t_layer = time() - t_start_layer
        #println("Capa $l - tiempo total: $(round(t_layer, digits=4)) s")
        #println("Capa $l - tiempo promedio por puerta: $(round(mean(iteration_times), digits=6)) s")
    end

    #t_total = time() - t_start_total
    #println("Tiempo total de todas las capas: $(round(t_total, digits=4)) s")

    return reduced_circuit
end

function circuit_to_qdits(circuit, i_qubits, i_qudits, d)
    N = length(i_qubits)
    n_qudits = length(i_qudits)
    @assert N == n_qudits * d "El número de qubits ($N) debe ser n_qudits * d ($(n_qudits*d))."

    # --- Crear combiners de entrada y salida ---
    combiners = Vector{Tuple{ITensor, ITensor}}(undef, n_qudits)
    qubit_groups = Vector{Vector{Index}}(undef, n_qudits)

    i_qubits_p = prime.(prime.(i_qubits))  # índices de salida

    for k in 1:n_qudits
        subinds_in  = i_qubits[(d*(k-1)+1):(d*k)]
        subinds_out = i_qubits_p[(d*(k-1)+1):(d*k)]

        comb_in  = combiner(subinds_in...)
        comb_out = combiner(subinds_out...)

        combiners[k] = (comb_in, comb_out)
        qubit_groups[k] = vcat(subinds_in, subinds_out)
    end

    # --- Aplicar combiners al circuito ---
    circuit_qdits = Vector{Vector{ITensor}}(undef, length(circuit))

    for (l, layer) in enumerate(circuit)
        new_layer = ITensor[]

        for k in 1:n_qudits
            group_inds = Set(qubit_groups[k])
            gates_in_group = [g for g in layer if !isempty(intersect(inds(g), group_inds))]

            if !isempty(gates_in_group)
                # Multiplicamos todas las puertas que actúan en esos qubits
                combined_gate = prod(gates_in_group)

                # Aplicamos ambos combiners (entrada y salida)
                comb_in, comb_out = combiners[k]
                g_qdit = comb_in * combined_gate
                g_qdit = g_qdit * comb_out

                push!(new_layer, g_qdit)
            end
        end

        circuit_qdits[l] = new_layer
    end

    return circuit_qdits
end



  L=2
  N=4
  i=siteinds("Qubit", N) 
  circuit=random_circuit(i,N,L)

  circuit = reduce_circuit(circuit, L) 

  finalcircuit=circuit_to_qdits(circuit, i, siteinds("Qudit", N÷2, dim=4), 2)

  @show finalcircuit

end