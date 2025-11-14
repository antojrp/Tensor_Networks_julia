using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads

let 
    ITensors.disable_warn_order()
    include("pTEBD.jl")
    using .pTEBD
    include("random_circuit.jl")
    using .random_circuits
    include("qudits.jl")
    using .qudits
    N=15
    L=2
    d=3
    blockv=d
    blockh=2 

    sites=siteinds("Qubit", N)
    qudit_sites=siteinds("Qudit", Int(N/d), dim=2^d)
    circuit=random_circuit(sites,N,L,reduced=true)
    qudit_circuit=qubit_to_qudit(circuit,sites, qudit_sites, d, blockh)
    for (nl, layer) in enumerate(qudit_circuit)
    println("Layer $nl:")
        for (k, U) in enumerate(layer)
            println("Gate $k:")
            println(inds(U))
        end
    end
    phi=MPS(qudit_sites,"0")
    Gammas, Deltas = vidal_form(phi, qudit_sites)
    apply_circuit!(Gammas, Deltas, qudit_circuit, qudit_sites)

end