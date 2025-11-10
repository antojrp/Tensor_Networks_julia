using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads

let 
    include("pTEBD.jl")
    using .pTEBD
    include("random_circuit.jl")
    using .random_circuits
    include("qudits.jl")
    using .qudits
    N=4
    L=2
    i=siteinds("Qubit", N)
    circuit=random_circuit(i,N,L,reduced=true)
    blockv=2
    blockh=2    
    reduced_circuit=block_circuit(circuit,i,blockv,blockh)
    for layer in reduced_circuit
        println("Nueva capa:")
        for U in layer
            println(inds(U))
        end
    end
end