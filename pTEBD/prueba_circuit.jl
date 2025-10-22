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
    N=10
    L=5
    i=siteinds("Qubit", N)
    phi=MPS(i,"0")
    Gammas, Deltas = vidal_form(phi, i)
    circuit=random_circuit(i,N,L)
    apply_circuit!(Gammas, Deltas, circuit, i)
    for layer in circuit
        phi=apply(layer,phi)
    end
    phi_reconstructed = mps_from_vidal(Gammas, Deltas, i)
    @assert abs(inner(phi,phi_reconstructed))^2 - 1 < 0.00001 "El circuito fallo"
    println("El circuito fue exitoso:",abs(inner(phi,phi_reconstructed))^2)

end