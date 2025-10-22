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

    N=5
    i=siteinds("Qubit", N)
    phi=random_mps(i; linkdims=6)
    Gammas, Deltas = vidal_form(phi, i)
    layer=layer2(N,i,1)
    apply_layer_parallel!(Gammas, Deltas, layer, i)
    phi=apply(layer,phi)
    phi_reconstructed = mps_from_vidal(Gammas, Deltas, i)
    @assert abs(inner(phi,phi_reconstructed))^2 - 1 < 0.00001 "El circuito fallo"
    println("El circuito fue exitoso", abs(inner(phi,phi_reconstructed))^2)

end