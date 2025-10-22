using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads

let 
    include("pTEBD.jl")
    using .pTEBD
    i=siteinds("Qubit", 20)
    phi=random_mps(i; linkdims=6)
    Gammas, Deltas = vidal_form(phi, i)
    phi_reconstructed = mps_from_vidal(Gammas, Deltas, i)
    @assert inner(phi,phi_reconstructed) - 1 < 0.00001 "La reconstrucción del MPS falló"
    println("La reconstrucción del MPS fue exitosa")
end