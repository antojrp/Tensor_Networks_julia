using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads

let 
    include("pTEBD(debug).jl")
    using .pTEBD
    include("random_circuit.jl")
    using .random_circuits
    BLAS.set_num_threads(1)
    println("Threads activos: ", nthreads())
    println("Threads LinearAlgebra activos: ", BLAS.get_num_threads())
    N=50
    L=10
    i=siteinds("Qubit", N)
    phi=MPS(i,"0")
    Gammas, Deltas = vidal_form(phi, i)
    circuit=random_circuit(i,N,L)
    @time apply_circuit!(Gammas, Deltas, circuit, i)
    circuit=random_circuit(i,N,L)
    Gammas, Deltas = vidal_form(phi, i)
    @time apply_circuit!(Gammas, Deltas, circuit, i)
end