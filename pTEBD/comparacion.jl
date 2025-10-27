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
    BLAS.set_num_threads(1)
    println("Threads activos: ", nthreads())
    println("Threads LinearAlgebra activos: ", BLAS.get_num_threads())
    N=40
    L=10
    i=siteinds("Qubit", N)
    phi=MPS(i,"0")
    Gammas, Deltas = vidal_form(phi, i)
    circuit=random_circuit(i,N,L)
    @time apply_circuit!(Gammas, Deltas, circuit, i)

    c=Gammas[20]*Deltas[20]
    println("")
    println("Tiempo 1")
    @time c=Gammas[20]*Deltas[20]
    println("Gamma: ", Base.summarysize(Gammas[20]), " Deltas ", Base.summarysize(Deltas[20]), " bytes")

    D=2^10
    i=Index(D)
    j=Index(2)
    k=Index(D)
    l=Index(D)
    Gamma2=ITensor[]
    Delta2=ITensor[]
    for m in 1:40
        A=random_itensor(i,k,j)
        B=random_itensor(k,l)
        push!(Gamma2,A)
        push!(Delta2,B)
    end


    C=Gamma2[20]*Delta2[20] 
    println("")
    println("Tiempo 2")
    @time C=Gamma2[20]*Delta2[20]
    println("Gamma2: ", Base.summarysize(Gamma2[20]), " Deltas2", Base.summarysize(Delta2[20]), " bytes")

    display(Gammas[20])
    display(Deltas[20])
    display(Gamma2[20])
    display(Delta2[20])
end