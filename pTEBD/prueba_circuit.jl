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
    BLAS.set_num_threads(6)
    println("Threads activos: ", nthreads())
    println("Threads LinearAlgebra activos: ", BLAS.get_num_threads())
    N=40
    L=10
    i=siteinds("Qubit", N)
    phi=MPS(i,"0")
    Gammas, Deltas = vidal_form(phi, i)

    circuit=random_circuit(i,N,L)
    @time apply_circuit!(Gammas, Deltas, circuit, i)
    Gammas, Deltas = vidal_form(phi, i)
    @time apply_circuit!(Gammas, Deltas, circuit, i)
    Gammas, Deltas = vidal_form(phi, i)
    @time apply_circuit!(Gammas, Deltas, circuit, i)
    

    phi=MPS(i,"0")
    @time for layer in circuit
        phi=apply(layer,phi)
    end
    phi=MPS(i,"0")
    @time for layer in circuit
        phi=apply(layer,phi)
    end
    phi=MPS(i,"0")
    @time for layer in circuit
        phi=apply(layer,phi)
    end

    phi_reconstructed = mps_from_vidal(Gammas, Deltas, i)
    @assert abs(inner(phi,phi_reconstructed))^2 - 1 < 0.00001 "El circuito fallo"
    println("El circuito fue exitoso:",abs(inner(phi,phi_reconstructed))^2)


    total_gammas = sum(Base.summarysize(g) for g in Gammas)
    println("Gammas ocupan: ", total_gammas, " bytes")
    total_deltas = sum(Base.summarysize(d) for d in Deltas)
    println("Deltas ocupan: ", total_deltas, " bytes")
    total_phi = Base.summarysize(phi)
    println("MPS original ocupa: ", total_phi, " bytes")


end