using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads
using Statistics

let 
    include("pTEBD.jl")
    using .pTEBD
    include("random_circuit.jl")
    using .random_circuits
    BLAS.set_num_threads(6)
    println("Threads activos: ", nthreads())
    println("Threads LinearAlgebra activos: ", BLAS.get_num_threads())

    N=20
    L=15
    num_runs = 10
    compute_stats=true
    coeficientes=true

    if coeficientes
        io= open("resultados/coeficientes_$(L).txt", "w")
        write(io,"Numero de qubits: $N \n")
        close(io)
    end

    if compute_stats
        io= open("resultados/entrelazamiento_$(L).txt", "w")
        close(io)
    end

    io= open("resultados/tiempos_$(L).txt", "w")
    close(io)

    sites=siteinds("Qubit", N)
    phi=MPS(sites,"0")
    Gammas, Deltas = vidal_form(phi, sites)
    circuit=random_circuit(sites,N,L)
    apply_circuit!(Gammas, Deltas, circuit, sites)

    D = [Float64[] for _ in 1:L]
    Renyi = [Float64[] for _ in 1:L]
    times = [Float64[] for _ in 1:2*L]
   
    for run in 1:num_runs
        println("Run $run / $num_runs")
        Gammas, Deltas = vidal_form(phi, sites)
        circuit=random_circuit(sites,N,L)
        if compute_stats
            layer_times, Ds, Renyis = apply_circuit!(Gammas, Deltas, circuit, sites; compute_stats=compute_stats)
            
            for (l, Dval, Rval) in zip(1:L, Ds, Renyis)
                push!(D[l], Dval)
                push!(Renyi[l], Rval)
            end
        else
            layer_times = apply_circuit!(Gammas, Deltas, circuit, sites; compute_stats=compute_stats)
            for (l, t) in zip(1:2*L, layer_times)
                push!(times[l], t)
            end
        end
        if coeficientes
            io= open("resultados/coeficientes_$(L).txt", "a")
            for k in 1:dim(inds(Deltas[Int(floor(N/2))])[1])
                val=Deltas[Int(floor(N/2))][k, k]^2
                write(io, "$(val) \n")
            end
            write(io, "\n")
            close(io)
        end 
    end
    if compute_stats
        #io = open("resultados/entrelazamiento_$(L)_t$(BLAS.get_num_threads()).txt", "w")
        io = open("resultados/entrelazamiento_$(L).txt", "a")
        write(io, "Número de qubits: $(length(Gammas))  \n")
        write(io, "Layer\tD\tvar(D)\tRenyi\tvar(Renyi)\n")

        for l in 1:L
            meanD = mean(D[l])
            varD  = var(D[l])
            meanR = mean(Renyi[l])
            varR  = var(Renyi[l])
            write(io, "$l  $meanD  $varD  $meanR  $varR \n")
        end
        close(io)
    end

    io_time = open("resultados/tiempos_$(L).txt", "a")
    write(io_time, "Número de qubits: $(length(Gammas))\n")
    write(io_time, "Layer\tTime\tvar(Time)\n")

    for (l, Tvals) in enumerate(times)
        meanT = mean(Tvals)
        varT  = var(Tvals)
        write(io_time, "$l  $meanT  $varT\n")
    end

    total_time = sum(mean.(times))
    varianza_total = mean(var.(times))
    write(io_time, "Total= $total_time\n")
    write(io_time, "Varianza= $varianza_total\n")
    close(io_time)
end