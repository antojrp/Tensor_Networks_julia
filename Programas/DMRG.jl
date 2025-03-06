using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads



let
  ITensors.disable_warn_order() 
  Ni = 100
  m=5
  nsim=10
  paso=100
  nsweeps = 5
  dim = [10,20,30,40,100]
  J=1
  g=10
  cutoff = [1E-15]

  io1 = open("resultados/DMRG_energia_$nsweeps.txt","w")
  io2 = open("resultados/DMRG_tiempo_$nsweeps.txt","w")
  close(io1)
  close(io2)  
  os1 = OpSum()
  os2 = OpSum()

  for num in 1:m
    N=Ni+(num-1)*paso

    io1 = open("resultados/DMRG_energia_$nsweeps.txt","a")
    io2 = open("resultados/DMRG_tiempo_$nsweeps.txt","a")
    print("Numero de qubits: $N \n")
    print("Threads (LinearAlgebra) :$(BLAS.get_num_threads())\n")
    print("Threads (Julia) :$(Threads.nthreads())\n\n\n")
    write(io1,"Numero de qubits: $N \n")
    write(io2,"Numero de qubits: $N \n")
    write(io1,"Layer   Energy  var(Energy)  \n")
    write(io2,"Layer   Time var(Time)\n")

    sites = siteinds("Qubit",N)
    for j=1:N-1
      os1 += "Z",j,"Z",j+1

    end
    for j=1:N
      os2 += "X",j
    end
    os1=-os1*J
    os2=-os2*J*g
    H = MPO(os1+os2,sites)

    energy=zeros(nsim,nsweeps)
    time=zeros(nsim,nsweeps)


    for k in 1:nsim
      psi = random_mps(sites;linkdims=1)
      write(io1,"0 $(inner(psi',H,psi)) \n")
      for i in 1:nsweeps
        print(inner(psi',H,psi))
        t=@elapsed energy[k,i],psi = dmrg(H,psi;nsweeps=1,maxdim=dim[i],cutoff)
        time[k,i]=t
      end
    end

    tiempos=zeros(nsweeps)
    energias=zeros(nsweeps)
    total=zeros(nsim)
    media=0
    for i in 1:nsweeps
      write(io1,"$i")
      write(io2,"$i")
      for k in 1:nsim
        tiempos[i]=tiempos[i]+time[k,i]
        energias[i]=energias[i]+energy[k,i]

        total[k]=total[k]+time[k,i]
        media=media+time[k,i]
      end
      varianzat=0
      varianzae=0
      tiempos[i]=tiempos[i]/(nsim)
      energias[i]=energias[i]/(nsim)
      for k in 1:nsim
        varianzat=varianzat+(time[k,i]-tiempos[i])^2
        varianzae=varianzae+(energy[k,i]-energias[i])^2
      end
      varianzat=(varianzat/(nsim))^0.5
      varianzae=(varianzae/(nsim))^0.5
      write(io1,"  $(energias[i])  $varianzae")
      write(io2,"  $(tiempos[i])  $varianzat")
      write(io1,"\n")
      write(io2,"\n")
    end
    media=media/(nsim)
    varianza=0
    for i in 1:nsim
      varianza=varianza+(media-total[i])^2
    end

    varianza=(varianza/(nsim-2))^0.5
    write(io2,"Total= $media\n")
    write(io2,"Varianza= $varianza\n\n")

    close(io1)
    close(io2)
  end
end