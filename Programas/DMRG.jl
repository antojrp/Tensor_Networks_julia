using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads



let

  function entropy(psi,b)
    psi = orthogonalize(psi, b)
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    SvN = 0.0
    #print("Dimension: $(size(S, 1)) \n")
    for n in 1:size(S, 1)
      p = S[n,n]^2
      SvN -= p * log(p)
    end
    return SvN
  end

  function schmidt(state,N)
    # suma=0
    max=0
    for i in 1:N-1
      let
        if max < dim(inds(state[i])[end])
          max=dim(inds(state[i])[end])
        end
      end
    end
    # return suma/(N-1)
    return max
  end


  ITensors.disable_warn_order() 
  N=50
  nsim=20
  nsweeps=10
  nm=21
  J=1
  p=8
  maxdim=2^p

  cutoff = [1E-15]

  io1 = open("resultados/DMRG_L$(N)_8.txt","w")
  write(io1,"Gamma, E, varE, T, varT, dimension, vardimension \n")
  close(io1)

  for gamma in range(0, stop=2, length=nm)

    io1 = open("resultados/DMRG_L$(N)_8.txt","a")
    print("Numero de qubits: $N \n")
    print("Dimension max: $maxdim \n")
    print("p= $p \n")

    sites = siteinds("Qubit",N)
    os1 = OpSum()
    os2 = OpSum()
    for j in 1:N-1
      os1 += "Sz",j,"Sz",j+1
    end
    for j in 1:N
      os2 += "Sx",j
    end
    os1 += "Sz",1,"Sz",N
    os1=-os1*J
    os2=-os2*gamma
    H = MPO(os1+os2,sites)

    energy=0
    time=0
    tm=0
    Em=0
    vart=0
    varE=0
    entropym=0
    varentropy=0
    dimension=0
    vardimension=0

    for k in 1:nsim
      psi = random_mps(sites;linkdims=2^4)
      #print(inner(psi',H,psi))
      t=@elapsed for i in 1:nsweeps
        energy,psi = dmrg(H,psi;nsweeps=1,maxdim=maxdim,cutoff)
      end
      s = entropy(psi, round(Int, N/2))
      d=schmidt(psi,N)
      tm=tm+t
      Em=Em+energy
      entropym=entropym+s
      dimension=dimension+d
      vart=vart+t^2
      varE=varE+energy^2
      vardimension=vardimension+d^2
      varentropy=varentropy+(s)^2
    end
    Em /= nsim
    tm /= nsim
    dimension /= nsim
    entropym /= nsim
    varE = varE/nsim - Em^2
    vart = vart/nsim - tm^2
    varentropy = varentropy/nsim - entropym^2
    vardimension = vardimension/nsim - dimension^2

    write(io1,"$gamma $Em $varE $tm $vart $entropym $varentropy $dimension $vardimension\n")
    close(io1)
  end
end