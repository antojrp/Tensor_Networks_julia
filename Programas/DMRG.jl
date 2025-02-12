using ITensors
using ITensorMPS
using Random
using .Threads



let
  ITensors.disable_warn_order() 
  N = 100
  sites = siteinds("S=1",N)
  nsweeps = 5
  maxdim = 10
  cutoff = [1E-10]

  os = OpSum()
  for j=1:N-1
    os += "Sz",j,"Sz",j+1
    os += 1/2,"S+",j,"S-",j+1
    os += 1/2,"S-",j,"S+",j+1
  end
  H = MPO(os,sites)

  psi = random_mps(sites;linkdims=10)
  for i in 1:nsweeps
    energy,psi = dmrg(H,psi;1,maxdim,cutoff)
  end

end