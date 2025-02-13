using ITensors
using ITensorMPS
using Random
using .Threads



let
  ITensors.disable_warn_order() 
  N = 200
  sites = siteinds("S=1",N)
  nsweeps = 5
  maxdim = 100
  J=1
  cutoff = [1E-15]

  io1 = open("resultados/DMRG_$N.txt","w")

  os = OpSum()
  for j=1:N-1
    os += "Sz",j,"Sz",j+1
    os += 1/2,"S+",j,"S-",j+1
    os += 1/2,"S-",j,"S+",j+1
  end
  os=os*J
  H = MPO(os,sites)

  psi = random_mps(sites;linkdims=1)
  energy=zeros(nsweeps)
  time=zeros(nsweeps)
  for i in 1:nsweeps
    print(inner(psi',H,psi))
    t=@elapsed energy[i],psi = dmrg(H,psi;nsweeps=1,maxdim,cutoff)
    time[i]=t
  end

  for i in 1:nsweeps
    write(io1,"$i $(energy[i]) $(time[i])\n")
  end

  close(io1)

end