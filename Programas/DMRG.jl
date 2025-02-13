using ITensors
using ITensorMPS
using Random
using .Threads



let
  ITensors.disable_warn_order() 
  N = 1000
  sites = siteinds("S=1",N)
  nsweeps = 5
  dim = [10,20,30,40,100]
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
  psi = dmrg(H,psi;nsweeps=1,maxdim=10,cutoff)

  psi = random_mps(sites;linkdims=1)
  write(io1,"0 $(inner(psi',H,psi)) \n")
  energy=zeros(nsweeps)
  time=zeros(nsweeps)
  for i in 1:nsweeps
    print(inner(psi',H,psi))
    t=@elapsed energy[i],psi = dmrg(H,psi;nsweeps=1,maxdim=dim[i],cutoff)
    time[i]=t
  end

  total_time=0
  for i in 1:nsweeps
    total_time=total_time+time[i]
    write(io1,"$i $(energy[i]) $(time[i])\n")
  end

  close(io1)

end