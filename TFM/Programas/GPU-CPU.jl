using ITensors
using LinearAlgebra
using ITensorMPS
using Random
using .Threads

function imprimir(time,io1,nsim,ndims,D_i,paso)
    tiempos=zeros(ndims)
    total=zeros(nsim)
    media=0
    for i in 1:ndims
      write(io1,"$(D_i+(i-1)*paso)")
      for k in 1:nsim
        tiempos[i]=tiempos[i]+time[k,i]
        total[k]=total[k]+time[k,i]
        media=media+time[k,i]
      end
      varianzat=0
      tiempos[i]=tiempos[i]/(nsim)
      for k in 1:nsim
        varianzat=varianzat+(time[k,i]-tiempos[i])^2
      end
      varianzat=(varianzat/(nsim))^0.5
      write(io1,"  $(tiempos[i])  $varianzat")
      write(io1,"\n")
      write(io1,"\n")
    end
    media=media/(nsim)
    varianza=0
    for i in 1:nsim
      varianza=varianza+(media-total[i])^2
    end

    varianza=(varianza/(nsim-2))^0.5
    write(io1,"Total= $media\n")
    write(io1,"Varianza= $varianza\n\n")
end  

let
  D_i=100
  paso=100
  ndims=10
  nsim=10
  io1 = open("resultados/GPU-CPU_energia_t$(BLAS.get_num_threads()).txt","w")
  close(io1)

  time=zeros(nsim,ndims)
  for i in 1:ndims
    dim=D_i+(i-1)*paso
    io1 = open("resultados/GPU-CPU_energia_t$(BLAS.get_num_threads()).txt","a")
    write(io1,"D   Time  var(Time)  \n")
    l = Index(dim)
    m = Index(dim)
    n = Index(dim)
    for k in 1:nsim
      A = random_itensor(l, m)
      B = random_itensor(m, n)
      t= @elapsed A * B
      time[k,i]=t
    end
  end
  imprimir(time,io1,nsim,ndims,D_i,paso)
  close(io1)
  
  using CUDA # This will trigger the loading of `NDTensorsCUDAExt` in the 
  time=zeros(nsim,ndims)
  for i in 1:ndims
    dim=D_i+(i-1)*paso
    io1 = open("resultados/GPU-CPU_energia_t$(BLAS.get_num_threads()).txt","a")
    write(io1,"D   Time  var(Time)  \n")
    l = Index(dim)
    m = Index(dim)
    n = Index(dim)
    for k in 1:nsim
      A = random_itensor(l, m)
      B = random_itensor(m, n)
      Acu = cu(A)
      Bcu = cu(B)
      t=@elapsed Acu * Bcu
      time[k,i]=t
    end
  end
  imprimir(time,io1)
  close(io1)  
end