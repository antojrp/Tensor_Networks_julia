using ITensors
using LinearAlgebra
using ITensorMPS
using Random
using .Threads

function imprimir(time,io1)
    tiempos=zeros(ndims)
    total=zeros(nsim)
    media=0
    for i in 1:ndims
      write(io1,"$(dim_i+(i-1)*pasod))")
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
  ndim=10
  nsim=10
  io1 = open("resultados/DMRG_energia_D$(D_i)_t$(BLAS.get_num_threads()).txt","w")
  close(io1)


  for i in 1:ndims
    dim=dim_i+(i-1)*pasod
    io1 = open("resultados/DMRG_energia_D$(D_i)_t$(BLAS.get_num_threads()).txt","a")
    write(io1,"D   Time  var(Time)  \n")
    time=zeros(nsim,ndims)
    i, j, k = Index.((dim, dim, dim))
    for k in 1:nsim
      A = random_itensor(i, j)
      B = random_itensor(j, k)
      t= @elapsed A * B
      time[k,i]=t
    end
  end
  imprimir(time,io1)
  close(io1)
  
  using CUDA # This will trigger the loading of `NDTensorsCUDAExt` in the 
  for i in 1:ndims
    dim=dim_i+(i-1)*pasod
    io1 = open("resultados/DMRG_energia_D$(D_i)_t$(BLAS.get_num_threads()).txt","a")
    write(io1,"D   Time  var(Time)  \n")
    time=zeros(nsim,ndims)
    i, j, k = Index.((dim, dim, dim))
    for k in 1:nsim
      A = random_itensor(i, j)
      B = random_itensor(j, k)
      Acu = cu(A)
      Bcu = cu(B)
      t=@elapsed Acu * Bcu
      time[k,i]=t
    end
  end
  imprimir(time,io1)
  close(io1)  
end