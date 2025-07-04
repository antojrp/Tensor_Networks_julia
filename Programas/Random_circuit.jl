using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads

let
  ITensors.disable_warn_order()

  #Initialize state to |0>
  function init_state(N,i)
    phi=[]
    for j in 1:N
      j=ITensor(i[j])
      j[1]=1
      push!(phi,j)
    end
    return phi
  end

  #Contract state
  function state(phi,N)
    aux=phi[1]
    for i in 2:N
      aux=aux*phi[i]
    end
    return aux
  end

  #Create 1-qubit random gate given index i
  function random_1gate(i,random,j)
    #=
    options=["Phase","Rn"]
    s=rand(1:length(options))
    if s == 1
      p=rand()*2*pi      
      gate=op(options[s],i;ϕ=p)
    else
      p=rand()*2*pi
      t=rand()*2*pi
      l=rand()*2*pi
      gate=op(options[2],i,ϕ=p,θ=t,λ=l)
    end
    =#
    s=rand(1:2)
    s=(s+random[j])%3
    random[j]=s
    if s == 0
      gate=op("Rx", i;θ=π/2)
    
    elseif s == 1
      gate=op("Ry", i;θ=π/2)

    else
      gate=op("W", i)

    end

    return gate
  end


  ITensors.op(::OpName"fSim",::SiteType"Qubit")= [
      1  0                 0                 0
      0  cos(π/2)           -im*sin(π/2)        0
      0  -im*sin(π/2)       cos(π/2)            0
      0  0                 0                 exp(-im*π/6)
  ]

  ITensors.op(::OpName"W",::SiteType"Qubit")= (1/sqrt(2)) * [1 -sqrt(im); sqrt(-im) 1]

  

  #Create 2-qubit random gate given index i,j
  function random_2gate(i,j)
    #=
    options=["CNOT"]
    s=rand(1:length(options))
    n=rand(1:1)
    if n==1
      gate=op(options[s],i,j)
    else
      gate=op(options[s],j,i)
    end
    =#
    gate = op("fSim",i,j)
    return gate
  end

  #Create a layer of 1-qubit random gate given the number ob qubits (N) and a array of index i1
  function layer1(N,i1,random)
    l=ITensor[]
    for j in 1:N
      gate=random_1gate(i1[j],random,j)
      push!(l,gate)
    end
    return l
  end

  #Create a layer of 2-qubit random gate given the number ob qubits (N) and a array of index i1, m {1,2} is the qubit in which begins to apply the gates
  function layer2(N,i1,m)
    l=ITensor[]
    if m==2
      gate=delta(i1[1],i1[1]')
      push!(l,gate)
    end
    for j in m:2:N-1
      gate=random_2gate(i1[j],i1[j+1])
      push!(l,gate)
    end
    if m==2
      if N%2==0
        gate=delta(i1[N],i1[N]')
        push!(l,gate)
      end
    end
    if m==1
      if N%2==1
        gate=delta(i1[N],i1[N]')
        push!(l,gate)
      end
    end
    return l
  end

  #Creates a random circuit of L layers, each layer is composed by a 1-qubit layer and 2-qubit layer.
  function random_circuit(i1,N,L)
    circuit=[]
    random=rand(0:2,N)
    for i in 1:L
      layer=layer1(N,i1,random)
      push!(circuit,layer)
      #i2=i1'
      i2=i1
      layer=layer2(N,i2,2-(i%2))
      push!(circuit,layer)
      #i1=i2'
      i1=i2
    end
    return circuit,i1 
  end

  #Compute the mean between all the Schmidt ranks
  function schmidt(state,N)
    # suma=0
    max=0
    for i in 1:N-1
      let
        #print("Separación:$(i) Dimension: ")
        #print(inds(state[i])[end])
        #print(dim(inds(state[i])[end]))
        #print("\n")
        #suma=suma+dim(inds(state[i])[end])
        if max < dim(inds(state[i])[end])
          max=dim(inds(state[i])[end])
        end
      end
    end
    # return suma/(N-1)
    return max
  end

  function entropy(psi,b)
    psi = orthogonalize(psi, b)
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    SvN = 0.0
    for n=1:dim(S, 1)
      p = S[n,n]^2
      #SvN -= p * log(p)
      SvN += p^2
    end
    return -log2(SvN)
  end

  function eigenvalues(psi,b)
    io3 = open("resultados/Random_coeficientes_$(L)_t$(BLAS.get_num_threads()).txt","a")
    psi = orthogonalize(psi, b)
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    for n=1:dim(S, 1)
      p = S[n,n]^2
      write(io3,"$(p)\n")
    end
    write(io3,"\n")
    close(io3)
  end

  #Number of Layers
  L=15
  #Number of qubits starting
  Ni=15
  m=6
  #Number of simulations
  nsim=21

  mkpath("resultados")

  io1 = open("resultados/Random_entrelazamiento_$(L)_t$(BLAS.get_num_threads()).txt","w")
  io2 = open("resultados/Random_tiempo_$(L)_t$(BLAS.get_num_threads()).txt","w")
  io3 = open("resultados/Random_coeficientes_$(L)_t$(BLAS.get_num_threads()).txt","w")
  close(io1)
  close(io2)
  close(io3)

  for num in 1:m
    N=Ni+num-1
    #Print 
    io1 = open("resultados/Random_entrelazamiento_$(L)_t$(BLAS.get_num_threads()).txt","a")
    io2 = open("resultados/Random_tiempo_$(L)_t$(BLAS.get_num_threads()).txt","a")
    io3 = open("resultados/Random_coeficientes_$(L)_t$(BLAS.get_num_threads()).txt","a")
    print("Numero de qubits: $N \n")
    print("Threads (LinearAlgebra) :$(BLAS.get_num_threads())\n")
    print("Threads (Julia) :$(Threads.nthreads())\n\n\n")
    write(io1,"Numero de qubits: $N \n")
    write(io2,"Numero de qubits: $N \n")
    write(io3,"Numero de qubits: $N \n")
    close(io3)
    write(io1,"Layer   D  var(D)  Renyi  var(Renyi)\n")
    write(io2,"Layer   Time var(Time)\n")


    #Initialize bond and time arrays to 0
    max_bond=zeros(nsim,L)
    tiempo=zeros(nsim,L)
    renyi=zeros(nsim,L)

    for k in 1:nsim
      let
        #Creates index array
        i=siteinds("Qubit", N) 

        #Initialitates the state
        psi=MPS(i,"0")

        #Creates a random_circuit
        circuit,i=random_circuit(i,N,L)


        #Apply the circuit to the state
        @time for i in 1:L
          t=@elapsed for j in 1:2
            psi=apply(circuit[2*(i-1)+j],psi;cutoff=1E-10)
          end
          tiempo[k,i]=t
          max_bond[k,i]=schmidt(psi,N)
          renyi[k,i]=entropy(psi,div(N,2))
        end
        print("\n\n")
        if true
          eigenvalues(psi,div(N,2))
        end
      end
    end
    



    #Print results
    tiempol=zeros(L)
    max_bondl=zeros(L)
    renyil=zeros(L)
    total=zeros(nsim)
    media=0
    for i in 1:L
      write(io1,"$i")
      write(io2,"$i")
      for k in 2:nsim
        tiempol[i]=tiempol[i]+tiempo[k,i]
        max_bondl[i]=max_bondl[i]+max_bond[k,i]
        renyil[i]=renyil[i]+renyi[k,i]

        total[k]=total[k]+tiempo[k,i]
        media=media+tiempo[k,i]
      end
      varianzat=0
      varianzab=0
      varianzar=0
      tiempol[i]=tiempol[i]/(nsim-1)
      max_bondl[i]=max_bondl[i]/(nsim-1)
      renyil[i]=renyil[i]/(nsim-1)
      for k in 2:nsim
        varianzat=varianzat+(tiempo[k,i]-tiempol[i])^2
        varianzab=varianzab+(max_bond[k,i]-max_bondl[i])^2
        varianzar=varianzar+(renyi[k,i]-renyil[i])^2
      end
      varianzat=(varianzat/(nsim-1))^0.5
      varianzab=(varianzab/(nsim-1))^0.5
      varianzar=(varianzar/(nsim-1))^0.5
      write(io1,"  $(max_bondl[i])  $varianzab  $(renyil[i])  $varianzar")
      write(io2,"  $(tiempol[i])  $varianzat")
      write(io1,"\n")
      write(io2,"\n")
    end
    media=media/(nsim-1)
    varianza=0
    for i in 2:nsim
      varianza=varianza+(media-total[i])^2
    end

    varianza=(varianza/(nsim-2))^0.5
    write(io2,"Total= $media\n")
    write(io2,"Varianza= $varianza\n\n")

    close(io1)
    close(io2)

  end

end

