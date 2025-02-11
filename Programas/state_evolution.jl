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
  function random_1gate(i)
    #options=["H","Y","X","Z"]
    options=["Phase","Rn"]
    s=rand(1:length(options))
    if s == 1
      p=rand()*2*pi
      gate=op(options[s],i;ϕ=p)
    else
      p=rand()*2*pi
      t=rand()*2*pi
      l=rand()*2*pi
      gate=op(options[s],i,ϕ=p,θ=t,λ=l)
    end
    return gate
  end

  #Create 2-qubit random gate given index i,j
  function random_2gate(i,j)
    options=["CNOT"]
    s=rand(1:length(options))
    n=rand(1:1)
    if n==1
      gate=op(options[s],i,j)
    else
      gate=op(options[s],j,i)
    end
    return gate
  end

  #Create a layer of 1-qubit random gate given the number ob qubits (N) and a array of index i1
  function layer1(N,i1)
    l=[]
    for j in 1:N
      gate=random_1gate(i1[j])
      push!(l,gate)
    end
    return l
  end

  #Create a layer of 2-qubit random gate given the number ob qubits (N) and a array of index i1, m {1,2} is the qubit in which begins to apply the gates  
  function layer2(N,i1,m)
    l=[]
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
    for i in 1:L
      layer=layer1(N,i1)
      push!(circuit,layer)
      i2=i1'
      layer=layer2(N,i2,2-(i%2))
      push!(circuit,layer)
      i1=i2'
    end
    return circuit,i1 
  end

  #Computes the norm of the state A with N qubits
  function norm(A,N)
    suma=0
    for i in 1:2^N
      suma=suma+conj(A[i])*A[i]
    end
    return suma
  end

  #Apply the circuit to the state by applying gates one by one
  function contraction(phi,circuit,N,L,tiempo)
    aux=state(phi,N)
    s=aux
    #s=MPS(aux,inds(aux),cutoff=1E-10)
    for i in 1:L
      t=@elapsed for j in 1:2
        for gate in circuit[2*(i-1)+j] 
            s=s*gate
        end
      end
      tiempo[i]=tiempo[i]+t
    end
    return s
  end

  
  L=30
  N=20
  nsim=5
  io2 = open("$(N)_tiempo_contraccion_$(BLAS.get_num_threads())_$(Threads.nthreads()).txt","w")
  print("Numero de qubits: $N \n")
  print("$(BLAS.get_config())")
  print("Threads (LinearAlgebra) :$(BLAS.get_num_threads())\n")
  print("Threads (Julia) :$(Threads.nthreads())\n\n\n")
  write(io2,"Numero de qubits: $N \n\n\n")
  write(io2,"Threads (LinearAlgebra) :$(BLAS.get_num_threads())\n")
  write(io2,"Threads (Julia) :$(Threads.nthreads())\n\n\n")
  tiempo=zeros(L)

  let 
    i=siteinds("Qubit", N) 
    phi=init_state(N,i)
    circuit,i=random_circuit(i,N,L)
    @time statev = contraction(phi,circuit,N,L,tiempo)
  end

  tiempo=zeros(nsim,L)
  
    for k in 1:nsim
      let
        i=siteinds("Qubit", N) 
        phi=init_state(N,i)
        circuit,i=random_circuit(i,N,L)
        phi=state(phi,N)
        for i in 1:L
          t=@elapsed for j in 1:2
            for gate in circuit[2*(i-1)+j] 
                phi=phi*gate
            end
          end
          tiempo[k,i]=t
        end
        A=Array(phi,i)
        @show norm(A,N)
        print(A[1:5])
        print("\n\n")
      end
    end

    total=0
  for i in 1:L
    write(io2,"$i")
    for k in 1:nsim
      write(io2,"    $(tiempo[k,i])")
      print("    $(tiempo[k,i])")
      total=total+tiempo[k,i]
    end
    write(io2,"\n")
  end

  
  write(io2,"Total= $(total/(nsim-1))")
  print("Total= $(total/(nsim-1))")
  close(io2)

end