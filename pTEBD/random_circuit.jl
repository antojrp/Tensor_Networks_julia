module random_circuits

export init_state, random_1gate, random_2gate, layer1, layer2, random_circuit
using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads


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

  #Create 1-qubit random gate given index i
  function random_1gate(i,random,j)
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

  #Create 2-qubit gate given index i,j
  function random_2gate(i,j)
    gate = op("fSim",j,i)
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
  function random_circuit(i1,N,L;reduced=false)
    circuit=[]
    random=rand(0:2,N)
    for i in 1:L
      layer=layer1(N,i1,random)
      push!(circuit,layer)
      if reduced
        i2=i1'
      else
        i2=i1
      end
      layer=layer2(N,i2,2-(i%2))
      push!(circuit,layer)
      if reduced
        i1=i2'
      else  
        i1=i2
      end
    end
    return circuit
  end

end