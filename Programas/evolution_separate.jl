using ITensors
using ITensorMPS
using Random
using .Threads



let
  ITensors.disable_warn_order() 
  function init_state(N,i)
    phi=[]
    for j in 1:N
      j=ITensor(i[j])
      j[1]=1
      push!(phi,j)
    end
    return phi
  end

  function state_div(phi,D)
    state=[]
    cont=0
    aux=0
    for i in phi
      if cont==0
        aux=i
      else
        aux=aux*i
      end
      cont=cont+1
      if cont == D
        push!(state,aux)
        cont=0
      end
    end
    if cont != 0
      push!(state,aux)
    end
    return state
  end

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

  function layer1(N,i1)
    l=[]
    for j in 1:N
      gate=random_1gate(i1[j])
      push!(l,gate)
    end
    return l
  end

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

  function norm(A,N)
    suma=0
    for i in 1:2^N
      suma=suma+conj(A[i])*A[i]
    end
    return suma
  end

  function divide(circuit,N)
    circ=[]
      for layer in circuit
        l=[]
        cont=0
        division=[]
        for gate in layer
          cont=cont+length(inds(gate))/2
          if cont < N
            push!(division,gate)
          end
          if cont == N
            push!(division,gate)
            push!(l,division)
            division=[]
            cont=0
          end
          if cont > N 
            A,B,C=svd(gate,inds(gate)[2],inds(gate)[4])
            push!(division,A*B)
            push!(l,division)
            division=[]
            push!(division,C)
            cont=1
            if N==1
              push!(l,division)
              division=[]
            end
          end
        end
        if cont != 0
          push!(l,division)
        end
        push!(circ,l)
      end
    
    return circ
  end

  function contract_final_state(state)
    l=length(state)


    aux=[]
    if l%2==0
      @threads for i in 1:2:l-1
        @time m=state[i]*state[i+1]
        push!(aux,m)
      end
    else
      @time m=state[1]*state[2]*state[3]
      push!(aux,m)
      @threads for i in 4:2:l-1
        @time m=state[i]*state[i+1]
        push!(aux,m)
      end
    end
    if length(aux)==1
      return aux[1]
    else 
      contract_final_state(aux)
    end
  end

  function contraction(phi,circuit,N,D,L)
    circuitdiv=divide(circuit,D)
    state=state_div(phi,D)
    p=length(circuitdiv[1])
    @time @threads for j in 1:p
      @time for i in 1:2*L
        for gate in circuitdiv[i][j]
          state[j]=state[j]*gate
        end
      end
    end
    @time result = contract_final_state(state)
    return result
  end
  
  io = open("resultados.txt","w")


    for N in [30]
      L=10
      nt=Threads.nthreads()
      print("Threads:$nt \n Numero de qubits: $N \n\n\n")
      write(io,"Threads:$nt \n Numero de qubits: $N \n\n\n")
      i=siteinds("Qubit", N) 
      phi=init_state(N,i)
      @time circuit,i=random_circuit(i,N,L)
      time = []

      for p in [2,2,2,2,2,2]
        let
          D=cld(N,p)
          print("D: $D \n")
          #push!(time, @elapsed state = contraction(phi,circuit,N,D,L))
          @time state = contraction(phi,circuit,N,D,L)
          write(io,"D: $D \n")
          A=Array(state,i)
          @show A[1:5]
          @show norm(A,N)
          print("\n\n")
        end
      end
      #=
      suma=0
      for i in 2:11
        suma=suma+time[i]
      end
      suma=suma/10
      write(io,"Total: $suma \n\n")
      print("Total: $suma \n\n")
      =#
    end
end