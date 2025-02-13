using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads

let
    io = open("resultados/Expected_t$(BLAS.get_num_threads()).txt","w")
    print("Qubit    tiempo\n")
    close(io)
    nsim=5
    Num=30
    m=1

    i=siteinds("Qubit",4)
    phi=random_mps(ComplexF64,i;linkdims=2)  
    s=div(4,2)
    expect(phi,"Sz";sites=s)
    for p in 1:12
        max=2^p
        io = open("resultados/Expected_t$(BLAS.get_num_threads()).txt","a")
        write(io,"D:$max   \n")
        close(io)
        for k in 1:m
            io = open("resultados/Expected_t$(BLAS.get_num_threads()).txt","a")
            t=zeros(nsim)
            N=Num+k-1
            write(io,"Qubits:$(Num+k-1)\n")
            i=siteinds("Qubit",N)
            link=zeros(Int,N-1)
            for j in 1:div(N-1,2)+(1-N%2)
                a=2^j
                if a > max
                    link[j]=max
                    link[N-j]=max 
                else
                    link[j]=a
                    link[N-j]=a   
                end
                
            end

            for j in 1:nsim
                phi=random_mps(ComplexF64,i;linkdims=link)  
                s=div(N,2)
                t[j]=@elapsed expect(phi,"Sz";sites=s)
            end
            media=0
            for k in 1:nsim
                media=media+t[k]
            end
            media=media/nsim
            varianza=0
            for i in 1:nsim
                varianza=varianza+(media-t[i])^2
            end
            varianza=(varianza/(nsim-1))^0.5
            write(io,"Total= $media\n")
            write(io,"Varianza= $varianza\n\n")
            close(io)
        end
    end
    close(io)
end