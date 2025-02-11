using LinearAlgebra
using ITensors
using ITensorMPS
using .Threads

Num=41
m=5
i=siteinds("Qubit",2)
phi=random_mps(ComplexF64,i) 
io = open("Tamaño_MPS_Dmax_6.txt","w")
write(io,"N    Tamaño(bytes)\n") 
close(io) 

for p in 16:16
    io = open("Tamaño_MPS_Dmax_6.txt","a")
    max=2^p
    print("D:$max \n")
    write(io,"D:$max \n")
    close(io) 
    let
        for k in 1:m
            io = open("Tamaño_MPS_Dmax_6.txt","a")
            N=Num+k-1
            print("$N   ")
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
            @time phi=MPS(ComplexF64,i;linkdims=link)  
            for q in phi
                q[1]=1
            end  
            write(io,"$N    $(Base.summarysize(phi))\n") 
            close(io)
        end
    end
end
