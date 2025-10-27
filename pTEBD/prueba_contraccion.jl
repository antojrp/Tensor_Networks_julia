using LinearAlgebra
using ITensors
using ITensorMPS
using Random
using .Threads

let 

    D=2^10
    i=Index(2^9)
    j=Index(2)
    k=Index(D)
    l=Index(2^10)
    A=random_itensor(i,j,k)
    B=random_itensor(k,l)
    println("Dimension indices A: ", dim(inds(A)))
    println("Dimension indices B: ", dim(inds(B)))
    @time C=A*B
    @time C=A*B

end