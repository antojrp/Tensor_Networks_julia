using ITensors, Random

let
Random.seed!(1234)

# Creamos un Gamma3 denso, complejo
D1 = 2
D2 = 1024
D3 = 512
i = Index(D1, "Qubit,Site,n=3")
j = Index(D2, "Link,u")
k = Index(D2, "Link,v")
l = Index(D2, "Link,v")

A = random_itensor(i, j, k)
B = diag_itensor(rand(D2), k, l)
C = A * B
@time C = A * B

A = random_itensor(k, i, j)
B = diag_itensor(rand(D2), k, l)
@time C = A * B

A=random_itensor(j, k, i)
B=diag_itensor(rand(D2), k, l)
@time C = A * B

permuted_A = permute(A, i, j, k)
A=random_itensor(j, k, i)

@time permuted_A = permute(A, i, j, k)
@time C=permuted_A*B

end