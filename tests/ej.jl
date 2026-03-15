using Flux

result = softmax([-1, -1, -0.2])
# println(result)

x = Bool[1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1]
y = Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1]

println(2 < size(x, 2) == size(y, 2))