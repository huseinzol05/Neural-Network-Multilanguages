using DataFrames, CSV

function softmax(X, grad)
    if grad
        p = softmax(X,false)
        return p .* (1-p)
    else
        maxs = findmax(X,2)[1]
        maxs = repeat(maxs, outer=[1,size(X)[2]])
        X = exp(X - maxs)
        sums = sum(X,2)
        sums = repeat(sums, outer=[1,size(X)[2]])
        return X ./ sums
    end
end

function sigmoid(X, grad)
    if grad
        return sigmoid(X,false) .* (1-sigmoid(X,false))
    else
        return 1 ./ (1 - exp(-X))
    end
end

function cross_entropy(X, Y, grad)
    if grad
        X = clamp(X, 1e-15, 1 - 1e-15)
        return -(Y ./ X) + (1 - Y) ./ (1 - X)
    else
        X = clamp(X, 1e-15, 1 - 1e-15)
        return -Y .* log(X) - (1 - Y) .* log(1 - X)
    end
end

iris = readtable("iris.csv",separator=',')
X = convert(Array, iris[:,2:size(iris)[2]-1])
Y = convert(Array, iris[:,size(iris)[2]])
unique_flowers = unique(Y)

flowers = Int64[]
for i in 1:size(X)[1]
    push!(flowers,find(unique_flowers .== Y[i])[1])
end

onehot = zeros((size(X)[1],size(unique_flowers)[1]))
for i in 1:size(X)[1]
    onehot[i, flowers[i]] = 1
end

EPOCH = 100
LEARNING_RATE = 0.0005
W1 = randn((size(X)[2],64))
W2 = randn((64,64))
W3 = randn((64,size(unique_flowers)[1]))

for i in 1:EPOCH
    a1 = X * W1
    z1 = sigmoid(a1,false)
    a2 = z1 * W2
    z2 = sigmoid(a2,false)
    a3 = z2 * W3
    y_hat = softmax(a3,false)
    accuracy = mean(findmax(y_hat,2)[2] .== findmax(onehot,2)[2])
    cost = mean(cross_entropy(y_hat,onehot,false))
    dy_hat = cross_entropy(y_hat,onehot,true)
    da3 = softmax(a3,true) .* dy_hat
    dW3 = transpose(z2) * da3
    dz2 = da3 * transpose(W3)
    da2 = sigmoid(a2, true) .* dz2
    dW2 = transpose(z1) * da2
    dz1 = da2 * transpose(W2)
    da1 = sigmoid(a1, true) .* dz1
    dW1 = transpose(X) * da1
    W3 += -LEARNING_RATE * dW3
    W2 += -LEARNING_RATE * dW2
    W1 += -LEARNING_RATE * dW1
    @printf("epoch %d, cost %f, accuracy %f\n",i, cost, accuracy)
end
