#!/usr/bin/ruby -w

require "numo/narray"
require 'csv'

def sigmoid(x, grad)
    if grad
        return sigmoid(x, false) * (1 - sigmoid(x, false))
    else
        return 1 / (1 + Numo::NMath.exp((x * -1)))
    end
end

def softmax(x, grad)
    if grad
        p = softmax(x, false)
        return p * (1-p)
    else
        e_x = Numo::NMath.exp(x - x.max(axis:1,keepdims:true))
        return e_x / e_x.sum(axis:1,keepdims:true)
    end
end

def cross_entropy(x, y, grad)
    if grad
        x = x.clip(1e-15, 1-1e-15)
        return -(y / x) + (1 - y) / (1 - x)
    else
        x = x.clip(1e-15, 1-1e-15)
        return -y * Numo::NMath.log(x) - (1 - y) * Numo::NMath.log(1 - x)
    end
end

def accuracy(x, y)
    correct = 0
    $i = 0
    while $i < x.shape[0]
        if x[$i] == y[$i]
            correct += 1
        end
        $i +=1
    end
    return (correct * 1.0) / x.shape[0]
end

# read our csv
csv = CSV.read('iris.csv', :headers=>true)
sepal_length = csv['SepalLengthCm'].map(&:to_f)
sepal_width = csv['SepalWidthCm'].map(&:to_f)
petal_length = csv['PetalLengthCm'].map(&:to_f)
petal_width = csv['PetalWidthCm'].map(&:to_f)
flowers = csv['Species']
unique_flowers = flowers.uniq

# change from ruby array to numo array
iris = Numo::NArray.[](sepal_length, sepal_width, petal_length, petal_width).transpose

# our Y and onehot-encoder
flowers_int = Array.new(flowers.length)
onehot = Numo::DFloat.zeros(flowers.length, unique_flowers.length)
$i = 0
while $i < flowers.length do
    flowers_int[$i] = unique_flowers.find_index(flowers[$i])
    onehot[$i, unique_flowers.find_index(flowers[$i])] = 1
    $i +=1
end

# our global variables
size_layer = 64
learning_rate = 1e-4
epoch = 50

w1 = Numo::DFloat.new(iris.shape[1], size_layer).rand
b1 = Numo::DFloat.new(size_layer).rand
w2 = Numo::DFloat.new(size_layer, size_layer).rand
b2 = Numo::DFloat.new(size_layer).rand
w3 = Numo::DFloat.new(size_layer, unique_flowers.length).rand
b3 = Numo::DFloat.new(unique_flowers.length).rand

iteration = 0
while iteration < epoch
    a1 = iris.dot(w1) + b1
    z1 = sigmoid(a1,false)
    a2 = z1.dot(w2) + b2
    z2 = sigmoid(a2,false)
    a3 = z2.dot(w3) + b3
    y_hat = softmax(a3,false)
    acc = accuracy(y_hat.max_index(1), onehot.max_index(1))
    cost = cross_entropy(y_hat,onehot, false).mean()
    dy_hat = cross_entropy(y_hat,onehot, true)
    da3 = softmax(a3, true) * dy_hat
    dw3 = z2.transpose.dot(da3)
    db3 = da3.sum(0)
    dz2 = da3.dot(w3.transpose)
    da2 = sigmoid(a2, true) * dz2
    dw2 = z1.transpose.dot(da2)
    db2 = da2.sum(0)
    dz1 = da2.dot(w2.transpose)
    da1 = sigmoid(a1, true) * dz1
    dw1 = iris.transpose.dot(da1)
    db1 = da1.sum(0)
    w3 += -learning_rate * dw3
    b3 += -learning_rate * db3
    w2 += -learning_rate * dw2
    b2 += -learning_rate * db2
    w1 += -learning_rate * dw1
    b1 += -learning_rate * db1
    p "epoch #{iteration+1}, loss #{cost}, accuracy #{acc}"
    iteration += 1
end
