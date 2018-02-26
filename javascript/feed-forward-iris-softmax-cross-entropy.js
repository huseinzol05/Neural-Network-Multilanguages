module.paths.push('/usr/local/lib/node_modules');
var csv = require('csv-array');
var nj = require('numjs');

function sigmoid(X, grad){
    if(grad){
        minus = nj.sigmoid(X).multiply(-1).add(1);
        return nj.sigmoid(X).multiply(minus);
    }
    else{
        return nj.sigmoid(X);
    }
}

function softmax(X, grad){
    if(grad){
        p = softmax(X,false);
        minus = p.multiply(-1).add(1);
        return p.multiply(minus);
    }
    else{
        softmax_results = [];
        for(var i = 0; i < X.shape[0];i++) softmax_results.push(nj.softmax(X.slice([i,i+1], null)).tolist()[0]);
        return nj.array(softmax_results);
    }
}

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return maxIndex;
}

function cross_entropy(X, Y, grad){
    if(grad){
        left = Y.divide(X).multiply(-1);
        middle = Y.multiply(-1).add(1);
        right = X.multiply(-1).add(1);
        right_handside = middle.divide(right);
        return left.add(right_handside);
    }
    else{
        left = Y.multiply(-1).multiply(nj.log(X));
        middle = Y.multiply(-1).add(1);
        right = nj.log(X.multiply(-1).add(1));
        right_handside = middle.multiply(right)
        return left.subtract(right_handside)
    }
}

function accuracy(X, Y){
    correct = 0;
    for(var i = 0; i < X.shape[0]; i++) {
        x = indexOfMax(X.slice([i,i+1], null).tolist()[0]);
        y = indexOfMax(Y.slice([i,i+1], null).tolist()[0]);
        if(x==y)correct++;
    }
    return (correct * 1.0) / X.shape[0];
}

// global variables
var EPOCH = 50;
var LEARNING_RATE = 0.00005;
var BATCH_SIZE = 10;

csv.parseCSV("iris.csv", function(data){
    var X = [], labels = [], Y = [];
    
    for(var i = 1; i < data.length; i++) {
        X.push(data[i].slice(1,-1).map(Number));
        labels.push(data[i].slice(-1)[0]);
    }
    X = nj.array(X);
    flowers = [...new Set(labels)];
    onehot = nj.zeros([labels.length,flowers.length]);
    for(var i = 0; i < data.length; i++) {
        Y.push(flowers.indexOf(labels[i]));
        onehot.set(i,flowers.indexOf(labels[i]),1);
    }
    
    // had to remove last row, numjs's bug
    onehot = onehot.slice([null,-1],null)
    X = X.slice([null,-1],null)
    console.log(X.shape, onehot.shape);
    
    var W1 = nj.random([X.shape[1],64]);
    var W2 = nj.random([64,64]);
    var W3 = nj.random([64,flowers.length]);
    for(var i = 0; i < EPOCH; i++) {
        total_cost = 0, total_acc = 0;
        for(var k = 0; k < Math.floor(X.shape[0] / BATCH_SIZE) * BATCH_SIZE; k += BATCH_SIZE){
            a1 = nj.dot(X, W1);
            z1 = sigmoid(a1, false);
            a2 = nj.dot(z1, W2);
            z2 = sigmoid(a2, false);
            a3 = nj.dot(z2, W3);
            y_hat = softmax(a3, false);
            total_acc += accuracy(y_hat, onehot);
            total_cost += cross_entropy(y_hat,onehot,false).mean();
            dy_hat = cross_entropy(y_hat,onehot, true);
            da3 = softmax(a3, true).multiply(dy_hat);
            dW3 = nj.dot(z2.T, da3);
            dz2 = nj.dot(da3, W3.T);
            da2 = sigmoid(a2, true).multiply(dz2);
            dW2 = nj.dot(z1.T, da2);
            dz1 = nj.dot(da2, W2.T);
            da1 = sigmoid(a1, true).multiply(dz1);
            dW1 = nj.dot(X.T, da1);
            W3 = W3.add(dW3.multiply(LEARNING_RATE * -1));
            W2 = W2.add(dW2.multiply(LEARNING_RATE * -1));
            W1 = W1.add(dW1.multiply(LEARNING_RATE * -1));
        }
        total_acc /= Math.floor(X.shape[0] / BATCH_SIZE);
        total_cost /= Math.floor(X.shape[0] / BATCH_SIZE);
        console.log('epoch: '+(i+1)+', loss: '+total_cost+', accuracy: '+total_acc);
    }
}, false);