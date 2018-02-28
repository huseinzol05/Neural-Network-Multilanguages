# Neural-Network-Multilanguages
implement Gradient Descent Feed-forward and Recurrent Neural Network on different languages, only use vector / linear algebra library.

## Support

#### Ruby
  - [x] feed-forward iris
  - [ ] recurrent generator
  - [ ] recurrent forecasting

#### Python
  - [x] feed-forward iris
  - [x] recurrent generator
  - [ ] recurrent forecasting

#### Javascript
  - [x] feed-forward iris
  - [ ] recurrent generator
  - [ ] recurrent forecasting

#### Go
  - [x] feed-forward iris
  - [ ] recurrent generator
  - [ ] recurrent forecasting

#### C++
  - [x] feed-forward iris
  - [ ] recurrent generator
  - [ ] recurrent forecasting

#### Julia
  - [x] feed-forward iris
  - [ ] recurrent generator
  - [ ] recurrent forecasting

#### PHP
  - [x] feed-forward iris
  - [ ] recurrent generator
  - [x] recurrent forecasting

## Instructions

1. Go to any language folder.
2. run install.sh
3. run the program.

## Neural Network Architectures

1. Feed-forward Neural Network to predict Iris dataset.
  * 3 layers included input and output layer
  * first 2 layers squashed into sigmoid function
  * last layer squashed into softmax function
  * loss function is cross-entropy

2. Vanilla Recurrent Neural Network to generate text.
  * 1 hidden layer
  * tanh as activation function
  * softmax and cross entropy combination for derivative
  * sequence length = 15

3. Vanilla Recurrent Neural Network to predict TESLA market.
  * 1 hidden layer
  * tanh as activation function
  * mean square error for derivative
  * sequence length = 5

All implemention like max(), mean(), softmax(), cross_entropy(), sigmoid() are hand-coded, no other libraries.

## Status

Will update overtime.

## Warning

You would not see high accuracy for other languages that natively are not using float64. During backpropagation, the changes are very small, float32 ignored it.

## Authors

* **Husein Zolkepli** - *Initial work* - [huseinzol05](https://github.com/huseinzol05)
