# Neural-Network-Multilanguages
implement Gradient Descent Feed-forward and Recurrent Neural Network on different languages.

## Support

- [x] Ruby
- [x] Python
- [x] Javascript
- [x] Go
- [ ] C++
- [ ] Java
- [ ] Scala
- [ ] Julia
- [ ] PHP

## Instructions

1. Go to any language folder.
2. run install.sh
3. run the program.

## Neural Network Architectures

1. Feed-forward Neural Network
  * 3 layers included input and output layer
  * first 2 layers squashed into sigmoid function
  * last layer squashed into softmax function
  * loss function is cross-entropy

2. Vanilla Recurrent Neural Network
  * 1 hidden layer
  * tanh as activation function
  * softmax and cross entropy combination for derivative
  * sequence length = 15

## Status

Will update overtime.

## Warning

You would not see high accuracy for other languages that natively are not using float64. During backpropagation, the changes are very small, float32 ignored it.

## Authors

* **Husein Zolkepli** - *Initial work* - [huseinzol05](https://github.com/huseinzol05)
