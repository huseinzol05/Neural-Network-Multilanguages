#!/bin/bash

# install g++ 5
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-5 g++-5

git clone https://github.com/QuantStack/xtl.git
cd xtl
mkdir build && cd build
cmake ..
make
make install
cd ../..

git clone https://github.com/QuantStack/xtensor.git
cd xtensor
mkdir build && cd build
cmake ..
make
make install
cd ../..

git clone https://github.com/QuantStack/xtensor-blas.git
cd xtensor-blas
mkdir build && cd build
cmake ..
make
make install
cd ../..

