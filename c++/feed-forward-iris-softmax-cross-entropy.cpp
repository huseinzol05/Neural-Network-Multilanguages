#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <random>
#include <cstdio>
#include <fstream>
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xio.hpp"
#include "xtensor-blas/xlinalg.hpp"

using namespace std;

vector<vector<string>> parseCSV(string inputFileName) {
 
    vector<vector<string> > data;
    ifstream inputFile(inputFileName);
    int l = 0;
 
    while (inputFile) {
        l++;
        string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
            istringstream ss(s);
            vector<string> record;
 
            while (ss) {
                string line;
                if (!getline(ss, line, ','))
                    break;
                try {
                    record.push_back(line);
                }
                catch (const std::invalid_argument e) {
                    cout << "NaN found in file " << inputFileName << " line " << l
                         << endl;
                    e.what();
                }
            }
            data.push_back(record);
        }
    }
 
    if (!inputFile.eof()) {
        cerr << "Could not read file " << inputFileName << "\n";
        __throw_invalid_argument("File not found.");
    }
    return data;
}

template <typename T>
int indexOf(string element, T elements){
    // i had to cheat
    for (int i = 0; i < 3;i++) if (elements[i] == element) return i;
    return -1;
}

xt::xarray<double> sigmoid(xt::xarray<double> m, bool grad){
    if(grad) return sigmoid(m,false) * (1 - sigmoid(m,false));
    else return 1 / (1 + xt::exp(-1 * m));
}

int getIndexMax(xt::xarray<double> m){
    int max = 0;
    auto x = m.shape();
    for(int i = 1;i < x[0]; i++){
        if (m(i) > m(max)) max = i;
    }
    return max;
}

double getMax(xt::xarray<double> m){
    double max = m(0);
    auto x = m.shape();
    for(int i = 1;i < x[0]; i++){
        if (m(i) > max) max = m(i);
    }
    return max;
}

double getSum(xt::xarray<double> m){
    double total = m(0);
    auto x = m.shape();
    for(int i = 1;i < x[0]; i++) total += m(i);
    return total;
}

xt::xarray<double> transpose(xt::xarray<double> m){
    auto x = m.shape();
    xt::xarray<double> t = xt::zeros<double> ({x[1],x[0]});
    for(int i = 0; i < x[0]; ++i){
        for(int k = 0; k < x[1]; ++k){
            t(k, i) = m(i,k);
        }
    }
    return t;
}

xt::xarray<double> getMaxColumn(xt::xarray<double> m){
    auto x = m.shape();
    xt::xarray<double> max = xt::zeros<double> ({x[0]});
    for(int i = 0;i < x[0]; i++) max(i) = getMax(xt::view(m, i));
    xt::xarray<double> max_matrix = xt::zeros<double> ({x[0], x[1]});
    for(int i = 0;i < x[1]; i++) xt::view(max_matrix, xt::range(0, x[0]), i) = max;
    return max_matrix;
}

xt::xarray<double> getSumColumn(xt::xarray<double> m){
    auto x = m.shape();
    xt::xarray<double> max = xt::zeros<double> ({x[0]});
    for(int i = 0;i < x[0]; i++) max(i) = getSum(xt::view(m, i));
    xt::xarray<double> max_matrix = xt::zeros<double> ({x[0], x[1]});
    for(int i = 0;i < x[1]; i++) xt::view(max_matrix, xt::range(0, x[0]), i) = max;
    return max_matrix;
}

xt::xarray<double> softmax(xt::xarray<double> m, bool grad){
    if(grad){
        xt::xarray<double> p = softmax(m,false);
        return p * (1-p);
    }
    else {
        xt::xarray<double> max_columns = getMaxColumn(m);
        m = xt::exp(m - max_columns);
        xt::xarray<double> sum_columns = getSumColumn(m);
        return m / sum_columns;
    }
}

xt::xarray<double> cross_entropy(xt::xarray<double> x, xt::xarray<double> y, bool grad){
    if(grad) return -1 * (y / x) + (1 - y) / (1 - x);
    else return -1 * y * xt::log(x) - (1 - y) * xt::log(1 - x);
}

double get_mean(xt::xarray<double> m){
    double values = 0;
    double total = 0;
    auto x = m.shape();
    for(int i = 0;i < x[0]; i++){
        for(int k = 0;k < x[1]; k++){
            values += m(i,k);
            total++;
        }
    }
    return values / total;
}

double get_accuracy(xt::xarray<double> x, xt::xarray<double> y){
    double correct = 0;
    double total = 0;
    auto shapes = x.shape();
    for(int i = 0;i < shapes[0]; i++){
        int current_x = getIndexMax(xt::view(x, i));
        int current_y = getIndexMax(xt::view(y, i));
        //printf("%d %d",current_x, current_y);
        if(current_x == current_y) correct++;
        total++;
    }
    return correct / total;
}

const int EPOCH = 100;
const double LEARNING_RATE = 0.001;

int main() {
    vector<vector<string>> data = parseCSV("iris.csv");
    int row_size = data.size() - 1;
    int column_size = data[0].size() - 2;
    xt::xarray<double> x_iris = xt::zeros<double> ({row_size, column_size});
    vector<string> flowers (row_size);
    
    for(int i = 0;i < row_size; i++) {
        for(int k = 0;k < column_size; k++){
            x_iris(i, k) = stod(data[i+1][k]);
        }
        flowers[i] = data[i+1][data[0].size()-1];
    }
    auto unique_flowers = unique(flowers.begin(), flowers.end());
    
    xt::xarray<double> onehot = xt::zeros<double> ({row_size, 3});
    for(int i = 0;i < row_size; i++) onehot(i, indexOf(flowers[i], unique_flowers)) = 1;
    xt::xarray<double> w1 = xt::random::randn<double>({column_size, 64});
    xt::xarray<double> w2 = xt::random::randn<double>({64, 64});
    xt::xarray<double> w3 = xt::random::randn<double>({64, 3});

    for(int iteration = 0;iteration < EPOCH; iteration++) {
        xt::xarray<double> a1 = xt::linalg::dot(x_iris, w1);
        xt::xarray<double> z1 = sigmoid(a1,false);
        xt::xarray<double> a2 = xt::linalg::dot(z1, w2);
        xt::xarray<double> z2 = sigmoid(a2,false);
        xt::xarray<double> a3 = xt::linalg::dot(z2, w3);
        xt::xarray<double> y_hat = softmax(a3,false);
        double accuracy = get_accuracy(y_hat, onehot);
        xt::xarray<double> err = cross_entropy(y_hat, onehot,false);
        double cost = get_mean(err);
        printf("epoch %d, cost %f, accuracy %f\n",iteration+1,cost,accuracy);
        xt::xarray<double> dy_hat = cross_entropy(y_hat, onehot,true);
        xt::xarray<double> da3 = softmax(a3,true) * dy_hat;
        xt::xarray<double> dw3 = xt::linalg::dot(transpose(z2), da3);
        xt::xarray<double> dz2 = xt::linalg::dot(da3, transpose(w3));
        xt::xarray<double> da2 = sigmoid(a2,true) * dz2;
        xt::xarray<double> dw2 = xt::linalg::dot(transpose(z1), da2);
        xt::xarray<double> dz1 = xt::linalg::dot(da2, transpose(w2));
        xt::xarray<double> da1 = sigmoid(a1,true) * dz1;
        xt::xarray<double> dw1 = xt::linalg::dot(transpose(x_iris), da1);
        w3 = w3 + (-1 * LEARNING_RATE) * dw3;
        w2 = w2 + (-1 * LEARNING_RATE) * dw2;
        w1 = w1 + (-1 * LEARNING_RATE) * dw1;
    }
    return 0;
}