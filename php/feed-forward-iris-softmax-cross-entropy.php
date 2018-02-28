<?php
error_reporting(0);
require __DIR__ . '/vendor/autoload.php';

use MathPHP\LinearAlgebra\Vector;
use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;

function generateZero($length){
    $array = array();
    for($i = 0; $i < $length; $i++) array_push($array, 0);
    return $array;
}

function randomMatrix($x, $y){
    $array = array();
    for($i = 0; $i < $x; $i++){
        $inner = array();
        for($k = 0; $k < $y; $k++){
            array_push($inner, mt_rand() / mt_getrandmax());
        }
        array_push($array,$inner);
    }
    return MatrixFactory::create($array);
}

function getMax($array){
    $max = $array[0];
    for($i = 1; $i < sizeof($array); $i++){
        if($array[$i] > $max) $max = $array[$i];
    }
    return $max;
}

function getSum($array){
    $sum = $array[0];
    for($i = 1; $i < sizeof($array); $i++) $sum += $array[$i];
    return $sum;
}

function getIndexMax($array){
    $max = 0;
    for($i = 1; $i < sizeof($array); $i++){
        if($array[$i] > $array[$max]) $max = $i;
    }
    return $max;
}

function getAccuracy($x, $y){
    $x = $x->getMatrix();
    $y = $y->getMatrix();
    $correct = 0;
    for($i = 0; $i < sizeof($x); $i++){
        $index_x = getIndexMax($x[$i]);
        $index_y = getIndexMax($y[$i]);
        if($index_x == $index_y) $correct++;
    }
    return $correct / sizeof($x);
}

function getMean($array){
    $sum = 0;
    for($i = 0; $i < $array->getM(); $i++) for($k = 0; $k < $array->getN(); $k++) $sum += $array[$i][$k];
    return $sum / ($array->getM() * $array->getN());
}

function getMaxColumn($m){
    $array = $m->getMatrix();
    $rows = $m->getM();
    $columns = $m->getN();
    $maxs = array();
    for($i = 0; $i < $rows; $i++) array_push($maxs, getMax($array[$i]));
    $array_max = array();
    for($i = 0; $i < $columns; $i++) array_push($array_max, $maxs);
    return MatrixFactory::create($array_max)->transpose();
}

function getSumColumn($m){
    $array = $m->getMatrix();
    $rows = $m->getM();
    $columns = $m->getN();
    $sums = array();
    for($i = 0; $i < $rows; $i++) array_push($sums, getSum($array[$i]));
    $array_sum = array();
    for($i = 0; $i < $columns; $i++) array_push($array_sum, $sums);
    return MatrixFactory::create($array_sum)->transpose();
}

function apply_function($m, $func){
    $m = $m->getMatrix();
    for($i = 0; $i < sizeof($m); $i++) for($k = 0; $k < sizeof($m[0]); $k++) $m[$i][$k] = $func($m[$i][$k]);
    return MatrixFactory::create($m);
}

function divide($a, $b){
    $a = $a->getMatrix();
    for($i = 0; $i < sizeof($a); $i++) for($k = 0; $k < sizeof($a[0]); $k++) $a[$i][$k] /= $b[$i][$k];
    return MatrixFactory::create($a);
}

function softmax($m, $grad){
    if($grad){
        $p = softmax($m,false);
        $ones = MatrixFactory::one($p->getM(), $p->getN());
        $ones = $ones->subtract($p);
        return $p->hadamardProduct($ones);
    }
    else{
        $max_columns = getMaxColumn($m);
        $m = $m->subtract($max_columns);
        $m = apply_function($m, exp);
        $sum_columns = getSumColumn($m);
        return divide($m, $sum_columns);
    }
}

function clipping($m, $min, $max){
    $m = $m->getMatrix();
    for($i = 0; $i < sizeof($m); $i++){
        for($k = 0; $k < sizeof($m[0]); $k++){
            if($m[$i][$k] < $min) $m[$i][$k] = $min;
            if($m[$i][$k] > $max) $m[$i][$k] = $max;
        }
    }
    return MatrixFactory::create($m);
}

function cross_entropy($x, $y, $grad){
    if($grad){
        $x = clipping($x, 1e-15, 1-1e-15);
        $x = $x->getMatrix();
        for($i = 0; $i < sizeof($x); $i++){
            for($k = 0; $k < sizeof($x[0]); $k++){
                $x[$i][$k] = -1 * ($y[$i][$k] / ($x[$i][$k]+1e-15)) + (1 - $y[$i][$k]) / (1 - $x[$i][$k]+1e-15);
            }
        }
        return MatrixFactory::create($x);
    }
    else{
        $x = clipping($x, 1e-15, 1-1e-15);
        $x = $x->getMatrix();
        for($i = 0; $i < sizeof($x); $i++){
            for($k = 0; $k < sizeof($x[0]); $k++){
                $x[$i][$k] = -1 * $y[$i][$k] * log($y[$i][$k]+1e-15) - (1 - $y[$i][$k]) * log(1 - $x[$i][$k]+1e-15);
            }
        }
        return MatrixFactory::create($x);
    }
}

function sigmoid($m, $grad){
    if($grad){
        $ones = MatrixFactory::one($m->getM(), $m->getN());
        $ones = $ones->subtract(sigmoid($m,false));
        return sigmoid($m,false)->hadamardProduct($ones);
    }
    else{
        $m = $m->getMatrix();
        for($i = 0; $i < sizeof($m); $i++) for($k = 0; $k < sizeof($m[0]); $k++) $m[$i][$k] = 1 / (1 + exp(-1 * $m[$i][$k]));
        return MatrixFactory::create($m);
    }
}

function derivative_cross_entropy_softmax($p, $y){
    $p = $p->getMatrix();
    for($i = 0; $i < sizeof($p); $i++){
        $p[$i][$y[$i]] -= 1;
    }
    return MatrixFactory::create($p);
}

$EPOCH = 100;
$LEARNING_RATE = 0.00005;
$iris = array();
$flowers = array();
$file = fopen("iris.csv","r");
$i = 0;

while(!feof($file)){
    $temp = fgetcsv($file);
    if($i > 0) {
        array_push($iris, array_slice($temp,1,-1));
        array_push($flowers, $temp[sizeof($temp)-1]);
    }
    $i++;
}
fclose($file);
$iris = array_slice($iris,0,-1);
$flowers = array_slice($flowers,0,-1);
$x_iris = MatrixFactory::create($iris);
$unique_flowers = array_values(array_unique($flowers));

$onehot = array();
$y_iris = array();
for($i = 0; $i < sizeof($iris); $i++) {
    $zeros = generateZero(sizeof($unique_flowers));
    $zeros[array_search($flowers[$i], $unique_flowers)] = 1;
    array_push($onehot, $zeros);
    array_push($y_iris, array_search($flowers[$i], $unique_flowers));
}
$onehot = MatrixFactory::create($onehot);
$W1 = randomMatrix($x_iris->getN(),64);
$W2 = randomMatrix(64,64);
$W3 = randomMatrix(64,sizeof($unique_flowers));

for($iteration = 0; $iteration < $EPOCH; $iteration++) {
    $a1 = $x_iris->multiply($W1);
    $z1 = sigmoid($a1,false);
    $a2 = $z1->multiply($W2);
    $z2 = sigmoid($a2,false);
    $a3 = $z2->multiply($W3);
    $y_hat = softmax($a3,false);
    $accuracy = getAccuracy($y_hat, $onehot);
    $cost = getMean(cross_entropy($y_hat,$onehot,false));
    $dy_hat = cross_entropy($y_hat,$onehot,true);
    $da3 = softmax($a3, true)->hadamardProduct($dy_hat);
    $dW3 = $z2->transpose()->multiply($da3);
    $dz2 = $da3->multiply($W3->transpose());
    $da2 = sigmoid($a2, true)->hadamardProduct($dz2);
    $dW2 = $z1->transpose()->multiply($da2);
    $dz1 = $da2->multiply($W2->transpose());
    $da1 = sigmoid($a1, true)->hadamardProduct($dz1);
    $dW1 = $x_iris->transpose()->multiply($da1);
    $W3 = $W3->add($dW3->scalarMultiply(-1*$LEARNING_RATE));
    $W2 = $W2->add($dW2->scalarMultiply(-1*$LEARNING_RATE));
    $W1 = $W1->add($dW1->scalarMultiply(-1*$LEARNING_RATE));
    printf("epoch %u, cost %f, accuracy %f\n",$iteration+1,$cost,$accuracy);
}
?>