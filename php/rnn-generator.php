<?php
error_reporting(0);
require __DIR__ . '/vendor/autoload.php';
use MathPHP\LinearAlgebra\Vector;
use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;

function constant2D($x, $y, $val){
    $array = array();
    for($i = 0; $i < $x; $i++) {
        $inner = array();
        for($k = 0; $k < $y; $k++) {
            array_push($inner, $val);
        }
        array_push($array, $inner);
    }
    return MatrixFactory::create($array);
}

function divide($a, $b){
    $a = $a->getMatrix();
    for($i = 0; $i < sizeof($a); $i++) for($k = 0; $k < sizeof($a[0]); $k++) $a[$i][$k] /= $b[$i][$k];
    return MatrixFactory::create($a);
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

function constants($x, $y, $val){
    $array = array();
    for($i = 0; $i < $x; $i++){
        $inner = array();
        for($k = 0; $k < $y; $k++){
            array_push($inner, $val);
        }
        array_push($array,$inner);
    }
    return MatrixFactory::create($array);
}

function normalized($x){
    $m = $x->getMatrix();
    $max = max($m[0]);
    $min = min($m[0]);
    $maxs = constants(sizeof($m),sizeof($m[0]),$max);
    $mins = constants(sizeof($m),sizeof($m[0]),$min);
    $top = $x->subtract($mins);
    $bottom = $maxs->subtract($mins);
    return array(divide($top, $bottom), array($min, $max));
}

function unnormalized($x, $min, $max){
    $maxs = constants($x->getM(),$max);
    $mins = constants($x->getM(),$min);
    $left = $x->hadamardProduct($maxs->subtract($mins));
    return $left->add($mins);
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
function our_tanh($m, $grad){
    if($grad){
        $m = $m->getMatrix();
        for($i = 0; $i < sizeof($m); $i++) for($k = 0; $k < sizeof($m[0]); $k++) $m[$i][$k] = 1 - pow(tanh($m[$i][$k]),2);
        return MatrixFactory::create($m);
    }
    else{
        $m = $m->getMatrix();
        for($i = 0; $i < sizeof($m); $i++) for($k = 0; $k < sizeof($m[0]); $k++) $m[$i][$k] = tanh($m[$i][$k]);
        return MatrixFactory::create($m);
    }
}

function forward_multiply_gate($w, $x){
    return $w->multiply($x);
}

function backward_multiply_gate($w, $x, $dz){
    $dW = $dz->transpose()->multiply($x);
    $dx = $w->transpose()->multiply($dz->transpose());
    return array($dW, $dx);
}

function forward_add_gate($x1, $x2){
    return $x1->add($x2);
}

function backward_add_gate($x1, $x2, $dz){
    $ones_x1 = MatrixFactory::one($x1->getM(),$x1->getN());
    $ones_x2 = MatrixFactory::one($x2->getM(),$x2->getN());
    $dx1 = $dz->hadamardProduct($ones_x1);
    $dx2 = $dz->hadamardProduct($ones_x2);
    return array($dx1, $dx2);
}

function forward_recurrent($x, $prev_state, $U, $W, $V){
    $mul_u = forward_multiply_gate($x, $U->transpose());
    $mul_w = forward_multiply_gate($prev_state, $W->transpose());
    $add_previous_now = forward_add_gate($mul_u, $mul_w);
    $current_state = our_tanh($add_previous_now);
    $mul_v = forward_multiply_gate($current_state, $V->transpose());
    return array($mul_u, $mul_w, $add_previous_now, $current_state, $mul_v);
}

function backward_recurrent($x, $prev_state, $U, $W, $V, $d_mul_v, $saved_graph){
    $mul_u = $saved_graph[0];
    $mul_w = $saved_graph[1];
    $add_previous_now = $saved_graph[2];
    $current_state = $saved_graph[3];
    $mul_v = $saved_graph[4];
    
    $backward_multiply_V = backward_multiply_gate($V, $current_state, $d_mul_v);
    $dV = $backward_multiply_V[0];
    $dcurrent_state = $backward_multiply_V[1];
    $dadd_previous_now = our_tanh($add_previous_now, true)->hadamardProduct($dcurrent_state->transpose());
    $backward_add = backward_add_gate($mul_w, $mul_u, $dadd_previous_now);
    $dmul_w = $backward_add[0];
    $dmul_u = $backward_add[1];
    $backward_multiply_W = backward_multiply_gate($W, $prev_state, $dmul_w);
    $dW = $backward_multiply_W[0];
    $dprev_state = $backward_multiply_W[1];
    $backward_multiply_U = backward_multiply_gate($U, $x, $dmul_u);
    $dU = $backward_multiply_U[0];
    $dx = $backward_multiply_U[1];
    return array($dprev_state, $dU, $dW, $dV);
}

?>
