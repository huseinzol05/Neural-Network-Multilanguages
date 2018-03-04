# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
sns.set()

def get_vocab(file, lower = False):
    with open(file, 'r') as fopen:
        data = fopen.read()
    if lower:
        data = data.lower()
    vocab = list(set(data))
    return data, vocab

def embed_to_onehot(data, vocab):
    onehot = np.zeros((len(data), len(vocab)), dtype = np.float32)
    for i in range(len(data)):
        onehot[i, vocab.index(data[i])] = 1.0
    return onehot

text, text_vocab = get_vocab('consumer.h', lower = False)
onehot = embed_to_onehot(text, text_vocab)

learning_rate = 0.0001
batch_size = 64
sequence_length = 12
epoch = 1000
num_layers = 2
size_layer = 128
possible_batch_id = range(len(text) - sequence_length - 1)
dimension = onehot.shape[1]

U = np.random.randn(size_layer, dimension) / np.sqrt(size_layer)
W = np.random.randn(size_layer, size_layer) / np.sqrt(size_layer)
V = np.random.randn(dimension, size_layer) / np.sqrt(dimension)

def tanh(x, grad=False):
    if grad:
        output = np.tanh(x)
        return (1.0 - np.square(output))
    else:
        return np.tanh(x)

def mean_square_error(x, y, grad=False):
    if grad:
        np.mean(-2*(y-x))
    else:
        return np.mean(np.square(y-x))

def forward_multiply_gate(w, x):
    return np.dot(w, x)

def backward_multiply_gate(w, x, dz):
    dW = np.dot(dz.T, x)
    dx = np.dot(w.T, dz.T)
    return dW, dx

def forward_add_gate(x1, x2):
    return x1 + x2

def backward_add_gate(x1, x2, dz):
    dx1 = dz * np.ones_like(x1)
    dx2 = dz * np.ones_like(x2)
    return dx1, dx2

def forward_recurrent(x, prev_state, U, W, V):
    mul_u = forward_multiply_gate(x, U.T)
    mul_w = forward_multiply_gate(prev_state, W.T)
    add_previous_now = forward_add_gate(mul_u, mul_w)
    current_state = tanh(add_previous_now)
    mul_v = forward_multiply_gate(current_state, V.T)
    return (mul_u, mul_w, add_previous_now, current_state, mul_v)

def backward_recurrent(x, prev_state, U, W, V, d_mul_v, saved_graph):
    mul_u, mul_w, add_previous_now, current_state, mul_v = saved_graph
    dV, dcurrent_state = backward_multiply_gate(V, current_state, d_mul_v)
    dadd_previous_now = tanh(add_previous_now, True) * dcurrent_state.T
    dmul_w, dmul_u = backward_add_gate(mul_w, mul_u, dadd_previous_now)
    dW, dprev_state = backward_multiply_gate(W, prev_state, dmul_w)
    dU, dx = backward_multiply_gate(U, x, dmul_u)
    return (dprev_state, dU, dW, dV)
