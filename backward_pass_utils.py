import numpy as np
import os
import matplotlib.pyplot as plt
import math

def relu_backward(dA, activation_cache):
	##Since dz = da * g'(z)
	Z = activation_cache["Z"]
	
	return np.multiply(dA, np.int64(Z > 0))  


def linear_activation_backward(dA, cache, activation, regularization, lambd):
	#cache is of a particular layer, this function is run for every layer to calculate dA_prev for preceding layer
	linear_cache, activation_cache = cache

	
	##To calculate dZ we need dA
	dZ = relu_backward(dA, activation_cache)
	
	##dZ of current layer is necessary to calculate every other gradient
	dA_prev, dW, db = linear_backward(dZ, linear_cache, regularization, lambd)

	##dA_prev is gradient of previous layer's output that is A_prev	
	return dA_prev, dW, db


def linear_backward(dZ, cache, regularization, lambd):
	##As explained earlier, we need dz of any layer before calculating other gradients of that layer
	##dz is caculated from dA , which is then used to calculate dA_prev	
	##Cache is linear cache that contains previous layer A and w, b of current layer	
	
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = (1/m)*(np.dot(dZ,A_prev.T)) + (lambd/m)*W
	db = (1/m)*(np.sum(dZ, axis = 1, keepdims = True))
	dA_prev = np.dot(W.T,dZ)
	
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)

	return dA_prev, dW, db

