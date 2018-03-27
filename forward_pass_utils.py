import numpy as np
import os
import matplotlib.pyplot as plt
import math


def compute_cost_with_L2(AL, Y, parameters, lambd):
	m = Y.shape[1]
	list1 = []
	L = len(parameters) // 2
	
	##L2 regularization cost
	for i in range(1, L + 1):
		list1.append( np.sum( np.square(parameters["W" + str(i)]) ) )
	L2_cost = (lambd/(2 * m))*( np.sum(list1) )

	#Below code computes cost for softmax function
	cross_entropy_cost = (-1/m)*np.sum( np.sum(np.multiply(Y, np.log(AL )), axis = 0 ) )	
	##Total cost	
	cost = cross_entropy_cost + L2_cost
	return cost


def compute_cost(AL, Y, parameters, regularization):
	m = Y.shape[1]

	##Compute loss from AL and y, none other cache information is needed
	cost = compute_cost_with_L2(AL, Y, parameters, 0.1)
	
	##To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	cost = np.squeeze(cost)
	assert(cost.shape == ()),"cost's shape is incorrect"

	return cost


def relu(Z):
	A = Z * (Z > 0)
	activation_cache = {"Z": Z}

	assert(A.shape == Z.shape),"a's shape is not same as z in relu function"
	return A, activation_cache

def softmax(Z):
	#We subtract with maximum to avoid any overflow
	A = np.exp(Z - np.amax(Z))
	A = A/np.sum(A , axis = 0)	
	activation_cache = {"Z": Z}

	assert(A.shape == Z.shape),"a's shape is not same as z in softmax function"
	return A, activation_cache

def linear_forward(A, W, b):
	Z = np.dot(W,A) + b

	assert(Z.shape == (W.shape[0], A.shape[1])),"Shape of calculated Z is incorrect in linear_forward function"
	cache = (A, W, b)

	return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
	if activation == "relu":
		##Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		##Linear cache contains previous layer's activation output, current layer's W and b
		##Activation cache contains current layers Z
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)
	
	elif activation == "softmax":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = softmax(Z)
		
	assert (A.shape == (W.shape[0], A_prev.shape[1])),"Calculated shape of A is incorrect in linear_activation_forward "
	#we make a cache tuple and store linear and activation cache in it
	#linear_cache contains previous input, current w,b
	#activation_cache contains current z	
	cache = (linear_cache, activation_cache)

	return A, cache

