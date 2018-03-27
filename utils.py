import numpy as np
import os
import matplotlib.pyplot as plt
import math

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	np.random.seed(seed)
	
	m = X.shape[1]
	mini_batches = []
	permutation = list(np.random.permutation(m))
	shuffled_X = X[ :, permutation]
	shuffled_Y = Y[ :, permutation].reshape((2, m))

	num_complete_minibatches = math.floor(m/mini_batch_size)
	for k in range(num_complete_minibatches):
		mini_batch_X = shuffled_X[ :, k*(mini_batch_size) : (k + 1)*(mini_batch_size)]
		mini_batch_Y = shuffled_Y[ :, k*(mini_batch_size) : (k + 1)*(mini_batch_size)]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size : ]
		mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size : ]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	return mini_batches



def shuffle(a, b):
	permutation = list(np.random.permutation(a.shape[1]))
	a = a[ :, permutation]
	b = b[ :, permutation].reshape((2,a.shape[1]))
	return a, b


def initialize_parameters_he(layer_dims):	
	parameters = {}
	
	##number of layers in the network
	L = len(layer_dims)

	##He initialization works well for networks with ReLU activations.
	##As the length of layer dims goes from 0 to L-1 , we go from 1 to L-1 as 0th layer is i/p layer
	##The initialization is done this way to avoid exploding and diminishing of output of each layer	
	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l - 1])
		parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1])),"dimensions of parameter W in dict is incorrect"
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1)),"dimensions of parameter b in dict is incorrect"
	return parameters


def initialize_velocity_momentum(parameters):
	v = {}
	##Here we length of parameters is W of each layer plus b of each layer and so we divide the total length by 2
	L = len(parameters) // 2
	
	for i in range(1, L + 1):
		v["dw" + str(i)] = np.zeros((parameters["W" + str(i)].shape[0], parameters["W" + str(i)].shape[1]))
		v["db" + str(i)] = np.zeros((parameters["b" + str(i)].shape[0], parameters["b" + str(i)].shape[1]))
		assert(v["dw" + str(i)].shape == parameters["W" + str(i)].shape),"shape of vdw is not same as parameters in initialize momentum"
		assert(v["db" + str(i)].shape == parameters["b" + str(i)].shape),"shape of vdb is not same as parameters in initialize momentum"
	
	return v


def update_parameters_with_momentum(parameters, grads, learning_rate, v, beta):
	"""
	Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient 	descent will "oscillate" toward convergence. Using momentum can reduce these oscillations.

	Momentum takes into account the past gradients to smooth out the update. We will store the 'direction' of the previous gradients in the variable  vv . Formally, this will be the exponentially 	weighted average of the gradient on previous steps. You can also think of  vv  as the "velocity" of a ball rolling downhill, building up speed (and momentum) according to the direction of the gradient/slope of the hill.
	"""

	##//operator is used to divide with integral result
	L = len(parameters) // 2

	##Update rule for each parameter. Use a for loop.
	for l in range(1, L + 1):
		v["dw" + str(l)] = beta*v["dw" + str(l)] + (1 - beta)*grads["dW" + str(l)]
		v["db" + str(l)] = beta*v["db" + str(l)] + (1 - beta)*grads["db" + str(l)]
		parameters["W" + str(l)] -= (learning_rate * v["dw" + str(l)])
		parameters["b" + str(l)] -= (learning_rate * v["db" + str(l)])

	return parameters, v




def load_data():
	x_train = np.array([ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ])
	y_train = np.zeros((2, 10))
	
	for i in range(x_train.shape[1]):
		if x_train[0][i] % 2 == 0:
			y_train[0][i] = 1
			y_train[1][i] = 0
		else:
			y_train[0][i] = 0
			y_train[1][i] = 1
				
	x_test = np.array([ [99, 100, 101] ])
	y_test = np.zeros((2, 3))
	
	for i in range(x_test.shape[1]):
		if x_test[0][i] % 2 == 0:
			y_test[0][i] = 1
			y_test[1][i] = 0
		else:
			y_test[0][i] = 0
			y_test[1][i] = 1
	
	return x_train, y_train, x_test, y_test




