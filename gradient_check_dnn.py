import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import bigfloat
import json

def initialize_velocity_momentum(parameters):
	v = {}	
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

	#// operator is used to divide with integral result
	L = len(parameters) // 2 # number of layers in the neural network

	# Update rule for each parameter. Use a for loop.
	for l in range(1, L + 1):
		v["dw" + str(l)] = beta*v["dw" + str(l)] + (1 - beta)*grads["dW" + str(l)]
		v["db" + str(l)] = beta*v["db" + str(l)] + (1 - beta)*grads["db" + str(l)]
		parameters["W" + str(l)] -= (learning_rate * v["dw" + str(l)])
		parameters["b" + str(l)] -= (learning_rate * v["db" + str(l)])

	return parameters, v



def initialize_parameters_he(layer_dims):	
	parameters = {}
	#current length is 5 i.e 0 to 4
	# number of layers in the network
	L = len(layer_dims)

	#He initialization works well for networks with ReLU activations.
	#parameters will be from w1 to w4 as there are 4 layers
	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l - 1])
		parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1])),"dimensions of parameter W in dict is incorrect"
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1)),"dimensions of parameter b in dict is incorrect"


	return parameters


def relu(Z):
	A = Z * (Z > 0)
	activation_cache = {"Z": Z}

	assert(A.shape == Z.shape),"a's shape is not same as z in relu function"
	return A, activation_cache


def softmax(Z):
	A = np.exp(Z - np.amax(Z))
	A = A/np.sum(A, axis = 0)
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
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
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


def L_model_forward(X, parameters):
	#In simple words, caches contain every layer's tuple of both linear cache and activation cache
	cache_main = []
	A = X
	L = len(parameters) // 2                  # number of layers in the neural network

	# Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	#we basically iterate over all the layers	
	for l in range(1, L):
		A_prev = A
		A, cache_temp = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
		cache_main.append(cache_temp)
	
	# Implement LINEAR -> Softmax. Add "cache" to the "caches" list.
	#For last layer	
	AL, cache_temp = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "softmax")
	cache_main.append(cache_temp)


	assert(AL.shape == (38,X.shape[1]))

	#Finally we return last layer activation funcn and caches containing information of all layers
	return AL, cache_main


def compute_cost_with_L2(AL, Y, parameters, lambd):
	m = Y.shape[1]
	list1 = []
	L = len(parameters) // 2
	
	for i in range(1, L + 1):
		list1.append( np.sum( np.square(parameters["W" + str(i)]) ) )
	L2_cost = (lambd/(2 * m))*( np.sum(list1) )

	cross_entropy_cost = (-1/m)*np.sum( np.sum(np.multiply(Y, np.log(AL)), axis = 0 ) )	
	cost = cross_entropy_cost + L2_cost
	return cost

def compute_cost(AL, Y, parameters, regularization):
	m = Y.shape[1]

	# Compute loss from aL and y, none other cache information is needed
	cost = compute_cost_with_L2(AL, Y, parameters, 0.5)
	
	cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ()),"cost's shape is incorrect"

	return cost

def linear_backward(dZ, cache, regularization, lambd):
	#Cache is linear cache that contains previous layer A and w, b of current layer	
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = (1/m)*(np.dot(dZ,A_prev.T)) + (lambd/m)*W
	db = (1/m)*(np.sum(dZ, axis = 1, keepdims = True))
	dA_prev = np.dot(W.T,dZ)
	
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)

	return dA_prev, dW, db


def relu_backward(dA, activation_cache):
	#Since dz = da * g'(z)
	Z = activation_cache["Z"]
	return np.multiply(dA, np.int64(Z > 0))  


def linear_activation_backward(dA, cache, activation, regularization, lambd):
	#cache is of a particular layer, this function is run for every layer to calculate dA_prev for preceding layer
	linear_cache, activation_cache = cache

	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache, regularization, lambd)

	#dA_prev is output of current layer and input to layer preceding it	
	return dA_prev, dW, db

def L_model_backward(AL, Y, caches, regularization, lambd, last_activation):
	grads = {}
	L = len(caches)	# the number of layers
	print(L)
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

	# Initializing the backpropagation
	#Below dAL is input to the last layer
	dAL = (-1)* np.divide(Y, AL)  # derivative of cost with respect to AL

	#takes output layer tuple
	current_cache = []
	current_cache = caches[L-1]

	dZ = AL - Y 
	linear_cache, activation_cache = current_cache
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, linear_cache, regularization, lambd)	
	
	#following loop starts from l - 2 layer as last layer is l - 1 for which da_prev calculated i.e da
	for l in reversed(range(L-1)):
		# lth layer: (RELU -> LINEAR) gradients.
		# Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
		#as caches is a list, it goes from 0 to L - 1		
		current_cache = caches[l]
		#For every layer we are storing that layer's dw, db and previous layer's da
		#Basically dAl is input to dAl-1	
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu", regularization, lambd)
		#grads is a dictionary from 1 to L
		grads["dA" + str(l + 1)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads


def dict_to_vector(parameters):
	final = []
	L = len(parameters) // 2
	for i in range(1 , L + 1):
		final.append(np.reshape(parameters["W" + str(i)], (-1, 1)))
		final.append(np.reshape(parameters["b" + str(i)], (-1, 1)))
	
	arr = np.array(final[0])
	for i in range(1, L*2):
		arr = np.concatenate((arr, np.array(final[i]) ), axis = 0)	
	return arr

def dict_to_vector1(gradients):
	final = []
	L = len(gradients) // 2
	for i in range(1 , L + 1):
		final.append(np.reshape(gradients["dW" + str(i)], (-1, 1)))
		final.append(np.reshape(gradients["db" + str(i)], (-1, 1)))
	
	arr = np.array(final[0])
	for i in range(1, L*2):
		arr = np.concatenate((arr, np.array(final[i]) ), axis = 0)
	return arr

def vector_to_dict(parameters_vector, parameters):
	final = {}
	count = 0
	L = len(parameters) // 2
	for i in range(1 , L + 1):
		final["W" + str(i)] = np.reshape(np.array(parameters_vector[count :  count + parameters["W" + str(i)].size, -1]) , (parameters["W" + str(i)].shape[0], parameters["W" + str(i)].shape[1]))
		count = count + parameters["W" + str(i)].size
		final["b" + str(i)] = np.reshape(np.array(parameters_vector[count :  count + parameters["b" + str(i)].size, -1] ) , (parameters["b" + str(i)].shape[0], parameters["b" + str(i)].shape[1]))
		count = count + parameters["b" + str(i)].size		
	return final	



def gradient_check(parameters, gradients, X, Y, epsilon = 1e-7):
	parameters_values = dict_to_vector(parameters)
	grad = dict_to_vector1(gradients)
	num_parameters = parameters_values.shape[0]
	J_plus = np.zeros( ( num_parameters, 1) )
	J_minus = np.zeros( ( num_parameters, 1) )
	gradapprox = np.zeros( ( num_parameters, 1) )	
	
	for i in range(num_parameters):
		thetaplus = np.copy(parameters_values)
		thetaplus[i][0] = parameters_values[i][0] + epsilon
		
		thetaminus = np.copy(parameters_values)
		thetaminus[i][0] = parameters_values[i][0] - epsilon

		AL, _ = L_model_forward(X, vector_to_dict(thetaplus, parameters))
		J_plus[i] = compute_cost(AL, Y, vector_to_dict(thetaplus, parameters), "L2")

		AL, _ = L_model_forward(X, vector_to_dict(thetaminus, parameters))
		J_minus[i] = compute_cost(AL, Y, vector_to_dict(thetaminus, parameters), "L2")
		
		gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
	numerator = np.linalg.norm(grad - gradapprox)
	denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
	difference = numerator/denominator
	if difference > 2e-7:
        	print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
	else:
        	print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")


def convert(grads):
	L = len(grads) - 6
	gradients = {}	
	for i in range(1, L + 1):
		gradients["dW" + str(i)] = grads["dW" + str(i)]
		gradients["db" + str(i)] = grads["db" + str(i)]
	return gradients

def L_layer_model(X, Y, layer_dims, learning_rate, epochs, print_cost, optimizer):
	m = X.shape[1]
	parameters = initialize_parameters_he(layer_dims)

	v = initialize_velocity_momentum(parameters)

	AL, caches = L_model_forward(X, parameters)
	cost = compute_cost(AL, Y, parameters, "L2")
	grads = L_model_backward(AL, Y, caches, "L2", 0.5, "softmax")
		
	if print_cost:
		print("Cost after epoch 0: %f" %(cost))
	gradients = convert(grads)
	difference = gradient_check(parameters, gradients, X, Y)

	return parameters



def load_data():
	x_train = np.array([ [2, 4],
              [3,5] ])
	y_train = np.zeros((38, 2))
	y_train[0,:] = 1
	x_test = np.array([ [2, 4],
              [3,5] ])
	y_test = np.zeros((38, 2))
	y_test[0,:] = 1
	
	return x_train, y_train, x_test, y_test




#main code starts here
train_x_orig, train_y, test_x_orig, test_y = load_data()


train_x = train_x_orig/255
test_x = test_x_orig/255

layer_dims = [2, 4, 4, 38]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

parameters = L_layer_model(train_x,train_y, layer_dims, 0.005, 2, True, "momentum")


