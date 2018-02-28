import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
'''
This code uses momentum as optimizer, L2 as regulizer, batch gradient descent
The code checks training and test accuracy
'''
def initialize_parameters_he(layer_dims):	
	parameters = {}
	
	# number of layers in the network
	L = len(layer_dims)

	#He initialization works well for networks with ReLU activations.
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
	# number of layers in the neural network
	L = len(parameters) // 2

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

	#Below code computes cost for softmax function
	cross_entropy_cost = (-1/m)*np.sum( np.sum(np.multiply(Y, np.log(AL )), axis = 0 ) )	
	cost = cross_entropy_cost + L2_cost
	return cost

def compute_cost(AL, Y, parameters, regularization):
	m = Y.shape[1]

	# Compute loss from aL and y, none other cache information is needed
	cost = compute_cost_with_L2(AL, Y, parameters, 0.1)
	
	cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ()),"cost's shape is incorrect"

	return cost



def linear_backward(dZ, cache, regularization, lambd):
	#As explained earlier, we need dz of any layer before calculating other gradients of that layer
	#dz is caculated from dA , which is then used to calculate dA_prev	
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


	dZ = relu_backward(dA, activation_cache)
	dA_prev, dW, db = linear_backward(dZ, linear_cache, regularization, lambd)

	#dA_prev is output of current layer and input to layer preceding it	
	return dA_prev, dW, db


def L_model_backward(AL, Y, caches, regularization, lambd, last_activation):
	grads = {}
	L = len(caches)	# the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

	# Initializing the backpropagation
	#Below dAL is input to the last layer
	dAL = (-1)* np.divide(Y, AL )  # derivative of cost with respect to AL

	#takes output layer tuple
	current_cache = []
	current_cache = caches[L-1]
	#daL that is passed as function argument is the input to the last layer
	#dAL that is saved in grads dict is output of last layer to second last layer	
	
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


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	np.random.seed(seed)
	
	m = X.shape[1]
	mini_batches = []
	permutation = np.random.permutation(m)
	shuffled_X = X[ :, permutation]
	shuffled_Y = Y[ :, permutation].reshape((38, m))

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
		

def L_layer_model(X, Y, layer_dims, learning_rate, epochs, print_cost, batch_size):
	costs = []
	seed = 10
	t = 0
	m = X.shape[1]
	parameters = initialize_parameters_he(layer_dims)

	v = initialize_velocity_momentum(parameters)
	#Here we are implementing batch gradient descent
	for i in range(0, epochs):
		seed += 1		
		minibatches = random_mini_batches(X, Y, 64, seed)
		for minibatch in minibatches:
			minibatch_X, minibatch_Y = minibatch			
			AL, caches = L_model_forward( minibatch_X, parameters)
			cost = compute_cost(AL, minibatch_Y, parameters, "L2")
			grads = L_model_backward(AL, minibatch_Y, caches, "L2", 0.5, "softmax")
			t += 1
			parameters, v = update_parameters_with_momentum(parameters, grads, learning_rate, v, 0.9)
		#The below code saves the parameters after each epoch			
		for k in range(1, len(layer_dims)):
			np.save("parametersW%i.npy" %(k), parameters["W" + str(k)])
			np.save("parametersb%i.npy" %(k), parameters["b" + str(k)])

		if print_cost:
			print("Cost after epoch %i: %f" %(i, cost))
			print(predict_train_accuracy(X, Y, parameters))
		if print_cost:
			costs.append(cost)

	
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	return parameters



def predict_train_accuracy(train_x, train_y, parameters):
	AL, caches = L_model_forward(train_x, parameters)
	
	assert(AL.shape == train_y.shape),"shapes are not equal of al and y_train in predict"
	count = 0
	for i in range(AL.shape[1]):
		max1 = np.amax(AL[:, i: i + 1])
		AL[:, i: i + 1] = 1 * (AL[:, i: i + 1] == max1)
		if np.any(AL[:, i: i + 1] - train_y[:, i: i + 1]) == False:
			count += 1

	accuracy = (count/AL.shape[1]) * 100
	print("Train accuracy is %i" %(accuracy) )

def predict_test_accuracy(test_x, test_y, parameters):
	AL1, caches = L_model_forward(test_x, parameters)

	count1 = 0
	assert(AL1.shape == test_y.shape),"shapes are not equal of al and y_test in predict"
	
	for i in range(AL1.shape[1]):
		max2 = np.amax(AL1[:, i: i + 1])
		AL1[:, i: i + 1] = 1 * (AL1[:, i: i + 1] == max2)
		if np.any(AL1[:, i: i + 1] - test_y[:, i: i + 1]) == False:
			count1 += 1
	accuracy1 = (count1/AL1.shape[1]) * 100
	print("Test accuracy is %i" %(accuracy1) )


def shuffle(a, b):
	permutation = list(np.random.permutation(a.shape[1]))
	a = a[ :, permutation]
	b = b[ :, permutation].reshape((38,a.shape[1]))
	return a, b
	
#main code starts here
#Load data is a function that loads data
train_x_orig, train_y, test_x_orig, test_y = load_data()

#Normalizing
train_x = train_x_orig/255
test_x = test_x_orig/255


test_x, test_y = shuffle(test_x, test_y)
#below line describes a 5-layer network
#Input layer has 12288 nodes and output layer has 38 o/ps or classes
layer_dims = [12288, 500, 500, 500, 400, 300, 38]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

parameters = L_layer_model(train_x,train_y, layer_dims, 0.01, 250, True, 64)


pred_train = predict_train_accuracy(train_x, train_y, parameters)

pred_test = predict_test_accuracy(test_x, test_y, parameters)

