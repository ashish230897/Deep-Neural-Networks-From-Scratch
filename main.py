import numpy as np
import os
import matplotlib.pyplot as plt
import math
from utils import *
from forward_pass_utils import *
from backward_pass_utils import *
'''
This code uses momentum as optimizer, L2 as regulizer, batch gradient descent
The code checks training and test accuracy
'''

def predict_train_accuracy(train_x, train_y, parameters):
	AL, caches = L_model_forward(train_x, parameters)
	
	assert(AL.shape == train_y.shape),"shapes are not equal of al and y_train in predict"
	count = 0
	for i in range(AL.shape[1]):
		##Below two lines converts the softmax output to one hot encoding		
		max1 = np.amax(AL[:, i: i + 1])
		AL[:, i: i + 1] = 1 * (AL[:, i: i + 1] == max1)
		
		##Increasing the count only if difference is 0		
		if np.any(AL[:, i: i + 1] - train_y[:, i: i + 1]) == False:
			count += 1

	accuracy = (count/AL.shape[1]) * 100
	print("Train accuracy is %i" %(accuracy) )

def predict_test_accuracy(test_x, test_y, parameters):
	##Works similar to predict_train_accuracy	
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


def L_model_forward(X, parameters):
	##In simple words, caches contain every layer's tuple of both linear cache and activation cache which are used during backward prop
	##Linear cache consists of previous activation, current
	cache_main = []
	
	A = X
	
	##number of layers in the neural network
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


	assert(AL.shape == (2,X.shape[1]))

	#Finally we return last layer activation funcn and caches containing information of all layers
	return AL, cache_main


def L_model_backward(AL, Y, caches, regularization, lambd, last_activation):
	grads = {}
	
	##The number of layers
	L = len(caches)
	
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	##Initializing the backpropagation
	##Below dAL is input to the last layer
	##derivative of cost with respect to AL
	dAL = (-1)* np.divide(Y, AL )

	##takes output layer tuple
	current_cache = []
	
	##L-1 indexing since list goes from 0 to L-1 for length of L
	current_cache = caches[L-1]
	
	##Derivative of loss with respect to Z of o/p layer	
	dZ = AL - Y
	
	linear_cache, activation_cache = current_cache
	
	##Every layer has it's own dw, db but ba of previous layer i.e da_prev
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, linear_cache, regularization, lambd)	
	
	#following loop starts from l - 2 layer as last layer is l - 1 for which da_prev calculated i.e da
	for l in reversed(range(L-1)):
		##lth layer: (RELU -> LINEAR) gradients.
		##as caches is a list, it goes from 0 to L - 1		
		current_cache = caches[l]
		
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu", regularization, lambd)
		
		##grads is a dictionary from 1 to L
		grads["dA" + str(l + 1)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads


		

def L_layer_model(X, Y, layer_dims, learning_rate, epochs, print_cost, batch_size):
	costs = []
	seed = 10
	t = 0
	m = X.shape[1]
	
	##First initialize parameters for each layer and then also initialize parameters for momentum for each layer
	parameters = initialize_parameters_he(layer_dims)
	v = initialize_velocity_momentum(parameters)


	#Here we are implementing batch gradient descent
	for i in range(0, epochs):
		seed += 1
		minibatches = random_mini_batches(X, Y, 64, seed)
		for minibatch in minibatches:
			minibatch_X, minibatch_Y = minibatch			
			
			##A single forward pass			
			AL, caches = L_model_forward( minibatch_X, parameters)
			
			##Cost computing			
			cost = compute_cost(AL, minibatch_Y, parameters, "L2")
			
			##Backward Pass			
			grads = L_model_backward(AL, minibatch_Y, caches, "L2", 0.5, "softmax")
			
			t += 1
			##Update parameters
			parameters, v = update_parameters_with_momentum(parameters, grads, learning_rate, v, 0.9)
		
		##The below code saves the parameters after each epoch			
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

	
#main code starts here
#Load data is a function that loads data, dummy valuea are loaded
train_x_orig, train_y, test_x_orig, test_y = load_data()

#Normalizing
train_x = train_x_orig/255
test_x = test_x_orig/255


train_x, train_y = shuffle(train_x, train_y)
#below line describes a 5-layer network
#Input layer has 12288 nodes and output layer has 38 o/ps or classes
layer_dims = [1, 20, 30, 2]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

parameters = L_layer_model(train_x,train_y, layer_dims, 0.01, 10000, True, 1)


pred_train = predict_train_accuracy(train_x, train_y, parameters)

pred_test = predict_test_accuracy(test_x, test_y, parameters)

