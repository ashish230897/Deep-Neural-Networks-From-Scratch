import numpy as np
import matplotlib.pyplot as plt
from forward_pass_utils import *
from backward_pass_utils import *
from utils import *

'''
This code uses momentum as optimizer, L2 as regulizer, batch gradient descent
The code checks training and test accuracy
'''


def predict_train_accuracy(train_x, train_y, parameters):
    al, caches = L_model_forward(train_x, parameters)

    assert (al.shape == train_y.shape), "shapes are not equal of al and y_train in predict"
    count = 0
    for i in range(al.shape[1]):
        # Below two lines converts the sigmoid's output to zero or one
        if al[0, i] >= 0.5:
            al[0, i] = 1
        else:
            al[0, i] = 0

        # Increasing the count only if both outputs are same
        if al[0, i] == train_y[0, i]: count += 1

    accuracy = (count / al.shape[1]) * 100
    print("Train accuracy is %i" % (accuracy))


def L_model_forward(x, parameters):
    # caches contain every layer's tuple of both linear cache and activation cache
    # which are used during backward prop
    # Linear cache consists of previous activation
    cache_main = []
    a = x
    num_layers = len(parameters) // 2

    # Implement [LINEAR -> RELU]*(L-1).
    for l in range(1, num_layers):
        a_prev = a
        a, cache_temp = linear_activation_forward(a_prev, parameters["W" + str(l)],
                                                  parameters["b" + str(l)], "relu")
        cache_main.append(cache_temp)

    # Implement LINEAR -> Sigmoid since the current problem is a binary classification one.
    # For last layer
    al, cache_temp = linear_activation_forward(a, parameters["W" + str(num_layers)],
                                               parameters["b" + str(num_layers)], "sigmoid")
    cache_main.append(cache_temp)

    assert (al.shape == (1, x.shape[1]))

    # Finally we return last layer activation funcn and caches containing information of all layers
    return al, cache_main


def L_model_backward(al, y, caches, lambd):
    grads = {}
    num_layers = len(caches)

    m = al.shape[1]
    y = y.reshape(al.shape)

    # Below dal is input to the last layer
    # derivative of cost with respect to al is dal
    dal = - (np.divide(y, al) - np.divide(1 - y, 1 - al))

    # num_layers - 1 indexing since list goes from 0 to num_layers - 1 for length of num_layers
    current_cache = caches[num_layers - 1]
    linear_cache, activation_cache = current_cache

    # Every layer has it's own dw, db but da of previous layer i.e da_prev
    grads["dA" + str(num_layers)], grads["dW" + str(num_layers)], grads["db" + str(num_layers)] =\
                                                    linear_activation_backward(dal, current_cache,
                                                                               "sigmoid", lambd)

    # Following loop starts from l - 2 layer as last layer is l - 1 for which da_prev calculated i.e da
    for l in reversed(range(num_layers - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]

        da_prev_temp, dw_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    "relu", lambd)

        # grads is a dictionary from 1 to num_layers
        grads["dA" + str(l + 1)] = da_prev_temp
        grads["dW" + str(l + 1)] = dw_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def L_layer_model(x, y, layer_dims, learning_rate, epochs, print_cost, batch_size, load_pretrained):
    '''
        x :               The training data
        y :               The output labels
        print_cost :      If true print cost
        load_pretrained : If true, do not initialize the parameters
                          randomly but initialize from pretrained ones.
        layer_dims :      It is a list containing the number of hidden units in each layer.
    '''
    costs = []
    seed = 10
    lambd = 0.3  # regularization parameter

    # First initialize parameters for each layer
    # Initialize parameters for momentum for each layer
    parameters = initialize_parameters_he(layer_dims, load_pretrained)
    v = initialize_velocity_momentum(parameters)

    # Here we are implementing batch gradient descent
    for i in range(0, epochs):
        seed += 1
        minibatches = random_mini_batches(x, y, batch_size, seed)
        for minibatch in minibatches:
            minibatch_x, minibatch_y = minibatch

            # A single forward pass
            al, caches = L_model_forward(minibatch_x, parameters)

            # Cost computing
            cost = compute_cost(al, minibatch_y, parameters, lambd)

            # Backward Pass
            grads = L_model_backward(al, minibatch_y, caches, lambd, "softmax")

            # Update parameters
            parameters, v = update_parameters_with_momentum(parameters, grads, learning_rate, v, 0.9)

        # The below code saves the parameters after each epoch
        for k in range(1, len(layer_dims)):
            np.save("parametersW%i.npy" % (k), parameters["W" + str(k)])
            np.save("parametersb%i.npy" % (k), parameters["b" + str(k)])

        if print_cost and i % 2 is 0:
            print("Cost after epoch %i: %f" % (i, cost))
            predict_train_accuracy(x, y, parameters)
        if print_cost:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per twos)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters



# ~~~~~~~~~~~~~~~~~~~~~~~~~~Training starts here~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load data is a function that loads data
train_x_orig, train_y_orig = load_data("./Data/Train/")

train_x, train_y = preprocess(train_x_orig, train_y_orig)

print(np.shape(train_x))
print(np.shape(train_y))

# below line describes a 5-layer network
# Input layer has 12288 nodes
layer_dims = [12288, 20, 7, 5, 1]

parameters = L_layer_model(train_x, train_y, layer_dims, 0.0075, 100, True, 256, False)
