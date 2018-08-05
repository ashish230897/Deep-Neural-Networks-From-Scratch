import numpy as np


def compute_cost_with_L2(al, y, parameters, lambd):
    m = y.shape[1]  # Number of examples
    parameters_sum = []
    num_layers = len(parameters) // 2

    # L2 regularization cost
    for i in range(1, num_layers + 1):
        parameters_sum.append( np.sum( np.square(parameters["W" + str(i)]) ) )
    l2_cost = (lambd/(2 * m))*( np.sum(parameters_sum) )

    # Below code computes cost for sigmoid function
    cross_entropy_cost = (-1/m)*( np.sum( y*np.log(al) ) + np.sum( (1 - y)*np.log(1 - al) ) )

    return cross_entropy_cost + l2_cost


def compute_cost(al, y, parameters, lambd):
    # Compute loss from al and y, none other cache information is needed
    cost = compute_cost_with_L2(al, y, parameters, lambd)

    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost)
    assert(cost.shape == ()),"cost's shape is incorrect"

    return cost


def relu(z):
    a = z * (z > 0)
    activation_cache = {"Z": z}

    assert(a.shape == z.shape),"a's shape is not same as z in relu function"
    return a, activation_cache


def sigmoid(z):
    a = 1/(1 + np.exp(-1*z))
    activation_cache = {"Z": z}

    assert(a.shape == z.shape),"a's shape is not same as z in sigmoid function"
    return a, activation_cache


def linear_forward(a, w, b):
    z = np.dot(w,a) + b

    assert(z.shape == (w.shape[0], a.shape[1])),"Shape of calculated Z is incorrect in linear_forward function"
    cache = (a, w, b)

    return z, cache


def linear_activation_forward(a_prev, w, b, activation):
    # Linear cache contains previous layer's activation output, current layer's W and b
    # Activation cache contains current layer's Z

    if activation == "relu":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)

    elif activation == "sigmoid":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)

    assert (a.shape == (w.shape[0], a_prev.shape[1])),"Calculated shape of A is incorrect"

    # linear_cache contains previous input, current w,b
    # activation_cache contains current z
    cache = (linear_cache, activation_cache)

    return a, cache


